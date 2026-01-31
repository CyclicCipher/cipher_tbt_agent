"""
Keyboard output system for discrete character actions.

The agent produces discrete actions by selecting characters from a virtual keyboard.
Uses softmax over character vocabulary to produce probability distribution.

For math tasks:
- Digits: 0-9
- Operators: + - * / =
- Special: space, backspace, enter

Agent learns to "type" responses by predicting character sequences.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class KeyboardVocabulary:
    """
    Defines the character vocabulary for keyboard output.

    Maps between characters, indices, and one-hot encodings.
    """

    def __init__(self, characters: List[str]):
        """
        Initialize vocabulary.

        Args:
            characters: List of characters in vocabulary
        """
        self.characters = characters
        self.char_to_idx = {char: idx for idx, char in enumerate(characters)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(characters)

    def encode(self, char: str) -> int:
        """Map character to index."""
        return self.char_to_idx.get(char, 0)  # Default to first char if unknown

    def decode(self, idx: int) -> str:
        """Map index to character."""
        return self.idx_to_char.get(idx, self.characters[0])

    def one_hot(self, char: str) -> np.ndarray:
        """Convert character to one-hot vector."""
        vec = np.zeros(self.vocab_size, dtype=np.float32)
        vec[self.encode(char)] = 1.0
        return vec

    def from_one_hot(self, vec: np.ndarray) -> str:
        """Convert one-hot (or soft) vector to character."""
        idx = np.argmax(vec)
        return self.decode(idx)


# Standard vocabularies for different tasks

DIGITS_VOCAB = KeyboardVocabulary(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

MATH_VOCAB = KeyboardVocabulary([
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Digits
    '+', '-', '*', '/', '=',                            # Operators
    ' ', '\b', '\n'                                     # Special: space, backspace, enter
])

ASTERISK_VOCAB = KeyboardVocabulary(['*'])  # For counting tasks (e.g., "5" → "*****")


class KeyboardOutput(nn.Module):
    """
    Keyboard output layer for discrete character actions.

    Converts network state to character probabilities via softmax.
    Agent selects characters to "type" responses.
    """

    def __init__(
        self,
        input_size: int,
        vocabulary: KeyboardVocabulary,
        temperature: float = 1.0,
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize keyboard output layer.

        Args:
            input_size: Size of input from network (top layer state)
            vocabulary: Character vocabulary
            temperature: Softmax temperature (higher = more exploration)
            dtype: PyTorch dtype
        """
        super().__init__()

        self.vocabulary = vocabulary
        self.temperature = temperature
        self.dtype = dtype

        # Linear projection: network state → character logits
        self.output_projection = nn.Linear(
            input_size,
            vocabulary.vocab_size,
            dtype=dtype
        )

        # Initialize with small weights (prevents initial bias)
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.1)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, network_state: torch.Tensor) -> torch.Tensor:
        """
        Compute character probabilities from network state.

        Args:
            network_state: Top layer state (input_size,)

        Returns:
            Character probabilities (vocab_size,)
        """
        # Project to logits
        logits = self.output_projection(network_state)

        # Apply temperature and softmax
        probs = torch.softmax(logits / self.temperature, dim=0)

        return probs

    def sample_character(self, network_state: torch.Tensor) -> str:
        """
        Sample a character from the probability distribution.

        Args:
            network_state: Top layer state

        Returns:
            Sampled character
        """
        probs = self.forward(network_state)

        # Sample from distribution
        idx = torch.multinomial(probs, num_samples=1).item()

        return self.vocabulary.decode(idx)

    def greedy_character(self, network_state: torch.Tensor) -> str:
        """
        Select most probable character (greedy decoding).

        Args:
            network_state: Top layer state

        Returns:
            Most probable character
        """
        probs = self.forward(network_state)
        idx = torch.argmax(probs).item()
        return self.vocabulary.decode(idx)

    def get_target_logits(self, target_char: str) -> torch.Tensor:
        """
        Get target distribution for supervised learning.

        Args:
            target_char: Target character

        Returns:
            One-hot target distribution (vocab_size,)
        """
        target = torch.zeros(self.vocabulary.vocab_size, dtype=self.dtype)
        target[self.vocabulary.encode(target_char)] = 1.0
        return target


class KeyboardSequenceGenerator:
    """
    Generates sequences of characters from network predictions.

    Used for testing and inference.
    """

    def __init__(
        self,
        keyboard_output: KeyboardOutput,
        max_length: int = 20,
        stop_char: str = '\n'
    ):
        """
        Initialize sequence generator.

        Args:
            keyboard_output: KeyboardOutput module
            max_length: Maximum sequence length
            stop_char: Character that ends sequence (e.g., newline)
        """
        self.keyboard_output = keyboard_output
        self.max_length = max_length
        self.stop_char = stop_char

    def generate_sequence(
        self,
        network_forward_fn,
        initial_input: torch.Tensor,
        sampling: bool = False
    ) -> List[str]:
        """
        Generate a sequence of characters.

        Args:
            network_forward_fn: Function that takes input and returns top layer state
            initial_input: Initial input to network
            sampling: If True, sample characters. If False, use greedy decoding.

        Returns:
            List of characters
        """
        sequence = []
        current_input = initial_input

        for _ in range(self.max_length):
            # Forward through network
            top_state = network_forward_fn(current_input)

            # Generate character
            if sampling:
                char = self.keyboard_output.sample_character(top_state)
            else:
                char = self.keyboard_output.greedy_character(top_state)

            sequence.append(char)

            # Stop if end character
            if char == self.stop_char:
                break

            # TODO: Update current_input with character feedback
            # For now, just continue with same input
            # (In full implementation, character would be fed back as input)

        return sequence


def sequence_to_string(sequence: List[str]) -> str:
    """Convert character sequence to string."""
    return ''.join(sequence).replace('\n', '').replace('\b', '')


def count_asterisks(sequence: List[str]) -> int:
    """Count number of asterisks in sequence (for counting tasks)."""
    return sum(1 for char in sequence if char == '*')
