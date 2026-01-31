"""
Visual encoding utilities for rendering math expressions and numbers as pixel arrays.

Provides brain-like sensory input: convert symbols to images (like human vision),
no tokenization - network learns from raw pixels.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple


class VisualEncoder:
    """
    Encodes math expressions, digits, and quantities as pixel arrays.

    No tokenization - everything is rendered as images, just like human vision.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (28, 28),
        font_size: int = 20,
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize visual encoder.

        Args:
            image_size: (width, height) of rendered images
            font_size: Font size for text rendering
            dtype: PyTorch dtype for output tensors
        """
        self.image_size = image_size
        self.font_size = font_size
        self.dtype = dtype

        # Try to load a monospace font, fall back to default
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
        except:
            # Fallback to default font
            self.font = ImageFont.load_default()

    def render_text(self, text: str) -> torch.Tensor:
        """
        Render text as grayscale pixel array.

        Args:
            text: Text to render (e.g., "3", "+", "5")

        Returns:
            Flattened pixel array as tensor (width * height,)
        """
        # Create blank white image
        img = Image.new('L', self.image_size, color=255)
        draw = ImageDraw.Draw(img)

        # Get text bounding box for centering
        bbox = draw.textbbox((0, 0), text, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Center text
        x = (self.image_size[0] - text_width) // 2
        y = (self.image_size[1] - text_height) // 2

        # Draw text in black
        draw.text((x, y), text, fill=0, font=self.font)

        # Convert to numpy array and normalize to [0, 1]
        pixels = np.array(img, dtype=np.float32) / 255.0

        # Invert: black background, white text (standard for MNIST-style)
        pixels = 1.0 - pixels

        # Flatten and convert to tensor
        return torch.from_numpy(pixels.flatten()).to(self.dtype)

    def render_dots(self, count: int, max_count: int = 9) -> torch.Tensor:
        """
        Render dots representing a quantity.

        Args:
            count: Number of dots to render (0-max_count)
            max_count: Maximum number of dots (determines layout)

        Returns:
            Flattened pixel array as tensor (width * height,)
        """
        # Create blank black image
        img = Image.new('L', self.image_size, color=0)
        draw = ImageDraw.Draw(img)

        if count == 0:
            # Empty image for zero
            pixels = np.zeros(self.image_size, dtype=np.float32)
        else:
            # Arrange dots in a grid
            # For 1-9: use 3x3 grid, for 10+: use 4x3 grid
            if max_count <= 9:
                grid_cols = 3
                grid_rows = 3
            else:
                grid_cols = 4
                grid_rows = 3

            # Calculate dot positions
            dot_radius = min(self.image_size) // (2 * max(grid_cols, grid_rows) + 2)
            spacing_x = self.image_size[0] // (grid_cols + 1)
            spacing_y = self.image_size[1] // (grid_rows + 1)

            # Draw dots
            dots_drawn = 0
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if dots_drawn >= count:
                        break

                    x = spacing_x * (col + 1)
                    y = spacing_y * (row + 1)

                    # Draw filled circle (dot)
                    draw.ellipse(
                        [(x - dot_radius, y - dot_radius),
                         (x + dot_radius, y + dot_radius)],
                        fill=255  # White dot
                    )

                    dots_drawn += 1

                if dots_drawn >= count:
                    break

            # Convert to numpy array and normalize
            pixels = np.array(img, dtype=np.float32) / 255.0

        # Flatten and convert to tensor
        return torch.from_numpy(pixels.flatten()).to(self.dtype)

    def render_digit_with_dots(self, digit: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render a digit and its corresponding dot representation.

        For teaching digit semantics: "3" → ●●●

        Args:
            digit: Digit to render (0-9)

        Returns:
            Tuple of (digit_pixels, dots_pixels)
        """
        digit_img = self.render_text(str(digit))
        dots_img = self.render_dots(digit)
        return digit_img, dots_img

    def render_expression_sequence(
        self,
        expression: str,
        include_result: bool = True
    ) -> List[torch.Tensor]:
        """
        Render math expression as sequence of images (character by character).

        Args:
            expression: Expression like "2+3=5"
            include_result: Whether to include result in sequence

        Returns:
            List of pixel tensors, one per character/symbol
        """
        sequence = []

        # Split by equals sign if present
        if '=' in expression:
            parts = expression.split('=')
            problem = parts[0].strip()
            result = parts[1].strip() if len(parts) > 1 else ""

            # Render problem characters
            for char in problem:
                if char.strip():  # Skip whitespace
                    sequence.append(self.render_text(char))

            # Render equals sign
            sequence.append(self.render_text('='))

            # Render result if requested
            if include_result and result:
                for char in result:
                    if char.strip():
                        sequence.append(self.render_text(char))
        else:
            # No equals sign, just render all characters
            for char in expression:
                if char.strip():
                    sequence.append(self.render_text(char))

        return sequence

    def get_input_size(self) -> int:
        """Get size of flattened pixel array (for network input layer)."""
        return self.image_size[0] * self.image_size[1]


# Example usage functions
def create_digit_semantics_dataset(
    digits: List[int] = list(range(10)),
    encoder: VisualEncoder = None
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create dataset for teaching digit semantics.

    Args:
        digits: List of digits to include (default: 0-9)
        encoder: VisualEncoder instance (creates new if None)

    Returns:
        List of (digit_image, dots_image) pairs
    """
    if encoder is None:
        encoder = VisualEncoder()

    dataset = []
    for digit in digits:
        digit_img, dots_img = encoder.render_digit_with_dots(digit)
        dataset.append((digit_img, dots_img))

    return dataset


def create_addition_dataset(
    max_operand: int = 9,
    encoder: VisualEncoder = None
) -> List[Tuple[List[torch.Tensor], torch.Tensor]]:
    """
    Create dataset for addition learning.

    Args:
        max_operand: Maximum value for operands
        encoder: VisualEncoder instance

    Returns:
        List of (expression_sequence, result_image) pairs
    """
    if encoder is None:
        encoder = VisualEncoder()

    dataset = []

    for a in range(max_operand + 1):
        for b in range(max_operand + 1):
            result = a + b
            if result <= 99:  # Keep results to 2 digits
                # Render expression sequence: "2", "+", "3", "="
                expr = f"{a}+{b}="
                sequence = encoder.render_expression_sequence(expr, include_result=False)

                # Render result
                result_img = encoder.render_text(str(result))

                dataset.append((sequence, result_img))

    return dataset
