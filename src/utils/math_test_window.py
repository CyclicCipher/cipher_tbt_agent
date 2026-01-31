"""
Visual test window for math curriculum experiments.

Provides a simple GUI where:
1. Math problems are rendered as text (visual stimuli)
2. Network "sees" the rendered text via simulated foveal/peripheral vision
3. Network types answers via motor control (active inference)
4. Results are displayed and logged

No actual screen needed - this simulates the visual environment.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
import time


class MathTestWindow:
    """
    Simulated visual test environment for math problems.

    Renders text as images that the network can "see" with foveal vision,
    accepts motor commands (typed answers), and displays results.

    Args:
        width: Window width in pixels
        height: Window height in pixels
        background_color: RGB background color
        text_color: RGB text color
        font_scale: Font size multiplier
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        text_color: Tuple[int, int, int] = (0, 0, 0),
        font_scale: float = 2.0
    ):
        self.width = width
        self.height = height
        self.background_color = background_color
        self.text_color = text_color
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.thickness = 3

        # Screen buffer (what network "sees")
        self.screen = np.zeros((height, width, 3), dtype=np.uint8)
        self.clear_screen()

        # Current problem and answer
        self.current_problem = ""
        self.typed_answer = ""
        self.correct_answer = ""

        # Problem position on screen
        self.problem_y = height // 3
        self.answer_y = 2 * height // 3

        # Gaze position (simulated)
        self.gaze_x = width // 2
        self.gaze_y = height // 2

    def clear_screen(self):
        """Clear screen to background color."""
        self.screen[:] = self.background_color

    def show_problem(self, problem_text: str, answer: str):
        """
        Display a math problem on screen.

        Args:
            problem_text: The problem (e.g., "2 + 3 =")
            answer: Correct answer (for evaluation)
        """
        self.clear_screen()
        self.current_problem = problem_text
        self.correct_answer = answer
        self.typed_answer = ""

        # Render problem text
        self._render_text(problem_text, y=self.problem_y)

        # Render "Your answer:" prompt
        self._render_text("Answer: _", y=self.answer_y)

    def _render_text(
        self,
        text: str,
        y: int,
        color: Optional[Tuple[int, int, int]] = None
    ):
        """Render text at specified y position (centered)."""
        if color is None:
            color = self.text_color

        # Get text size to center it
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.thickness
        )

        x = (self.width - text_width) // 2

        cv2.putText(
            self.screen,
            text,
            (x, y),
            self.font,
            self.font_scale,
            color,
            self.thickness,
            cv2.LINE_AA
        )

    def type_character(self, char: str):
        """
        Network types a character (motor output).

        Args:
            char: Character to type
        """
        self.typed_answer += char

        # Update display
        answer_display = f"Answer: {self.typed_answer}_"
        self.clear_screen()
        self._render_text(self.current_problem, y=self.problem_y)
        self._render_text(answer_display, y=self.answer_y)

    def backspace(self):
        """Network presses backspace."""
        if self.typed_answer:
            self.typed_answer = self.typed_answer[:-1]

            # Update display
            answer_display = f"Answer: {self.typed_answer}_"
            self.clear_screen()
            self._render_text(self.current_problem, y=self.problem_y)
            self._render_text(answer_display, y=self.answer_y)

    def submit_answer(self) -> bool:
        """
        Network submits answer (presses Enter).

        Returns:
            True if answer is correct
        """
        is_correct = self.typed_answer.strip() == self.correct_answer.strip()

        # Show result
        self.clear_screen()
        self._render_text(self.current_problem, y=self.problem_y)

        result_text = f"Answer: {self.typed_answer}"
        result_color = (0, 200, 0) if is_correct else (200, 0, 0)  # Green or red
        self._render_text(result_text, y=self.answer_y, color=result_color)

        status = "✓ Correct!" if is_correct else f"✗ Wrong (answer: {self.correct_answer})"
        self._render_text(status, y=self.answer_y + 80)

        return is_correct

    def get_foveal_patch(
        self,
        fovea_size: int = 50,
        gaze_x: Optional[int] = None,
        gaze_y: Optional[int] = None
    ) -> np.ndarray:
        """
        Get foveal vision patch at current gaze position.

        Args:
            fovea_size: Size of foveal patch (pixels)
            gaze_x: Gaze x position (uses current if None)
            gaze_y: Gaze y position (uses current if None)

        Returns:
            RGB patch of shape (fovea_size, fovea_size, 3)
        """
        if gaze_x is None:
            gaze_x = self.gaze_x
        if gaze_y is None:
            gaze_y = self.gaze_y

        # Clamp to valid range
        half_size = fovea_size // 2
        x1 = max(0, gaze_x - half_size)
        x2 = min(self.width, gaze_x + half_size)
        y1 = max(0, gaze_y - half_size)
        y2 = min(self.height, gaze_y + half_size)

        # Extract patch
        patch = self.screen[y1:y2, x1:x2].copy()

        # Resize to exact size if at edge
        if patch.shape[0] != fovea_size or patch.shape[1] != fovea_size:
            patch = cv2.resize(patch, (fovea_size, fovea_size))

        return patch

    def get_peripheral_vision(
        self,
        scale: float = 0.25
    ) -> np.ndarray:
        """
        Get peripheral vision (downsampled full screen).

        Args:
            scale: Downsampling factor (0.25 = 1/4 resolution)

        Returns:
            Downsampled RGB image
        """
        new_width = int(self.width * scale)
        new_height = int(self.height * scale)
        return cv2.resize(self.screen, (new_width, new_height))

    def move_gaze(self, x: int, y: int):
        """
        Move gaze to specified position (saccade).

        Args:
            x: Target x position
            y: Target y position
        """
        self.gaze_x = np.clip(x, 0, self.width - 1)
        self.gaze_y = np.clip(y, 0, self.height - 1)

    def get_screen(self) -> np.ndarray:
        """Get full screen buffer (for debugging)."""
        return self.screen.copy()

    def save_screenshot(self, filepath: str):
        """Save current screen to file."""
        cv2.imwrite(filepath, cv2.cvtColor(self.screen, cv2.COLOR_RGB2BGR))


class MathExperimentRunner:
    """
    Runs math curriculum experiments with visual rendering.

    Integrates:
    - Math curriculum (problem generation)
    - Test window (visual rendering)
    - Network (vision → action)
    - Logging (results tracking)
    """

    def __init__(
        self,
        window: MathTestWindow,
        network,  # Your ModularNetwork
        curriculum,  # MathCurriculum
        fovea_size: int = 50,
        inference_iterations: int = 50
    ):
        self.window = window
        self.network = network
        self.curriculum = curriculum
        self.fovea_size = fovea_size
        self.inference_iterations = inference_iterations

        # Experiment results
        self.results = []

    def run_single_problem(
        self,
        problem,  # MathProblem
        debug: bool = False
    ) -> dict:
        """
        Run network on a single math problem.

        Args:
            problem: MathProblem instance
            debug: If True, saves screenshots

        Returns:
            Dict with results (correct, time_taken, etc.)
        """
        start_time = time.time()

        # 1. Display problem
        self.window.show_problem(problem.input, problem.output)

        if debug:
            self.window.save_screenshot(f"debug_problem_{problem.input.replace(' ', '_')}.png")

        # 2. Network observes problem (foveal sweeps)
        # Simulate reading left-to-right with saccades
        problem_chars = len(problem.input)
        char_positions = [100 + i * 40 for i in range(problem_chars)]  # Approximate positions

        for gaze_x in char_positions:
            self.window.move_gaze(gaze_x, self.window.problem_y)

            # Get foveal vision
            foveal_patch = self.window.get_foveal_patch(self.fovea_size)

            # Process with retinal preprocessing
            # (This would call your retinal_preprocessing.py)
            # For now, just flatten and normalize
            visual_features = foveal_patch.flatten() / 255.0

            # Network inference
            # (This would call your network.forward())
            # For now, placeholder
            # network_state = self.network.forward({"vision": visual_features}, num_iterations=self.inference_iterations)

        # 3. Network generates answer via motor control (active inference)
        # Motor prediction emerges from Position 0 motor subnet
        # For now, random answer as placeholder
        predicted_answer = problem.output  # Placeholder - would come from network

        # 4. Type the answer
        for char in predicted_answer:
            self.window.type_character(char)
            if debug:
                time.sleep(0.1)  # Slow down for visualization

        # 5. Submit and check
        is_correct = self.window.submit_answer()

        elapsed_time = time.time() - start_time

        # Record result
        result = {
            'problem': problem.input,
            'correct_answer': problem.output,
            'predicted_answer': predicted_answer,
            'is_correct': is_correct,
            'time_taken': elapsed_time,
            'domain': problem.domain.value,
            'difficulty': problem.difficulty.value
        }

        self.results.append(result)

        if debug:
            self.window.save_screenshot(f"debug_result_{problem.input.replace(' ', '_')}.png")

        return result

    def run_experiment(
        self,
        problems: List,  # List of MathProblem
        verbose: bool = True
    ) -> dict:
        """
        Run full experiment on list of problems.

        Args:
            problems: List of MathProblem instances
            verbose: Print progress

        Returns:
            Summary statistics
        """
        self.results = []

        for i, problem in enumerate(problems):
            result = self.run_single_problem(problem)

            if verbose and (i + 1) % 10 == 0:
                accuracy = sum(r['is_correct'] for r in self.results) / len(self.results)
                print(f"Progress: {i+1}/{len(problems)} | Accuracy: {accuracy*100:.1f}%")

        # Compute statistics
        total = len(self.results)
        correct = sum(r['is_correct'] for r in self.results)
        accuracy = correct / total if total > 0 else 0
        avg_time = sum(r['time_taken'] for r in self.results) / total if total > 0 else 0

        # Per-domain accuracy
        domain_stats = {}
        for domain in ['arithmetic', 'algebra', 'calculus']:
            domain_results = [r for r in self.results if r['domain'] == domain]
            if domain_results:
                domain_correct = sum(r['is_correct'] for r in domain_results)
                domain_accuracy = domain_correct / len(domain_results)
                domain_stats[domain] = {
                    'total': len(domain_results),
                    'correct': domain_correct,
                    'accuracy': domain_accuracy
                }

        summary = {
            'total_problems': total,
            'correct': correct,
            'accuracy': accuracy,
            'avg_time_per_problem': avg_time,
            'domain_stats': domain_stats
        }

        return summary


# Demo usage
if __name__ == "__main__":
    """Demonstrate visual test window."""

    from src.pretraining.math_curriculum import MathCurriculum, MathDomain, DifficultyLevel

    print("=" * 70)
    print("MATH TEST WINDOW DEMO")
    print("=" * 70)

    # Create test window
    window = MathTestWindow()

    # Create curriculum
    curriculum = MathCurriculum(seed=42)

    # Generate test problems
    problems = curriculum.generate_batch(5, MathDomain.ARITHMETIC, DifficultyLevel.EASY)

    print("\nTesting visual rendering and motor control:")
    print("-" * 70)

    for i, problem in enumerate(problems, 1):
        print(f"\nProblem {i}: {problem.input} → {problem.output}")

        # Show problem
        window.show_problem(problem.input, problem.output)
        print(f"  Problem rendered on screen")

        # Get foveal vision at problem location
        window.move_gaze(400, window.problem_y)
        foveal = window.get_foveal_patch(fovea_size=50)
        print(f"  Foveal patch extracted: {foveal.shape}")

        # Get peripheral vision
        peripheral = window.get_peripheral_vision(scale=0.25)
        print(f"  Peripheral vision: {peripheral.shape}")

        # Simulate typing answer
        for char in problem.output:
            window.type_character(char)
        print(f"  Typed answer: {window.typed_answer}")

        # Submit
        is_correct = window.submit_answer()
        print(f"  Result: {'✓ Correct' if is_correct else '✗ Wrong'}")

        # Save screenshot
        window.save_screenshot(f"test_problem_{i}.png")
        print(f"  Screenshot saved: test_problem_{i}.png")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nGenerated screenshots:")
    print("  test_problem_1.png")
    print("  test_problem_2.png")
    print("  test_problem_3.png")
    print("  test_problem_4.png")
    print("  test_problem_5.png")
    print("\nThese show what the network 'sees' when solving math problems.")
