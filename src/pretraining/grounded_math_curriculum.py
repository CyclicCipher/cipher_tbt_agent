"""
Grounded math curriculum with visual quantities.

Implements proper mathematical learning progression:
1. Digit recognition (visual patterns 0-9)
2. Digit-to-quantity mapping (3 = ●●●)
3. Addition with quantities (●●● + ●● = ●●●●●, with digits)
4. Addition with digits only (abstraction)
5. Repeat for multiplication, division

This teaches what numbers MEAN, not just symbol manipulation.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
from src.pretraining.math_curriculum import MathProblem, MathDomain, DifficultyLevel


class QuantityRenderer:
    """
    Renders visual quantities (dots, objects) for grounded math learning.

    Teaches network that "3" means THREE objects, not just a symbol.
    """

    def __init__(
        self,
        object_size: int = 20,
        spacing: int = 25,
        canvas_width: int = 800,
        canvas_height: int = 200,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        object_color: Tuple[int, int, int] = (0, 0, 0)
    ):
        self.object_size = object_size
        self.spacing = spacing
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.background_color = background_color
        self.object_color = object_color

    def render_quantity(
        self,
        quantity: int,
        arrangement: str = "row"
    ) -> np.ndarray:
        """
        Render visual quantity as dots/circles.

        Args:
            quantity: How many objects (0-20)
            arrangement: "row", "grid", or "random"

        Returns:
            RGB image (canvas_height, canvas_width, 3)
        """
        canvas = np.full(
            (self.canvas_height, self.canvas_width, 3),
            self.background_color,
            dtype=np.uint8
        )

        if quantity == 0:
            return canvas

        if arrangement == "row":
            positions = self._arrange_row(quantity)
        elif arrangement == "grid":
            positions = self._arrange_grid(quantity)
        elif arrangement == "random":
            positions = self._arrange_random(quantity)
        else:
            raise ValueError(f"Unknown arrangement: {arrangement}")

        # Draw circles at each position
        for x, y in positions:
            cv2.circle(
                canvas,
                (int(x), int(y)),
                self.object_size // 2,
                self.object_color,
                -1  # Filled
            )

        return canvas

    def _arrange_row(self, quantity: int) -> List[Tuple[float, float]]:
        """Arrange objects in a horizontal row."""
        positions = []
        y = self.canvas_height // 2

        # Center the row
        total_width = quantity * self.spacing
        start_x = (self.canvas_width - total_width) // 2

        for i in range(quantity):
            x = start_x + i * self.spacing + self.spacing // 2
            positions.append((x, y))

        return positions

    def _arrange_grid(self, quantity: int) -> List[Tuple[float, float]]:
        """Arrange objects in a grid (for larger quantities)."""
        positions = []

        # Determine grid dimensions
        cols = int(np.ceil(np.sqrt(quantity)))
        rows = int(np.ceil(quantity / cols))

        # Center the grid
        grid_width = cols * self.spacing
        grid_height = rows * self.spacing
        start_x = (self.canvas_width - grid_width) // 2
        start_y = (self.canvas_height - grid_height) // 2

        for i in range(quantity):
            row = i // cols
            col = i % cols
            x = start_x + col * self.spacing + self.spacing // 2
            y = start_y + row * self.spacing + self.spacing // 2
            positions.append((x, y))

        return positions

    def _arrange_random(self, quantity: int) -> List[Tuple[float, float]]:
        """Arrange objects randomly (for variety)."""
        positions = []
        margin = self.object_size

        for _ in range(quantity):
            x = np.random.randint(margin, self.canvas_width - margin)
            y = np.random.randint(margin, self.canvas_height - margin)
            positions.append((x, y))

        return positions

    def render_digit_with_quantity(
        self,
        digit: int,
        font_scale: float = 3.0
    ) -> np.ndarray:
        """
        Render digit with corresponding quantity below.

        Example: "3" with ●●● below it

        Args:
            digit: Digit to render (0-9)
            font_scale: Size of digit text

        Returns:
            RGB image showing digit and quantity
        """
        # Create canvas
        canvas = np.full(
            (self.canvas_height + 200, self.canvas_width, 3),
            self.background_color,
            dtype=np.uint8
        )

        # Render digit at top
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(digit)
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, 3
        )

        x = (self.canvas_width - text_width) // 2
        y = text_height + 50

        cv2.putText(
            canvas,
            text,
            (x, y),
            font,
            font_scale,
            self.object_color,
            3,
            cv2.LINE_AA
        )

        # Render quantity below (offset by 150 pixels down)
        quantity_img = self.render_quantity(digit, arrangement="row")

        # Paste quantity below digit
        y_start = 150
        canvas[y_start:y_start + self.canvas_height, :] = quantity_img

        return canvas

    def render_addition_with_quantities(
        self,
        a: int,
        b: int,
        show_result: bool = False
    ) -> np.ndarray:
        """
        Render addition problem with visual quantities.

        Example:
        ●●● + ●● = ?
         3  +  2  = ?

        Or with result:
        ●●● + ●● = ●●●●●
         3  +  2  =   5

        Args:
            a: First addend
            b: Second addend
            show_result: Whether to show the answer

        Returns:
            RGB image showing addition with quantities
        """
        # Create large canvas
        canvas_height = 400
        canvas = np.full(
            (canvas_height, self.canvas_width, 3),
            self.background_color,
            dtype=np.uint8
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 3

        # Render first quantity
        qty1 = self.render_quantity(a, arrangement="row")
        canvas[50:50 + self.canvas_height, 50:50 + 400] = cv2.resize(
            qty1, (400, self.canvas_height)
        )

        # Render first digit below
        cv2.putText(canvas, str(a), (200, 280), font, font_scale, self.object_color, thickness, cv2.LINE_AA)

        # Plus sign
        cv2.putText(canvas, "+", (320, 150), font, font_scale, self.object_color, thickness, cv2.LINE_AA)
        cv2.putText(canvas, "+", (320, 280), font, font_scale, self.object_color, thickness, cv2.LINE_AA)

        # Render second quantity
        qty2 = self.render_quantity(b, arrangement="row")
        canvas[50:50 + self.canvas_height, 380:380 + 200] = cv2.resize(
            qty2, (200, self.canvas_height)
        )

        # Render second digit below
        cv2.putText(canvas, str(b), (450, 280), font, font_scale, self.object_color, thickness, cv2.LINE_AA)

        # Equals sign
        cv2.putText(canvas, "=", (550, 150), font, font_scale, self.object_color, thickness, cv2.LINE_AA)
        cv2.putText(canvas, "=", (550, 280), font, font_scale, self.object_color, thickness, cv2.LINE_AA)

        if show_result:
            result = a + b

            # Render result quantity
            qty_result = self.render_quantity(result, arrangement="row")
            canvas[50:50 + self.canvas_height, 620:620 + 150] = cv2.resize(
                qty_result, (150, self.canvas_height)
            )

            # Render result digit
            cv2.putText(canvas, str(result), (670, 280), font, font_scale, self.object_color, thickness, cv2.LINE_AA)
        else:
            # Question mark
            cv2.putText(canvas, "?", (670, 150), font, font_scale, self.object_color, thickness, cv2.LINE_AA)
            cv2.putText(canvas, "?", (670, 280), font, font_scale, self.object_color, thickness, cv2.LINE_AA)

        return canvas


class GroundedMathCurriculum:
    """
    Math curriculum with grounded learning progression.

    Phase 1: Digit recognition (visual patterns)
    Phase 2: Digit-quantity mapping (3 = ●●●)
    Phase 3: Addition with quantities (●●● + ●● = ●●●●●)
    Phase 4: Addition with digits only
    Phase 5: Multiplication with quantities
    Phase 6: Multiplication with digits only
    Phase 7: Division with quantities
    Phase 8: Division with digits only
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize grounded curriculum.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        self.renderer = QuantityRenderer()

    def generate_digit_recognition_dataset(
        self,
        samples_per_digit: int = 100
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Phase 1: Digit recognition.

        Learn visual patterns of digits 0-9.

        Args:
            samples_per_digit: How many examples per digit

        Returns:
            List of (image, label) tuples
        """
        dataset = []

        for digit in range(10):
            for _ in range(samples_per_digit):
                # Render digit with slight variations
                img = self._render_digit_varied(digit)
                dataset.append((img, digit))

        # Shuffle
        np.random.shuffle(dataset)

        return dataset

    def _render_digit_varied(self, digit: int) -> np.ndarray:
        """Render digit with slight variations (position, size)."""
        canvas = np.full((100, 100, 3), 255, dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Slight variation in size
        font_scale = 2.0 + np.random.uniform(-0.3, 0.3)

        # Slight variation in position
        x_offset = np.random.randint(-5, 5)
        y_offset = np.random.randint(-5, 5)

        text = str(digit)
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, 3)

        x = (100 - text_width) // 2 + x_offset
        y = (100 + text_height) // 2 + y_offset

        cv2.putText(canvas, text, (x, y), font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)

        return canvas

    def generate_quantity_mapping_dataset(
        self,
        samples_per_digit: int = 100
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Phase 2: Digit-to-quantity mapping.

        Learn that "3" = ●●● (three objects).

        Args:
            samples_per_digit: Examples per digit

        Returns:
            List of (image_with_quantity, digit) tuples
        """
        dataset = []

        for digit in range(10):
            for _ in range(samples_per_digit):
                # Render digit with corresponding quantity
                img = self.renderer.render_digit_with_quantity(digit)
                dataset.append((img, digit))

        np.random.shuffle(dataset)

        return dataset

    def generate_addition_with_quantities_dataset(
        self,
        num_problems: int = 500,
        max_value: int = 5
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Phase 3: Addition with visual quantities.

        Learn that ●●● + ●● = ●●●●● (with digits shown).

        Args:
            num_problems: Number of problems
            max_value: Maximum value for each addend

        Returns:
            List of (problem_image, answer) tuples
        """
        dataset = []

        for _ in range(num_problems):
            a = np.random.randint(0, max_value + 1)
            b = np.random.randint(0, max_value + 1)
            result = a + b

            # Render problem without answer (for input)
            img_problem = self.renderer.render_addition_with_quantities(a, b, show_result=False)

            dataset.append((img_problem, result))

        return dataset

    def generate_addition_digits_only_dataset(
        self,
        num_problems: int = 500,
        max_value: int = 9
    ) -> List[Tuple[str, int]]:
        """
        Phase 4: Addition with digits only (abstraction).

        Learn 3 + 2 = 5 without visual quantities.

        Args:
            num_problems: Number of problems
            max_value: Maximum value for each addend

        Returns:
            List of (problem_string, answer) tuples
        """
        dataset = []

        for _ in range(num_problems):
            a = np.random.randint(0, max_value + 1)
            b = np.random.randint(0, max_value + 1)
            result = a + b

            problem_str = f"{a} + {b} ="
            dataset.append((problem_str, result))

        return dataset


# Demo
if __name__ == "__main__":
    """Demonstrate grounded curriculum."""

    print("=" * 70)
    print("GROUNDED MATH CURRICULUM DEMO")
    print("=" * 70)

    curriculum = GroundedMathCurriculum(seed=42)

    # Phase 1: Digit recognition
    print("\nPhase 1: Digit Recognition")
    print("-" * 70)
    digit_dataset = curriculum.generate_digit_recognition_dataset(samples_per_digit=5)
    print(f"Generated {len(digit_dataset)} digit recognition examples")

    # Save sample
    sample_img, sample_label = digit_dataset[0]
    cv2.imwrite("phase1_digit_example.png", sample_img)
    print(f"  Sample: digit {sample_label} saved to phase1_digit_example.png")

    # Phase 2: Digit-quantity mapping
    print("\nPhase 2: Digit-to-Quantity Mapping")
    print("-" * 70)
    quantity_dataset = curriculum.generate_quantity_mapping_dataset(samples_per_digit=5)
    print(f"Generated {len(quantity_dataset)} quantity mapping examples")

    # Save sample
    sample_img, sample_label = quantity_dataset[15]  # Digit 3
    cv2.imwrite("phase2_quantity_mapping.png", sample_img)
    print(f"  Sample: digit {sample_label} with quantity saved to phase2_quantity_mapping.png")

    # Phase 3: Addition with quantities
    print("\nPhase 3: Addition with Quantities")
    print("-" * 70)
    addition_qty_dataset = curriculum.generate_addition_with_quantities_dataset(num_problems=5)
    print(f"Generated {len(addition_qty_dataset)} addition problems with quantities")

    # Save sample
    sample_img, sample_answer = addition_qty_dataset[0]
    cv2.imwrite("phase3_addition_quantities.png", sample_img)
    print(f"  Sample: answer = {sample_answer}, saved to phase3_addition_quantities.png")

    # Also show one with result
    result_img = curriculum.renderer.render_addition_with_quantities(3, 2, show_result=True)
    cv2.imwrite("phase3_addition_with_result.png", result_img)
    print(f"  Example with result: 3 + 2 = 5, saved to phase3_addition_with_result.png")

    # Phase 4: Addition digits only
    print("\nPhase 4: Addition (Digits Only)")
    print("-" * 70)
    addition_digits_dataset = curriculum.generate_addition_digits_only_dataset(num_problems=5)
    print(f"Generated {len(addition_digits_dataset)} digit-only addition problems")

    for i, (problem, answer) in enumerate(addition_digits_dataset[:5], 1):
        print(f"  {i}. {problem} → {answer}")

    print("\n" + "=" * 70)
    print("CURRICULUM PROGRESSION")
    print("=" * 70)

    print("\nComplete learning path:")
    print("  Phase 1: Recognize digits 0-9 (visual patterns)")
    print("  Phase 2: Map digits to quantities (3 = ●●●)")
    print("  Phase 3: Addition with quantities (●●● + ●● = ●●●●●)")
    print("  Phase 4: Addition with digits only (3 + 2 = 5)")
    print("  Phase 5: Multiplication with quantities")
    print("  Phase 6: Multiplication with digits only")
    print("  Phase 7: Division with quantities")
    print("  Phase 8: Division with digits only")

    print("\nThis teaches what numbers MEAN, not just symbol manipulation!")
