"""
Mathematics Curriculum for Predictive Coding Network

Implements a comprehensive math curriculum from arithmetic to calculus,
designed to test:
1. Stable learning on structured sequential tasks
2. Continual learning without catastrophic forgetting
3. Transfer learning across mathematical domains

Based on: docs/notes/mathematics_curriculum_vision.md
"""

import random
from typing import List, Tuple, Dict, Optional
from enum import Enum


class MathDomain(Enum):
    """Mathematical domains in curriculum."""
    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    CALCULUS = "calculus"
    GEOMETRY = "geometry"
    TRIGONOMETRY = "trigonometry"
    PROBABILITY = "probability"


class DifficultyLevel(Enum):
    """Difficulty levels for curriculum scheduling."""
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4


class MathProblem:
    """
    A single math problem with input, output, and metadata.

    Attributes:
        input: Problem statement (string or token sequence)
        output: Expected answer (string or token sequence)
        domain: Which math domain (arithmetic, algebra, etc.)
        difficulty: Difficulty level (1-4)
        metadata: Additional info (problem type, concepts, etc.)
    """

    def __init__(
        self,
        input: str,
        output: str,
        domain: MathDomain,
        difficulty: DifficultyLevel,
        metadata: Optional[Dict] = None
    ):
        self.input = input
        self.output = output
        self.domain = domain
        self.difficulty = difficulty
        self.metadata = metadata or {}

    def __repr__(self):
        return f"MathProblem({self.domain.value}, diff={self.difficulty.value}, '{self.input}' → '{self.output}')"


class ArithmeticGenerator:
    """Generates arithmetic problems (addition, subtraction, multiplication, division)."""

    @staticmethod
    def generate_addition(difficulty: DifficultyLevel) -> MathProblem:
        """Generate addition problem."""
        if difficulty == DifficultyLevel.EASY:
            a, b = random.randint(1, 10), random.randint(1, 10)
        elif difficulty == DifficultyLevel.MEDIUM:
            a, b = random.randint(10, 100), random.randint(10, 100)
        elif difficulty == DifficultyLevel.HARD:
            a, b = random.randint(100, 1000), random.randint(100, 1000)
        else:  # EXPERT
            a, b = random.randint(1000, 10000), random.randint(1000, 10000)

        result = a + b
        return MathProblem(
            input=f"{a} + {b} =",
            output=str(result),
            domain=MathDomain.ARITHMETIC,
            difficulty=difficulty,
            metadata={"operation": "addition", "operands": [a, b]}
        )

    @staticmethod
    def generate_subtraction(difficulty: DifficultyLevel) -> MathProblem:
        """Generate subtraction problem."""
        if difficulty == DifficultyLevel.EASY:
            a, b = random.randint(5, 10), random.randint(1, 5)
        elif difficulty == DifficultyLevel.MEDIUM:
            a, b = random.randint(50, 100), random.randint(10, 50)
        elif difficulty == DifficultyLevel.HARD:
            a, b = random.randint(500, 1000), random.randint(100, 500)
        else:  # EXPERT
            a, b = random.randint(5000, 10000), random.randint(1000, 5000)

        result = a - b
        return MathProblem(
            input=f"{a} - {b} =",
            output=str(result),
            domain=MathDomain.ARITHMETIC,
            difficulty=difficulty,
            metadata={"operation": "subtraction", "operands": [a, b]}
        )

    @staticmethod
    def generate_multiplication(difficulty: DifficultyLevel) -> MathProblem:
        """Generate multiplication problem."""
        if difficulty == DifficultyLevel.EASY:
            a, b = random.randint(1, 10), random.randint(1, 10)
        elif difficulty == DifficultyLevel.MEDIUM:
            a, b = random.randint(10, 50), random.randint(1, 10)
        elif difficulty == DifficultyLevel.HARD:
            a, b = random.randint(10, 100), random.randint(10, 100)
        else:  # EXPERT
            a, b = random.randint(100, 1000), random.randint(10, 100)

        result = a * b
        return MathProblem(
            input=f"{a} * {b} =",
            output=str(result),
            domain=MathDomain.ARITHMETIC,
            difficulty=difficulty,
            metadata={"operation": "multiplication", "operands": [a, b]}
        )

    @staticmethod
    def generate_division(difficulty: DifficultyLevel) -> MathProblem:
        """Generate division problem (always integer results)."""
        if difficulty == DifficultyLevel.EASY:
            b = random.randint(2, 10)
            result = random.randint(1, 10)
        elif difficulty == DifficultyLevel.MEDIUM:
            b = random.randint(2, 20)
            result = random.randint(10, 50)
        elif difficulty == DifficultyLevel.HARD:
            b = random.randint(10, 50)
            result = random.randint(10, 100)
        else:  # EXPERT
            b = random.randint(10, 100)
            result = random.randint(100, 1000)

        a = b * result  # Ensure integer division
        return MathProblem(
            input=f"{a} / {b} =",
            output=str(result),
            domain=MathDomain.ARITHMETIC,
            difficulty=difficulty,
            metadata={"operation": "division", "operands": [a, b]}
        )


class AlgebraGenerator:
    """Generates algebra problems (linear equations, quadratics, etc.)."""

    @staticmethod
    def generate_linear_equation(difficulty: DifficultyLevel) -> MathProblem:
        """Generate linear equation: ax + b = c, solve for x."""
        if difficulty == DifficultyLevel.EASY:
            a, b, x = random.randint(1, 5), random.randint(1, 10), random.randint(1, 10)
        elif difficulty == DifficultyLevel.MEDIUM:
            a, b, x = random.randint(2, 10), random.randint(5, 20), random.randint(1, 20)
        elif difficulty == DifficultyLevel.HARD:
            a, b, x = random.randint(5, 20), random.randint(10, 50), random.randint(1, 50)
        else:  # EXPERT
            a, b, x = random.randint(10, 50), random.randint(20, 100), random.randint(1, 100)

        c = a * x + b

        # Format as fraction if not integer
        if (c - b) % a == 0:
            solution = str((c - b) // a)
        else:
            solution = f"{c - b}/{a}"

        return MathProblem(
            input=f"Solve for x: {a}x + {b} = {c}",
            output=f"x = {solution}",
            domain=MathDomain.ALGEBRA,
            difficulty=difficulty,
            metadata={"equation_type": "linear", "coefficients": [a, b, c]}
        )

    @staticmethod
    def generate_quadratic(difficulty: DifficultyLevel) -> MathProblem:
        """Generate simple quadratic: x^2 = a, solve for x."""
        if difficulty == DifficultyLevel.EASY:
            x = random.randint(1, 5)
        elif difficulty == DifficultyLevel.MEDIUM:
            x = random.randint(5, 10)
        elif difficulty == DifficultyLevel.HARD:
            x = random.randint(10, 20)
        else:  # EXPERT
            x = random.randint(20, 50)

        a = x * x

        return MathProblem(
            input=f"Solve for x: x^2 = {a}",
            output=f"x = ±{x}",
            domain=MathDomain.ALGEBRA,
            difficulty=difficulty,
            metadata={"equation_type": "quadratic", "value": a}
        )

    @staticmethod
    def generate_factoring(difficulty: DifficultyLevel) -> MathProblem:
        """Generate factoring problem: (x + a)(x + b) = x^2 + cx + d."""
        if difficulty == DifficultyLevel.EASY:
            a, b = random.randint(1, 5), random.randint(1, 5)
        elif difficulty == DifficultyLevel.MEDIUM:
            a, b = random.randint(1, 10), random.randint(1, 10)
        elif difficulty == DifficultyLevel.HARD:
            a, b = random.randint(-10, 10), random.randint(-10, 10)
        else:  # EXPERT
            a, b = random.randint(-20, 20), random.randint(-20, 20)

        c = a + b
        d = a * b

        # Format coefficients nicely
        c_str = f"+ {c}" if c >= 0 else f"- {abs(c)}"
        d_str = f"+ {d}" if d >= 0 else f"- {abs(d)}"

        return MathProblem(
            input=f"Factor: x^2 {c_str}x {d_str}",
            output=f"(x + {a})(x + {b})" if a >= 0 and b >= 0 else f"(x {'+' if a >= 0 else ''}{a})(x {'+' if b >= 0 else ''}{b})",
            domain=MathDomain.ALGEBRA,
            difficulty=difficulty,
            metadata={"equation_type": "factoring", "factors": [a, b]}
        )


class CalculusGenerator:
    """Generates calculus problems (derivatives, integrals)."""

    @staticmethod
    def generate_power_rule_derivative(difficulty: DifficultyLevel) -> MathProblem:
        """Generate derivative using power rule: d/dx(x^n) = n*x^(n-1)."""
        if difficulty == DifficultyLevel.EASY:
            n = random.randint(2, 5)
        elif difficulty == DifficultyLevel.MEDIUM:
            n = random.randint(5, 10)
        elif difficulty == DifficultyLevel.HARD:
            n = random.randint(10, 20)
        else:  # EXPERT
            n = random.randint(20, 50)

        if n == 1:
            output = "1"
        elif n == 2:
            output = f"{n}x"
        else:
            output = f"{n}x^{n-1}"

        return MathProblem(
            input=f"d/dx(x^{n}) =",
            output=output,
            domain=MathDomain.CALCULUS,
            difficulty=difficulty,
            metadata={"rule": "power_rule", "exponent": n}
        )

    @staticmethod
    def generate_polynomial_derivative(difficulty: DifficultyLevel) -> MathProblem:
        """Generate polynomial derivative: d/dx(ax^n + bx^m) = ..."""
        if difficulty == DifficultyLevel.EASY:
            a, n = random.randint(1, 5), random.randint(2, 3)
            b, m = random.randint(1, 5), 1
        elif difficulty == DifficultyLevel.MEDIUM:
            a, n = random.randint(1, 10), random.randint(2, 5)
            b, m = random.randint(1, 10), random.randint(1, 3)
        elif difficulty == DifficultyLevel.HARD:
            a, n = random.randint(1, 20), random.randint(3, 10)
            b, m = random.randint(1, 20), random.randint(2, 5)
        else:  # EXPERT
            a, n = random.randint(1, 50), random.randint(5, 20)
            b, m = random.randint(1, 50), random.randint(3, 10)

        # Format output
        term1 = f"{a*n}x^{n-1}" if n > 2 else f"{a*n}x" if n == 2 else f"{a*n}"
        term2 = f"{b*m}x^{m-1}" if m > 2 else f"{b*m}x" if m == 2 else f"{b*m}"

        return MathProblem(
            input=f"d/dx({a}x^{n} + {b}x^{m}) =",
            output=f"{term1} + {term2}",
            domain=MathDomain.CALCULUS,
            difficulty=difficulty,
            metadata={"rule": "polynomial_derivative", "terms": [(a, n), (b, m)]}
        )

    @staticmethod
    def generate_power_rule_integral(difficulty: DifficultyLevel) -> MathProblem:
        """Generate integral using power rule: ∫x^n dx = x^(n+1)/(n+1) + C."""
        if difficulty == DifficultyLevel.EASY:
            n = random.randint(1, 3)
        elif difficulty == DifficultyLevel.MEDIUM:
            n = random.randint(3, 7)
        elif difficulty == DifficultyLevel.HARD:
            n = random.randint(7, 15)
        else:  # EXPERT
            n = random.randint(15, 30)

        output = f"x^{n+1}/{n+1} + C"

        return MathProblem(
            input=f"∫x^{n} dx =",
            output=output,
            domain=MathDomain.CALCULUS,
            difficulty=difficulty,
            metadata={"rule": "power_rule_integral", "exponent": n}
        )


class MathCurriculum:
    """
    Main curriculum class that generates and manages math problems.

    Supports:
    - Sequential curriculum (arithmetic → algebra → calculus)
    - Difficulty progression (easy → expert)
    - Interleaved training (mix domains)
    - Continual learning experiments
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize curriculum.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        self.generators = {
            MathDomain.ARITHMETIC: [
                ArithmeticGenerator.generate_addition,
                ArithmeticGenerator.generate_subtraction,
                ArithmeticGenerator.generate_multiplication,
                ArithmeticGenerator.generate_division,
            ],
            MathDomain.ALGEBRA: [
                AlgebraGenerator.generate_linear_equation,
                AlgebraGenerator.generate_quadratic,
                AlgebraGenerator.generate_factoring,
            ],
            MathDomain.CALCULUS: [
                CalculusGenerator.generate_power_rule_derivative,
                CalculusGenerator.generate_polynomial_derivative,
                CalculusGenerator.generate_power_rule_integral,
            ]
        }

    def generate_problem(
        self,
        domain: MathDomain,
        difficulty: DifficultyLevel,
        problem_type: Optional[str] = None
    ) -> MathProblem:
        """
        Generate a single math problem.

        Args:
            domain: Which math domain
            difficulty: Difficulty level
            problem_type: Specific problem type (e.g., "addition"), or random if None

        Returns:
            MathProblem instance
        """
        generators = self.generators[domain]

        if problem_type is not None:
            # Find specific generator by name
            generator = next(
                (g for g in generators if problem_type in g.__name__),
                None
            )
            if generator is None:
                raise ValueError(f"Unknown problem type: {problem_type}")
        else:
            # Random generator from domain
            generator = random.choice(generators)

        return generator(difficulty)

    def generate_batch(
        self,
        batch_size: int,
        domain: MathDomain,
        difficulty: DifficultyLevel
    ) -> List[MathProblem]:
        """
        Generate batch of problems from single domain.

        Args:
            batch_size: Number of problems
            domain: Which math domain
            difficulty: Difficulty level

        Returns:
            List of MathProblem instances
        """
        return [
            self.generate_problem(domain, difficulty)
            for _ in range(batch_size)
        ]

    def generate_sequential_curriculum(
        self,
        problems_per_domain: int = 1000,
        difficulty: DifficultyLevel = DifficultyLevel.EASY
    ) -> List[MathProblem]:
        """
        Generate sequential curriculum: arithmetic → algebra → calculus.

        Used to test catastrophic forgetting:
        - Train on arithmetic first
        - Then train on algebra (does it forget arithmetic?)
        - Then train on calculus (does it forget both?)

        Args:
            problems_per_domain: Problems per domain
            difficulty: Difficulty level

        Returns:
            List of problems in sequential order
        """
        problems = []

        # Phase 1: Arithmetic
        problems.extend(
            self.generate_batch(problems_per_domain, MathDomain.ARITHMETIC, difficulty)
        )

        # Phase 2: Algebra
        problems.extend(
            self.generate_batch(problems_per_domain, MathDomain.ALGEBRA, difficulty)
        )

        # Phase 3: Calculus
        problems.extend(
            self.generate_batch(problems_per_domain, MathDomain.CALCULUS, difficulty)
        )

        return problems

    def generate_interleaved_curriculum(
        self,
        problems_per_domain: int = 1000,
        difficulty: DifficultyLevel = DifficultyLevel.EASY,
        shuffle: bool = True
    ) -> List[MathProblem]:
        """
        Generate interleaved curriculum: mix all domains.

        Helps prevent catastrophic forgetting by continuously reviewing all domains.

        Args:
            problems_per_domain: Problems per domain
            difficulty: Difficulty level
            shuffle: Whether to shuffle problems

        Returns:
            List of problems (shuffled if shuffle=True)
        """
        problems = []

        for domain in [MathDomain.ARITHMETIC, MathDomain.ALGEBRA, MathDomain.CALCULUS]:
            problems.extend(
                self.generate_batch(problems_per_domain, domain, difficulty)
            )

        if shuffle:
            random.shuffle(problems)

        return problems

    def generate_progressive_curriculum(
        self,
        problems_per_difficulty: int = 500
    ) -> List[MathProblem]:
        """
        Generate progressive curriculum: easy → hard within each domain.

        Args:
            problems_per_difficulty: Problems per difficulty level

        Returns:
            List of problems with increasing difficulty
        """
        problems = []

        for domain in [MathDomain.ARITHMETIC, MathDomain.ALGEBRA, MathDomain.CALCULUS]:
            for difficulty in [DifficultyLevel.EASY, DifficultyLevel.MEDIUM,
                             DifficultyLevel.HARD, DifficultyLevel.EXPERT]:
                problems.extend(
                    self.generate_batch(problems_per_difficulty, domain, difficulty)
                )

        return problems

    def create_test_set(
        self,
        test_size_per_domain: int = 100,
        difficulty: DifficultyLevel = DifficultyLevel.EASY
    ) -> Dict[MathDomain, List[MathProblem]]:
        """
        Create test set for evaluating continual learning.

        Args:
            test_size_per_domain: Test problems per domain
            difficulty: Difficulty level

        Returns:
            Dict mapping domain to list of test problems
        """
        test_set = {}

        for domain in [MathDomain.ARITHMETIC, MathDomain.ALGEBRA, MathDomain.CALCULUS]:
            test_set[domain] = self.generate_batch(
                test_size_per_domain, domain, difficulty
            )

        return test_set


def tokenize_problem(problem: MathProblem, vocab: Dict[str, int]) -> Tuple[List[int], List[int]]:
    """
    Tokenize a math problem into input and output token sequences.

    Args:
        problem: MathProblem instance
        vocab: Vocabulary mapping tokens to integers

    Returns:
        (input_tokens, output_tokens) as lists of integers
    """
    # Simple character-level tokenization
    input_tokens = [vocab.get(c, vocab['<UNK>']) for c in problem.input]
    output_tokens = [vocab.get(c, vocab['<UNK>']) for c in problem.output]

    return input_tokens, output_tokens


def create_vocab() -> Dict[str, int]:
    """
    Create vocabulary for math problems.

    Returns:
        Dict mapping characters/tokens to integers
    """
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<START>': 2,
        '<END>': 3,
    }

    # Digits
    for i in range(10):
        vocab[str(i)] = len(vocab)

    # Operators
    for op in ['+', '-', '*', '/', '=', '^', '(', ')', '[', ']']:
        vocab[op] = len(vocab)

    # Letters (for variables)
    for c in 'abcdefghijklmnopqrstuvwxyz':
        vocab[c] = len(vocab)

    # Special math symbols
    for symbol in ['∫', 'd', '±', 'π', 'e', 'C']:
        vocab[symbol] = len(vocab)

    # Space and punctuation
    vocab[' '] = len(vocab)
    vocab[':'] = len(vocab)
    vocab[','] = len(vocab)

    return vocab


if __name__ == "__main__":
    """Demo: Generate sample problems from curriculum."""

    curriculum = MathCurriculum(seed=42)

    print("=" * 70)
    print("MATHEMATICS CURRICULUM DEMO")
    print("=" * 70)

    # Generate samples from each domain
    for domain in [MathDomain.ARITHMETIC, MathDomain.ALGEBRA, MathDomain.CALCULUS]:
        print(f"\n{domain.value.upper()} (Easy):")
        for _ in range(3):
            problem = curriculum.generate_problem(domain, DifficultyLevel.EASY)
            print(f"  {problem.input} → {problem.output}")

    print("\n" + "=" * 70)
    print("CURRICULUM TYPES")
    print("=" * 70)

    # Sequential curriculum
    sequential = curriculum.generate_sequential_curriculum(
        problems_per_domain=5,
        difficulty=DifficultyLevel.EASY
    )
    print(f"\nSequential curriculum: {len(sequential)} problems")
    print("First 3:", [p.domain.value for p in sequential[:3]])
    print("Middle 3:", [p.domain.value for p in sequential[5:8]])
    print("Last 3:", [p.domain.value for p in sequential[-3:]])

    # Interleaved curriculum
    interleaved = curriculum.generate_interleaved_curriculum(
        problems_per_domain=5,
        difficulty=DifficultyLevel.EASY,
        shuffle=True
    )
    print(f"\nInterleaved curriculum: {len(interleaved)} problems")
    print("Order:", [p.domain.value[:4] for p in interleaved[:15]])

    # Test set
    test_set = curriculum.create_test_set(test_size_per_domain=3)
    print(f"\nTest set:")
    for domain, problems in test_set.items():
        print(f"  {domain.value}: {len(problems)} problems")

    print("\n" + "=" * 70)
    print("VOCABULARY")
    print("=" * 70)
    vocab = create_vocab()
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Sample tokens: {list(vocab.items())[:20]}")

    print("\n" + "=" * 70)
    print("TOKENIZATION DEMO")
    print("=" * 70)
    problem = curriculum.generate_problem(MathDomain.ARITHMETIC, DifficultyLevel.EASY)
    input_tokens, output_tokens = tokenize_problem(problem, vocab)
    print(f"Problem: {problem.input} → {problem.output}")
    print(f"Input tokens: {input_tokens}")
    print(f"Output tokens: {output_tokens}")
