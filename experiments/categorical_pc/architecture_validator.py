"""
Categorical Architecture Validator

Type checker for categorical neural network architectures.
Validates that network configurations satisfy categorical constraints.

Usage:
    config = NetworkConfig()
    vision = config.add_subnet("vision", "product", ...)
    report = config.validate()

    if report.has_errors():
        print("Architecture violates categorical constraints!")
"""

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum


class SubnetType(Enum):
    """Categorical types for subnets."""
    SIMPLE = "simple"          # Basic functor A → B
    PRODUCT = "product"        # A × B (with projections)
    COPRODUCT = "coproduct"    # A + B (with injections)
    EXPONENTIAL = "exponential"  # B^A (function space)


@dataclass
class SubnetSpec:
    """Specification for a subnet with categorical type."""
    name: str
    type: SubnetType
    input_size: int
    output_size: int

    # For products/coproducts: component subnets
    components: List['SubnetSpec'] = field(default_factory=list)

    # For exponentials: domain and codomain
    domain: Optional['SubnetSpec'] = None
    codomain: Optional['SubnetSpec'] = None

    # Additional metadata
    has_weight_sharing: bool = False  # For conv layers (natural transformations)
    is_recurrent: bool = False  # For feedback connections

    def __hash__(self):
        return hash(self.name)


@dataclass
class Connection:
    """Connection (morphism) between subnets."""
    source: SubnetSpec
    target: SubnetSpec

    def __repr__(self):
        return f"{self.source.name} → {self.target.name}"


@dataclass
class ValidationReport:
    """Report of validation results."""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def is_valid(self) -> bool:
        return not self.has_errors()

    def print_report(self):
        """Print formatted validation report."""
        if self.is_valid():
            print("✓ Architecture is categorically valid!")
        else:
            print("✗ Architecture violates categorical constraints!")
            print("\nERRORS:")
            for error in self.errors:
                print(f"  ❌ {error}")

        if self.has_warnings():
            print("\nWARNINGS:")
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")


class CategoryTheoryValidator:
    """
    Validates neural network architectures against categorical constraints.

    Checks:
    1. Dimension compatibility (basic type checking)
    2. Product structures satisfy universal property
    3. Coproduct structures satisfy universal property
    4. Exponentials are proper function spaces
    5. Functors compose correctly
    6. Natural transformations (weight sharing where required)
    """

    def __init__(self):
        self.errors = []
        self.warnings = []

    def validate_architecture(self, config: 'NetworkConfig') -> ValidationReport:
        """Main validation entry point."""

        # 1. Basic type checking (dimensions)
        self._check_dimensions(config)

        # 2. Validate categorical structures
        self._validate_products(config)
        self._validate_coproducts(config)
        self._validate_exponentials(config)

        # 3. Check functorial composition
        self._validate_functor_composition(config)

        # 4. Verify universal properties
        self._check_universal_properties(config)

        # 5. Check natural transformations
        self._validate_natural_transformations(config)

        return ValidationReport(self.errors, self.warnings)

    def _check_dimensions(self, config: 'NetworkConfig'):
        """Basic dimensional analysis - do connections have matching sizes?"""
        for conn in config.connections:
            if conn.source.output_size != conn.target.input_size:
                self.errors.append(
                    f"Dimension mismatch in {conn}: "
                    f"{conn.source.name} outputs {conn.source.output_size} "
                    f"but {conn.target.name} expects {conn.target.input_size}"
                )

    def _validate_products(self, config: 'NetworkConfig'):
        """Check product structures satisfy universal property."""
        for subnet in config.subnets:
            if subnet.type != SubnetType.PRODUCT:
                continue

            # Product A×B must have output size = |A| + |B|
            if not subnet.components:
                self.errors.append(
                    f"Product {subnet.name} has no components "
                    "(products must have at least 2 components)"
                )
                continue

            expected_size = sum(c.output_size for c in subnet.components)

            if subnet.output_size != expected_size:
                self.errors.append(
                    f"Product {subnet.name} has output size {subnet.output_size} "
                    f"but should be {expected_size} (sum of component outputs: "
                    f"{[c.output_size for c in subnet.components]})"
                )

            # Products should have same input for all components
            # (All components process same input in parallel)
            expected_input = subnet.input_size
            for comp in subnet.components:
                if comp.input_size != expected_input:
                    self.warnings.append(
                        f"Product component {comp.name} has input size {comp.input_size} "
                        f"but product {subnet.name} has input size {expected_input}. "
                        f"Components should share input."
                    )

    def _validate_coproducts(self, config: 'NetworkConfig'):
        """Check coproduct structures satisfy universal property."""
        for subnet in config.subnets:
            if subnet.type != SubnetType.COPRODUCT:
                continue

            if not subnet.components:
                self.errors.append(
                    f"Coproduct {subnet.name} has no components "
                    "(coproducts must have at least 2 components)"
                )
                continue

            # Coproduct A+B typically has output size = max(|A|, |B|)
            # (One component active at a time via selection mechanism)
            max_component_size = max(c.output_size for c in subnet.components)

            if subnet.output_size < max_component_size:
                self.errors.append(
                    f"Coproduct {subnet.name} has output size {subnet.output_size} "
                    f"but needs at least {max_component_size} "
                    f"(max component size: {[c.output_size for c in subnet.components]})"
                )

    def _validate_exponentials(self, config: 'NetworkConfig'):
        """Check exponential structures (function spaces) B^A."""
        for subnet in config.subnets:
            if subnet.type != SubnetType.EXPONENTIAL:
                continue

            # Exponential B^A should map A → B
            if subnet.domain is None or subnet.codomain is None:
                self.errors.append(
                    f"Exponential {subnet.name} missing domain or codomain "
                    "(exponentials must specify B^A with domain A and codomain B)"
                )
                continue

            # Input should match domain output size
            if subnet.input_size != subnet.domain.output_size:
                self.errors.append(
                    f"Exponential {subnet.name} has input size {subnet.input_size} "
                    f"but domain {subnet.domain.name} outputs {subnet.domain.output_size}"
                )

            # Output should match codomain input size
            if subnet.output_size != subnet.codomain.input_size:
                self.errors.append(
                    f"Exponential {subnet.name} has output size {subnet.output_size} "
                    f"but codomain {subnet.codomain.name} expects {subnet.codomain.input_size}"
                )

    def _validate_functor_composition(self, config: 'NetworkConfig'):
        """Check that functors compose correctly (adjacent layers have matching dimensions)."""
        # Build adjacency graph
        adjacency = config.get_adjacency()

        # Check all paths of length 2 (F → G → H)
        for middle in config.subnets:
            predecessors = adjacency.get(middle, set())
            successors = set()
            for conn in config.connections:
                if conn.source == middle:
                    successors.add(conn.target)

            # For each path F → middle → G
            for F in predecessors:
                for G in successors:
                    # Check composition: F → middle → G
                    if F.output_size != middle.input_size:
                        # Already caught by dimension check
                        pass
                    if middle.output_size != G.input_size:
                        # Already caught by dimension check
                        pass

                    # Additional check: F ∘ G should be well-defined
                    # (This is implicit if dimensions match)

    def _check_universal_properties(self, config: 'NetworkConfig'):
        """
        Verify universal properties are satisfied.

        For products: Any subnet mapping to all components should factor through product.
        For coproducts: Any subnet receiving from all components should factor through coproduct.
        """

        # Check products
        for product in config.get_by_type(SubnetType.PRODUCT):
            # Find subnets that connect to all components
            for subnet in config.subnets:
                if subnet == product:
                    continue

                # Does subnet connect to all components?
                components_targeted = set()
                for conn in config.connections:
                    if conn.source == subnet and conn.target in product.components:
                        components_targeted.add(conn.target)

                if len(components_targeted) == len(product.components):
                    # Subnet maps to all components
                    # By universal property, should factor through product
                    has_product_connection = any(
                        conn.source == subnet and conn.target == product
                        for conn in config.connections
                    )

                    if not has_product_connection:
                        self.warnings.append(
                            f"{subnet.name} maps to all components of product {product.name} "
                            f"but doesn't factor through product (violates universal property). "
                            f"Consider adding direct connection {subnet.name} → {product.name}"
                        )

        # Check coproducts
        for coproduct in config.get_by_type(SubnetType.COPRODUCT):
            # Find subnets that receive from all components
            for subnet in config.subnets:
                if subnet == coproduct:
                    continue

                # Does subnet receive from all components?
                components_received = set()
                for conn in config.connections:
                    if conn.target == subnet and conn.source in coproduct.components:
                        components_received.add(conn.source)

                if len(components_received) == len(coproduct.components):
                    # Subnet receives from all components
                    # By universal property, should factor through coproduct
                    has_coproduct_connection = any(
                        conn.source == coproduct and conn.target == subnet
                        for conn in config.connections
                    )

                    if not has_coproduct_connection:
                        self.warnings.append(
                            f"{subnet.name} receives from all components of coproduct {coproduct.name} "
                            f"but doesn't factor through coproduct (violates universal property). "
                            f"Consider adding direct connection {coproduct.name} → {subnet.name}"
                        )

    def _validate_natural_transformations(self, config: 'NetworkConfig'):
        """
        Check natural transformations (e.g., weight sharing in conv layers).

        Natural transformations require structure preservation.
        For CNNs: weight sharing = natural transformation (translation invariance).
        """
        for subnet in config.subnets:
            # Heuristic: Subnets processing spatial data should use weight sharing
            if subnet.name.lower().startswith("conv") or "convolution" in subnet.name.lower():
                if not subnet.has_weight_sharing:
                    self.warnings.append(
                        f"Convolutional subnet {subnet.name} lacks weight sharing. "
                        f"Without weight sharing, it's not a natural transformation "
                        f"(violates translation invariance)"
                    )


class NetworkConfig:
    """
    Configuration specifying network architecture categorically.

    Allows building architecture spec and validating categorical constraints.
    """

    def __init__(self):
        self.subnets: List[SubnetSpec] = []
        self.connections: List[Connection] = []
        self._subnet_map: Dict[str, SubnetSpec] = {}

    def add_subnet(
        self,
        name: str,
        type: str,
        input_size: int,
        output_size: int,
        components: Optional[List[SubnetSpec]] = None,
        domain: Optional[SubnetSpec] = None,
        codomain: Optional[SubnetSpec] = None,
        has_weight_sharing: bool = False,
        is_recurrent: bool = False
    ) -> SubnetSpec:
        """
        Add subnet with categorical type.

        Args:
            name: Subnet identifier
            type: "simple", "product", "coproduct", or "exponential"
            input_size: Input dimension
            output_size: Output dimension
            components: For products/coproducts - list of component subnets
            domain: For exponentials - domain subnet A (in B^A)
            codomain: For exponentials - codomain subnet B (in B^A)
            has_weight_sharing: Whether subnet uses weight sharing (natural transformation)
            is_recurrent: Whether subnet has feedback connections

        Returns:
            SubnetSpec object
        """
        subnet_type = SubnetType(type)

        subnet = SubnetSpec(
            name=name,
            type=subnet_type,
            input_size=input_size,
            output_size=output_size,
            components=components or [],
            domain=domain,
            codomain=codomain,
            has_weight_sharing=has_weight_sharing,
            is_recurrent=is_recurrent
        )

        self.subnets.append(subnet)
        self._subnet_map[name] = subnet
        return subnet

    def connect(self, source: SubnetSpec, target: SubnetSpec):
        """Add connection (morphism) between subnets."""
        conn = Connection(source, target)
        self.connections.append(conn)

    def get_subnet(self, name: str) -> Optional[SubnetSpec]:
        """Get subnet by name."""
        return self._subnet_map.get(name)

    def get_by_type(self, subnet_type: SubnetType) -> List[SubnetSpec]:
        """Get all subnets of given categorical type."""
        return [s for s in self.subnets if s.type == subnet_type]

    def get_adjacency(self) -> Dict[SubnetSpec, Set[SubnetSpec]]:
        """Build adjacency map: subnet → set of predecessors."""
        adjacency = {subnet: set() for subnet in self.subnets}
        for conn in self.connections:
            adjacency[conn.target].add(conn.source)
        return adjacency

    def validate(self) -> ValidationReport:
        """Run categorical validation."""
        validator = CategoryTheoryValidator()
        return validator.validate_architecture(self)

    def print_architecture(self):
        """Print architecture summary."""
        print("=" * 70)
        print("CATEGORICAL ARCHITECTURE SPECIFICATION")
        print("=" * 70)

        print("\nSUBNETS:")
        for subnet in self.subnets:
            type_str = subnet.type.value.upper()
            print(f"  {subnet.name} ({type_str})")
            print(f"    Input:  {subnet.input_size}")
            print(f"    Output: {subnet.output_size}")

            if subnet.components:
                comp_names = [c.name for c in subnet.components]
                print(f"    Components: {comp_names}")

            if subnet.domain:
                print(f"    Domain: {subnet.domain.name}")
            if subnet.codomain:
                print(f"    Codomain: {subnet.codomain.name}")

            if subnet.has_weight_sharing:
                print(f"    Weight sharing: Yes (natural transformation)")

            if subnet.is_recurrent:
                print(f"    Recurrent: Yes (feedback)")

            print()

        print("CONNECTIONS:")
        for conn in self.connections:
            print(f"  {conn}")

        print("=" * 70)


if __name__ == "__main__":
    # Example: Validate simple vision → association → motor architecture

    print("Testing categorical validator with example architecture\n")

    config = NetworkConfig()

    # Simple linear pipeline
    vision = config.add_subnet("vision", "simple", input_size=30000, output_size=256)
    association = config.add_subnet("association", "simple", input_size=256, output_size=10)
    motor = config.add_subnet("motor", "simple", input_size=10, output_size=10)

    config.connect(vision, association)
    config.connect(association, motor)

    config.print_architecture()

    report = config.validate()
    report.print_report()

    print("\n" + "=" * 70)
    print("Testing product structure\n")

    # Product structure: Vision = Ventral × Dorsal
    config2 = NetworkConfig()

    ventral = config2.add_subnet("ventral", "simple", input_size=30000, output_size=512)
    dorsal = config2.add_subnet("dorsal", "simple", input_size=30000, output_size=256)
    vision_product = config2.add_subnet(
        "vision", "product",
        input_size=30000,
        output_size=768,  # 512 + 256
        components=[ventral, dorsal]
    )

    config2.print_architecture()
    report2 = config2.validate()
    report2.print_report()

    print("\n" + "=" * 70)
    print("Testing INVALID architecture (dimension mismatch)\n")

    config3 = NetworkConfig()

    vision_bad = config3.add_subnet("vision", "simple", input_size=30000, output_size=256)
    assoc_bad = config3.add_subnet("association", "simple", input_size=512, output_size=10)  # Wrong input!

    config3.connect(vision_bad, assoc_bad)

    config3.print_architecture()
    report3 = config3.validate()
    report3.print_report()
