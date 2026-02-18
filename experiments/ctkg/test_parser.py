"""Test that the DSL parser produces the same graph as the Python builder."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.ctkg.parser import parse_file, ParseError
from experiments.ctkg.domains.arithmetic import build_arithmetic_graph


def test_arithmetic_ctkg():
    """Parse arithmetic.ctkg and compare with build_arithmetic_graph()."""
    ctkg_path = os.path.join(os.path.dirname(__file__), 'domains', 'arithmetic.ctkg')

    # Parse the .ctkg file
    try:
        parsed = parse_file(ctkg_path)
    except ParseError as e:
        print(f"PARSE ERROR: {e}")
        return False

    # Build the reference graph
    ref = build_arithmetic_graph()

    errors = []

    # --- Compare concepts ---
    if set(parsed.concepts.keys()) != set(ref.concepts.keys()):
        errors.append(
            f"Concept names differ:\n"
            f"  parsed: {sorted(parsed.concepts.keys())}\n"
            f"  ref:    {sorted(ref.concepts.keys())}"
        )

    for name in sorted(set(parsed.concepts.keys()) & set(ref.concepts.keys())):
        pc = parsed.concepts[name]
        rc = ref.concepts[name]

        if pc.domain != rc.domain:
            errors.append(f"  {name}.domain: '{pc.domain}' != '{rc.domain}'")
        if pc.description != rc.description:
            errors.append(f"  {name}.description: '{pc.description}' != '{rc.description}'")
        if pc.input_type != rc.input_type:
            errors.append(f"  {name}.input_type: {pc.input_type} != {rc.input_type}")
        if pc.output_type != rc.output_type:
            errors.append(f"  {name}.output_type: {pc.output_type} != {rc.output_type}")
        if pc.supports_reverse != rc.supports_reverse:
            errors.append(f"  {name}.supports_reverse: {pc.supports_reverse} != {rc.supports_reverse}")
        if pc.pass_threshold != rc.pass_threshold:
            errors.append(f"  {name}.pass_threshold: {pc.pass_threshold} != {rc.pass_threshold}")
        if pc.max_epochs != rc.max_epochs:
            errors.append(f"  {name}.max_epochs: {pc.max_epochs} != {rc.max_epochs}")
        if pc.status != rc.status:
            # DSL concepts start as 'planned', Python ones are 'implemented'
            # This is expected — DSL doesn't know about implementation status
            pass

    # --- Compare prerequisites ---
    parsed_edges = [(p.source, p.target, p.role) for p in parsed.prerequisites]
    ref_edges = [(p.source, p.target, p.role) for p in ref.prerequisites]

    if sorted(parsed_edges) != sorted(ref_edges):
        errors.append(
            f"Prerequisites differ:\n"
            f"  parsed: {sorted(parsed_edges)}\n"
            f"  ref:    {sorted(ref_edges)}"
        )

    # --- Compare topological sort ---
    try:
        parsed_order = parsed.topological_sort()
        ref_order = ref.topological_sort()
        if parsed_order != ref_order:
            errors.append(
                f"Topological order differs:\n"
                f"  parsed: {parsed_order}\n"
                f"  ref:    {ref_order}"
            )
    except ValueError as e:
        errors.append(f"Topological sort failed: {e}")

    # --- Validate parsed graph ---
    validation_errors = parsed.validate()
    if validation_errors:
        errors.append(f"Parsed graph validation errors: {validation_errors}")

    # --- Report ---
    if errors:
        print("FAILURES:")
        for e in errors:
            print(f"  {e}")
        return False

    print("ALL CHECKS PASSED")
    print(f"  Concepts: {len(parsed.concepts)}")
    print(f"  Prerequisites: {len(parsed.prerequisites)}")
    print(f"  Topological order: {' -> '.join(parsed_order)}")
    print(f"  Validation: clean")

    # Show what DSL doesn't capture (by design)
    print("\nExpected differences (DSL vs Python builder):")
    for name in sorted(parsed.concepts.keys()):
        pc = parsed.concepts[name]
        rc = ref.concepts[name]
        if pc.status != rc.status:
            print(f"  {name}.status: DSL='{pc.status}' vs Python='{rc.status}' (expected)")
        if pc.generator_class != rc.generator_class:
            print(f"  {name}.generator_class: DSL='{pc.generator_class}' vs Python='{rc.generator_class}' (expected)")
        if pc.n_result != rc.n_result:
            print(f"  {name}.n_result: DSL='{pc.n_result}' vs Python='{rc.n_result}' (expected)")
        if pc.n_problems != rc.n_problems:
            print(f"  {name}.n_problems: DSL='{pc.n_problems}' vs Python='{rc.n_problems}' (expected)")
        if pc.is_atomic != rc.is_atomic:
            print(f"  {name}.is_atomic: DSL={pc.is_atomic} vs Python={rc.is_atomic} (check!)")

    return True


if __name__ == '__main__':
    success = test_arithmetic_ctkg()
    sys.exit(0 if success else 1)
