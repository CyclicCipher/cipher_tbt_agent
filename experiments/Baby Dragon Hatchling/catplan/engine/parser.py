"""CatPlan parser — reads .catplan files into Domain and Problem objects.

Handles domain definitions (category ... where), problem definitions
(problem ... domain ...), derived predicates, conditional effects,
quantified and disjunctive preconditions, equality constraints.

Line-based, indentation-sensitive, no external dependencies.
"""
from __future__ import annotations

import re
from pathlib import Path

from .types import (
    Type, Predicate, DerivedPredicate, Invariant,
    ActionParam, ActionDef, ConditionalEffect, CompositeAction, Adjunction,
    InitialAlgebra, Functor, NaturalTransformation,
    AtomCondition, ExistsCondition, ForallCondition,
    OrCondition, AndCondition, EqualityCondition, CountCondition, ConditionExpr,
    Effect, Domain, ObjectDecl, GroundAtom, NegatedGoalAtom, Problem,
)

# Keywords that start a new block inside a category.
_KEYWORDS = {
    "type", "pred", "derived", "invariant", "action",
    "compose", "adjunction", "initial_algebra", "functor", "natural",
    "category", "problem",
}


def parse_file(path: str | Path) -> tuple[list[Domain], list[Problem]]:
    """Parse a .catplan file. Returns (domains, problems)."""
    text = Path(path).read_text(encoding="utf-8")
    return parse(text)


def parse(text: str) -> tuple[list[Domain], list[Problem]]:
    """Parse CatPlan source text. Returns (domains, problems)."""
    lines = text.split("\n")
    domains: list[Domain] = []
    problems: list[Problem] = []

    i = 0
    while i < len(lines):
        line = _strip_comment(lines[i])
        stripped = line.strip()

        if stripped.startswith("category "):
            domain, i = _parse_domain(lines, i)
            domains.append(domain)
        elif stripped.startswith("problem "):
            problem, i = _parse_problem(lines, i)
            problems.append(problem)
        else:
            i += 1

    return domains, problems


# ---------------------------------------------------------------------------
# Domain parsing
# ---------------------------------------------------------------------------

def _parse_domain(lines: list[str], start: int) -> tuple[Domain, int]:
    header = _strip_comment(lines[start]).strip()
    m = re.match(r"category\s+(\w+)\s+where", header)
    if not m:
        raise ParseError(f"Bad category header: {header}", start)
    domain = Domain(name=m.group(1))

    i = start + 1
    while i < len(lines):
        line = _strip_comment(lines[i])
        stripped = line.strip()

        if not stripped:
            i += 1
            continue
        if not line[0].isspace() and stripped and not stripped.startswith("--"):
            break

        if stripped.startswith("type "):
            _parse_type(domain, stripped)
            i += 1
        elif stripped.startswith("pred "):
            _parse_predicate(domain, stripped)
            i += 1
        elif stripped.startswith("derived "):
            dp, i = _parse_derived(lines, i, domain)
            domain.derived[dp.name] = dp
        elif stripped.startswith("invariant "):
            inv, i = _parse_invariant(lines, i)
            domain.invariants.append(inv)
        elif stripped.startswith("action "):
            action, i = _parse_action(lines, i)
            domain.actions[action.name] = action
        elif stripped.startswith("compose "):
            comp, i = _parse_composite(lines, i)
            domain.composites[comp.name] = comp
        elif stripped.startswith("adjunction "):
            adj = _parse_adjunction(stripped)
            if adj:
                domain.adjunctions[adj.name] = adj
            i += 1
        elif stripped.startswith("initial_algebra "):
            ia, i = _parse_initial_algebra(lines, i)
            domain.initial_algebras[ia.name] = ia
        elif stripped.startswith("functor "):
            fn, i = _parse_functor(lines, i)
            domain.functors[fn.name] = fn
        else:
            i += 1

    return domain, i


def _parse_type(domain: Domain, line: str):
    m = re.match(r"type\s+(\w+)\s*(?:=\s*(.+))?", line)
    if not m:
        raise ParseError(f"Bad type: {line}")
    name = m.group(1)
    variants_str = m.group(2)
    variants = tuple(v.strip() for v in variants_str.split("|")) if variants_str else ()
    domain.types[name] = Type(name=name, variants=variants)


def _parse_predicate(domain: Domain, line: str):
    m = re.match(r"pred\s+(\w+)\s*:\s*(.+)", line)
    if not m:
        raise ParseError(f"Bad predicate: {line}")
    name = m.group(1)
    type_chain = [t.strip() for t in m.group(2).split("->")]
    if type_chain[-1].strip() == "Prop":
        param_types = tuple(type_chain[:-1])
    else:
        param_types = tuple(type_chain)
    domain.predicates[name] = Predicate(name=name, param_types=param_types)


def _parse_derived(lines: list[str], start: int, domain: Domain) -> tuple[DerivedPredicate, int]:
    """Parse: derived clear(b : Block) = not(exists b2 : Block . on(b2, b))"""
    header = _strip_comment(lines[start]).strip()
    m = re.match(r"derived\s+(\w+)\(([^)]*)\)\s*=\s*(.+)", header)
    if not m:
        raise ParseError(f"Bad derived: {header}", start)
    name = m.group(1)
    params = _parse_params(m.group(2))
    body_text = m.group(3).strip()

    # Collect continuation lines.
    i = start + 1
    while i < len(lines):
        line = _strip_comment(lines[i])
        stripped = line.strip()
        if stripped and stripped.split()[0] in _KEYWORDS:
            break
        if stripped:
            body_text += " " + stripped
        i += 1

    body = _parse_condition_expr(body_text, domain)
    param_types = tuple(p.type_name for p in params)
    return DerivedPredicate(name=name, param_types=param_types, body=body), i


def _parse_invariant(lines: list[str], start: int) -> tuple[Invariant, int]:
    header = _strip_comment(lines[start]).strip()
    m = re.match(r'invariant\s+"([^"]+)"', header)
    desc = m.group(1) if m else header
    body_lines = []
    i = start + 1
    while i < len(lines):
        line = _strip_comment(lines[i])
        stripped = line.strip()
        if stripped and stripped.split()[0] in _KEYWORDS:
            break
        if stripped:
            body_lines.append(stripped)
        i += 1
    raw = " ".join(body_lines).strip()
    # Try to parse the body as a ConditionExpr.
    cond = _parse_condition_expr(raw) if raw else None
    return Invariant(description=desc, raw_text=raw, condition=cond), i


def _parse_action(lines: list[str], start: int) -> tuple[ActionDef, int]:
    header = _strip_comment(lines[start]).strip()
    m = re.match(r"action\s+(\w+)\(([^)]*)\)", header)
    if not m:
        raise ParseError(f"Bad action header: {header}", start)
    name = m.group(1)
    params = _parse_params(m.group(2))
    action = ActionDef(name=name, params=params)

    i = start + 1
    while i < len(lines):
        line = _strip_comment(lines[i])
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.split()[0] in _KEYWORDS:
            break
        if stripped.startswith("require "):
            cond = _parse_condition_expr(stripped[8:].strip())
            if cond is not None:
                action.preconditions.append(cond)
        elif stripped.startswith("effect "):
            eff = _parse_effect(stripped[7:].strip())
            if eff is not None:
                action.effects.append(eff)
        elif stripped.startswith("when "):
            ce = _parse_conditional_effect(stripped[5:].strip())
            if ce is not None:
                action.conditional_effects.append(ce)
        i += 1

    return action, i


# ---------------------------------------------------------------------------
# D.1: Composite action parsing
# ---------------------------------------------------------------------------

def _parse_composite(lines: list[str], start: int) -> tuple[CompositeAction, int]:
    """Parse: compose pick_and_stack(h, b, target) = stack(h, b, target) . pick(h, b)"""
    header = _strip_comment(lines[start]).strip()
    m = re.match(r"compose\s+(\w+)\(([^)]*)\)\s*=\s*(.+)", header)
    if not m:
        raise ParseError(f"Bad compose: {header}", start)
    name = m.group(1)
    params = _parse_params(m.group(2))
    steps_text = m.group(3).strip()

    # Collect continuation lines.
    i = start + 1
    while i < len(lines):
        line = _strip_comment(lines[i])
        stripped = line.strip()
        if stripped and stripped.split()[0] in _KEYWORDS:
            break
        if stripped:
            steps_text += " " + stripped
        i += 1

    # Parse steps: "action2(args) . action1(args)" (rightmost applied first)
    step_strs = [s.strip() for s in steps_text.split(".")]
    step_strs.reverse()  # reverse so first-applied comes first

    steps = []
    param_names = {p.name for p in params}
    for step_str in step_strs:
        sm = re.match(r"(\w+)\(([^)]*)\)", step_str)
        if sm:
            action_name = sm.group(1)
            step_args = [a.strip() for a in sm.group(2).split(",") if a.strip()]
            # Build param mapping: step_param_position -> composite_param_name
            # For now, positional mapping.
            param_mapping = {}
            for j, arg in enumerate(step_args):
                if arg in param_names:
                    param_mapping[f"p{j}"] = arg
            steps.append((action_name, dict(zip(step_args, step_args))))

    return CompositeAction(name=name, params=params, steps=steps), i


# ---------------------------------------------------------------------------
# D.2: Adjunction parsing
# ---------------------------------------------------------------------------

def _parse_adjunction(line: str) -> Adjunction | None:
    """Parse: adjunction add_sub : add -| sub"""
    m = re.match(r"adjunction\s+(\w+)\s*:\s*(\w+)\s*-\|\s*(\w+)", line)
    if not m:
        return None
    return Adjunction(
        name=m.group(1),
        left_action=m.group(2),
        right_action=m.group(3),
        param_map={},  # filled in by domain-specific logic later
    )


# ---------------------------------------------------------------------------
# D.5: Initial algebra parsing
# ---------------------------------------------------------------------------

def _parse_initial_algebra(lines: list[str], start: int) -> tuple[InitialAlgebra, int]:
    """Parse:
    initial_algebra Nat
      carrier Block
      zero zero_block
      succ succ_action
    """
    header = _strip_comment(lines[start]).strip()
    m = re.match(r"initial_algebra\s+(\w+)", header)
    if not m:
        raise ParseError(f"Bad initial_algebra: {header}", start)
    name = m.group(1)
    carrier = ""
    zero = ""
    succ = ""

    i = start + 1
    while i < len(lines):
        line = _strip_comment(lines[i])
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.split()[0] in _KEYWORDS:
            break
        if stripped.startswith("carrier "):
            carrier = stripped.split()[1]
        elif stripped.startswith("zero "):
            zero = stripped.split()[1]
        elif stripped.startswith("succ "):
            succ = stripped.split()[1]
        i += 1

    return InitialAlgebra(name=name, carrier_type=carrier, zero=zero, succ=succ), i


# ---------------------------------------------------------------------------
# Functor parsing
# ---------------------------------------------------------------------------

def _parse_functor(lines: list[str], start: int) -> tuple[Functor, int]:
    """Parse:
    functor F : DomainA -> DomainB
      map TypeA -> TypeB
      map pred_a -> pred_b
    """
    header = _strip_comment(lines[start]).strip()
    m = re.match(r"functor\s+(\w+)\s*:\s*(\w+)\s*->\s*(\w+)", header)
    if not m:
        raise ParseError(f"Bad functor: {header}", start)
    name = m.group(1)
    source = m.group(2)
    target = m.group(3)
    type_map: dict[str, str] = {}
    pred_map: dict[str, str] = {}
    action_map: dict[str, str] = {}

    i = start + 1
    while i < len(lines):
        line = _strip_comment(lines[i])
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.split()[0] in _KEYWORDS:
            break
        m2 = re.match(r"map\s+(\w+)\s*->\s*(\w+)", stripped)
        if m2:
            # Heuristic: if both names are capitalized, it's a type map.
            # Otherwise it could be pred or action — we store in all three
            # and let the domain disambiguate at use time.
            type_map[m2.group(1)] = m2.group(2)
            pred_map[m2.group(1)] = m2.group(2)
            action_map[m2.group(1)] = m2.group(2)
        i += 1

    return Functor(
        name=name, source_domain=source, target_domain=target,
        type_map=type_map, predicate_map=pred_map, action_map=action_map,
    ), i


# ---------------------------------------------------------------------------
# Condition expression parsing (recursive descent)
# ---------------------------------------------------------------------------

def _parse_condition_expr(text: str, domain: Domain | None = None) -> ConditionExpr | None:
    """Parse a condition expression. Handles:
    - pred(a, b)
    - not(pred(a, b))
    - exists v : T . body
    - forall v : T . body
    - cond1 or cond2
    - a = b, a /= b
    """
    text = text.strip()
    if not text:
        return None

    # Disjunction: split on ' or ' (not inside parens).
    or_parts = _split_outside_parens(text, " or ")
    if len(or_parts) > 1:
        conds = []
        for part in or_parts:
            c = _parse_condition_expr(part.strip(), domain)
            if c is not None:
                conds.append(c)
        return OrCondition(conditions=tuple(conds)) if conds else None

    # Conjunction: split on ', ' (not inside parens).
    and_parts = _split_outside_parens(text, ", ")
    if len(and_parts) > 1:
        conds = []
        for part in and_parts:
            c = _parse_condition_expr(part.strip(), domain)
            if c is not None:
                conds.append(c)
        if len(conds) == 1:
            return conds[0]
        return AndCondition(conditions=tuple(conds)) if conds else None

    # Exists: exists v : T . body
    m = re.match(r"exists\s+(\w+)\s*:\s*(\w+)\s*\.\s*(.+)", text)
    if m:
        body = _parse_condition_expr(m.group(3).strip(), domain)
        if body is not None:
            return ExistsCondition(var_name=m.group(1), var_type=m.group(2), body=body)
        return None

    # Forall: forall v : T . body
    m = re.match(r"forall\s+(\w+)\s*:\s*(\w+)\s*\.\s*(.+)", text)
    if m:
        body = _parse_condition_expr(m.group(3).strip(), domain)
        if body is not None:
            return ForallCondition(var_name=m.group(1), var_type=m.group(2), body=body)
        return None

    # Count: count(var : Type . body) op value
    m = re.match(r"count\((\w+)\s*:\s*(\w+)\s*\.\s*(.+)\)\s*(<=|>=|=|<|>)\s*(\d+)", text)
    if m:
        var_name = m.group(1)
        var_type = m.group(2)
        body_text = m.group(3).strip()
        op = m.group(4)
        value = int(m.group(5))
        body = _parse_condition_expr(body_text, domain)
        if body is not None:
            return CountCondition(var_name=var_name, var_type=var_type,
                                 body=body, op=op, value=value)
        return None

    # Negation: not(...)
    # Must match the OUTERMOST parens only (not inner parens in nested expressions).
    if text.startswith("not(") and text.endswith(")"):
        # Verify these parens match (not a false positive from inner parens).
        depth = 0
        match_end = -1
        for ci, ch in enumerate(text):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    match_end = ci
                    break
        if match_end == len(text) - 1:
            inner = text[4:-1].strip()
            inner_cond = _parse_condition_expr(inner, domain)
            if inner_cond is not None:
                # Negate the inner condition.
                if isinstance(inner_cond, AtomCondition):
                    return AtomCondition(
                        predicate=inner_cond.predicate, args=inner_cond.args,
                        negated=not inner_cond.negated)
                elif isinstance(inner_cond, ExistsCondition):
                    # not(exists x . P) = forall x . not(P)
                    negated_body = _negate_condition(inner_cond.body)
                    return ForallCondition(
                        var_name=inner_cond.var_name,
                        var_type=inner_cond.var_type,
                        body=negated_body)
                elif isinstance(inner_cond, ForallCondition):
                    # not(forall x . P) = exists x . not(P)
                    negated_body = _negate_condition(inner_cond.body)
                    return ExistsCondition(
                        var_name=inner_cond.var_name,
                        var_type=inner_cond.var_type,
                        body=negated_body)
                else:
                    return inner_cond

    # Equality: a = b or a /= b
    m = re.match(r"(\w+)\s*/=\s*(\w+)", text)
    if m:
        return EqualityCondition(left=m.group(1), right=m.group(2), negated=True)
    m = re.match(r"(\w+)\s*=\s*(\w+)", text)
    if m:
        return EqualityCondition(left=m.group(1), right=m.group(2), negated=False)

    # Simple predicate: pred(a, b)
    m = re.match(r"(\w+)\(([^)]*)\)", text)
    if m:
        pred = m.group(1)
        args = tuple(a.strip() for a in m.group(2).split(",") if a.strip())
        return AtomCondition(predicate=pred, args=args, negated=False)

    return None


def _negate_condition(cond: ConditionExpr) -> ConditionExpr:
    """Negate a condition expression (De Morgan's laws)."""
    if isinstance(cond, AtomCondition):
        return AtomCondition(predicate=cond.predicate, args=cond.args,
                             negated=not cond.negated)
    elif isinstance(cond, EqualityCondition):
        return EqualityCondition(left=cond.left, right=cond.right,
                                 negated=not cond.negated)
    elif isinstance(cond, OrCondition):
        return AndCondition(conditions=tuple(_negate_condition(c) for c in cond.conditions))
    elif isinstance(cond, AndCondition):
        return OrCondition(conditions=tuple(_negate_condition(c) for c in cond.conditions))
    elif isinstance(cond, ExistsCondition):
        return ForallCondition(var_name=cond.var_name, var_type=cond.var_type,
                               body=_negate_condition(cond.body))
    elif isinstance(cond, ForallCondition):
        return ExistsCondition(var_name=cond.var_name, var_type=cond.var_type,
                               body=_negate_condition(cond.body))
    return cond


def _split_outside_parens(text: str, sep: str) -> list[str]:
    """Split text on sep, but only when not inside parentheses."""
    parts = []
    depth = 0
    current = ""
    i = 0
    while i < len(text):
        if text[i] == '(':
            depth += 1
            current += text[i]
        elif text[i] == ')':
            depth -= 1
            current += text[i]
        elif depth == 0 and text[i:i+len(sep)] == sep:
            parts.append(current)
            current = ""
            i += len(sep)
            continue
        else:
            current += text[i]
        i += 1
    if current:
        parts.append(current)
    return parts


# ---------------------------------------------------------------------------
# Effect parsing
# ---------------------------------------------------------------------------

def _parse_effect(text: str) -> Effect | None:
    set_to = True
    if text.startswith("not(") and text.endswith(")"):
        set_to = False
        text = text[4:-1]
    m = re.match(r"(\w+)\(([^)]*)\)", text)
    if not m:
        return None
    pred = m.group(1)
    args = tuple(a.strip().rstrip("_") for a in m.group(2).split(",") if a.strip())
    return Effect(predicate=pred, args=args, set_to=set_to)


def _parse_conditional_effect(text: str) -> ConditionalEffect | None:
    """Parse: on(b, x) : not(on(b, x)), clear(x)"""
    parts = text.split(" : ", 1)
    if len(parts) != 2:
        return None
    cond = _parse_condition_expr(parts[0].strip())
    if cond is None:
        return None
    # Split effects on ', ' but only outside parentheses.
    effect_strs = _split_outside_parens(parts[1].strip(), ", ")
    effects = []
    for es in effect_strs:
        eff = _parse_effect(es.strip())
        if eff is not None:
            effects.append(eff)
    if not effects:
        return None
    return ConditionalEffect(condition=cond, effects=tuple(effects))


# ---------------------------------------------------------------------------
# Param parsing
# ---------------------------------------------------------------------------

def _parse_params(params_str: str) -> list[ActionParam]:
    params = []
    for part in params_str.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"(\w+)\s*:\s*(\w+)", part)
        if m:
            params.append(ActionParam(name=m.group(1), type_name=m.group(2)))
    return params


# ---------------------------------------------------------------------------
# Problem parsing
# ---------------------------------------------------------------------------

def _parse_problem(lines: list[str], start: int) -> tuple[Problem, int]:
    header = _strip_comment(lines[start]).strip()
    m = re.match(r"problem\s+(\w+)", header)
    if not m:
        raise ParseError(f"Bad problem header: {header}", start)
    problem = Problem(name=m.group(1), domain_name="")

    section = None
    i = start + 1
    while i < len(lines):
        line = _strip_comment(lines[i])
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if line and not line[0].isspace():
            break

        if stripped.startswith("domain "):
            problem.domain_name = stripped.split()[1]
        elif stripped == "objects":
            section = "objects"
        elif stripped == "init":
            section = "init"
        elif stripped == "goal":
            section = "goal"
        elif section == "objects":
            m = re.match(r"(\w+)\s*:\s*(\w+)", stripped)
            if m:
                problem.objects[m.group(1)] = ObjectDecl(
                    name=m.group(1), type_name=m.group(2))
        elif section == "init":
            atom = _parse_ground_atom(stripped)
            if atom:
                problem.init.add(atom)
        elif section == "goal":
            if stripped.startswith("not(") and stripped.endswith(")"):
                inner = stripped[4:-1]
                atom = _parse_ground_atom(inner)
                if atom:
                    problem.neg_goal.add(NegatedGoalAtom(
                        predicate=atom.predicate, args=atom.args))
            else:
                atom = _parse_ground_atom(stripped)
                if atom:
                    problem.goal.add(atom)
        i += 1

    return problem, i


def _parse_ground_atom(text: str) -> GroundAtom | None:
    m = re.match(r"(\w+)\(([^)]*)\)", text)
    if not m:
        return None
    pred = m.group(1)
    args = tuple(a.strip() for a in m.group(2).split(",") if a.strip())
    return GroundAtom(predicate=pred, args=args)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _strip_comment(line: str) -> str:
    idx = line.find("--")
    return line[:idx] if idx >= 0 else line


class ParseError(Exception):
    def __init__(self, msg: str, line: int = -1):
        super().__init__(f"Line {line}: {msg}" if line >= 0 else msg)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m catplan.engine.parser <file.catplan>")
        sys.exit(1)
    domains, problems = parse_file(sys.argv[1])
    for d in domains:
        print(f"Domain: {d.name}")
        print(f"  Types: {list(d.types.keys())}")
        print(f"  Predicates: {list(d.predicates.keys())}")
        print(f"  Derived: {list(d.derived.keys())}")
        print(f"  Invariants: {len(d.invariants)}")
        print(f"  Actions: {list(d.actions.keys())}")
        for a in d.actions.values():
            print(f"    {a.name}({', '.join(p.name+':'+p.type_name for p in a.params)})")
            print(f"      require: {a.preconditions}")
            print(f"      effect:  {a.effects}")
            if a.conditional_effects:
                print(f"      when:    {a.conditional_effects}")
    for p in problems:
        print(f"\nProblem: {p.name} (domain: {p.domain_name})")
        print(f"  Objects: {list(p.objects.keys())}")
        print(f"  Init: {p.init}")
        print(f"  Goal: {p.goal}")
        if p.neg_goal:
            print(f"  NegGoal: {p.neg_goal}")
