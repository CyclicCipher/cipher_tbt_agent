"""CTKG DSL parser — reads .ctkg files into graph objects.

The DSL is an indentation-based language for defining knowledge graphs
with universal type primitives.

Top-level blocks: type, concept, functor, adjunction.
Fields are indented under their parent block.

Example:
    type digit = symbol(0, 1, 2, 3, 4, 5, 6, 7, 8, 9) ordered
    type op = symbol(ADD, SUB)

    concept addition
      domain arithmetic
      description "Single-digit addition"
      input digit op digit
      output carry digit
      requires counting via "grounds digit semantics"
      reversible
      process
        result = fold(b, a, succ)
        carry = compare(result, 9)

See DESIGN.md for the full grammar specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import re

from .graph import (
    Adjunction,
    Challenge,
    Concept,
    Functor,
    Interface,
    KnowledgeGraph,
    Override,
    Prerequisite,
    TypeDef,
)


# ---------------------------------------------------------------------------
# Parse errors
# ---------------------------------------------------------------------------

class ParseError(Exception):
    """Error during .ctkg parsing, with line number."""
    def __init__(self, message: str, line: int = 0, source: str = ''):
        self.line = line
        self.source = source
        loc = f"{source}:{line}" if source else f"line {line}"
        super().__init__(f"{loc}: {message}")


# ---------------------------------------------------------------------------
# Tokenizer (line-based)
# ---------------------------------------------------------------------------

@dataclass
class Line:
    """A non-empty, non-comment line from a .ctkg file."""
    number: int       # 1-indexed line number
    indent: int       # number of leading spaces
    content: str      # stripped content (no leading whitespace)
    raw: str = ''     # original line


def tokenize(text: str) -> List[Line]:
    """Split text into Line objects, stripping comments and blank lines."""
    lines: List[Line] = []
    for i, raw in enumerate(text.split('\n'), start=1):
        # Strip trailing whitespace
        stripped = raw.rstrip()
        if not stripped:
            continue
        # Strip comments (-- to end of line, but not inside strings)
        content = _strip_comment(stripped)
        if not content.strip():
            continue
        # Measure indent
        indent = len(content) - len(content.lstrip())
        lines.append(Line(
            number=i,
            indent=indent,
            content=content.strip(),
            raw=raw,
        ))
    return lines


def _strip_comment(line: str) -> str:
    """Remove -- comments that aren't inside quoted strings."""
    in_string = False
    for i, ch in enumerate(line):
        if ch == '"':
            in_string = not in_string
        elif ch == '-' and not in_string and i + 1 < len(line) and line[i + 1] == '-':
            return line[:i]
    return line


# ---------------------------------------------------------------------------
# Block grouping
# ---------------------------------------------------------------------------

@dataclass
class Block:
    """A top-level block (type, concept, functor, adjunction) with fields."""
    kind: str          # 'type', 'concept', 'functor', 'adjunction'
    name: str          # block name (or full header for type)
    fields: List[Line] # indented field lines
    line: int          # line number of the block header
    header: str = ''   # full header text (for single-line blocks like type)


def group_blocks(lines: List[Line], source: str = '') -> List[Block]:
    """Group lines into top-level blocks."""
    blocks: List[Block] = []
    current: Optional[Block] = None

    TOP_KEYWORDS = {'concept', 'functor', 'adjunction', 'interface'}

    for ln in lines:
        first_word = ln.content.split()[0] if ln.content else ''

        # Type definitions are single-line: type NAME = CONSTRUCTOR(...)
        if ln.indent == 0 and first_word == 'type':
            blocks.append(Block(
                kind='type',
                name='',  # parsed from header
                fields=[],
                line=ln.number,
                header=ln.content,
            ))
            current = None  # type blocks don't have children
        elif ln.indent == 0 and first_word in TOP_KEYWORDS:
            # Start new block
            parts = ln.content.split(None, 1)
            if len(parts) < 2:
                raise ParseError(
                    f"'{first_word}' requires a name", ln.number, source)
            current = Block(
                kind=parts[0],
                name=parts[1],
                fields=[],
                line=ln.number,
            )
            blocks.append(current)
        elif current is not None and ln.indent > 0:
            current.fields.append(ln)
        elif ln.indent == 0:
            raise ParseError(
                f"Unexpected top-level keyword: '{first_word}'",
                ln.number, source)
        else:
            raise ParseError(
                f"Indented line outside any block: '{ln.content}'",
                ln.number, source)

    return blocks


# ---------------------------------------------------------------------------
# Type definition parser
# ---------------------------------------------------------------------------

def _parse_type_def(block: Block, source: str = '') -> TypeDef:
    """Parse a type definition line.

    Syntax:
        type NAME = CONSTRUCTOR(param1, param2, ...) [annotation1] [annotation2]
        type NAME = CONSTRUCTOR [annotation1]

    Examples:
        type digit = symbol(0, 1, 2, 3, 4, 5, 6, 7, 8, 9) ordered
        type op = symbol(ADD, SUB)
        type carry = symbol(0, 1)
        type count_seq = seq(digit)
        type scratchpad = tuple(carry, digit, carry, digit, digit)
        type result = tagged(success: nat, error: bool)
    """
    header = block.header
    # Strip 'type' keyword
    rest = header[4:].strip()

    # Split on '='
    eq_pos = rest.find('=')
    if eq_pos < 0:
        raise ParseError(
            f"Type definition requires '=': '{header}'",
            block.line, source)

    name = rest[:eq_pos].strip()
    if not name or not re.match(r'^[a-zA-Z_]\w*$', name):
        raise ParseError(
            f"Invalid type name: '{name}'", block.line, source)

    rhs = rest[eq_pos + 1:].strip()

    # Parse constructor and optional params
    # Formats: "constructor" or "constructor(p1, p2, ...)" optionally followed
    # by annotation words
    annotations: Set[str] = set()

    paren_open = rhs.find('(')
    if paren_open >= 0:
        constructor = rhs[:paren_open].strip()
        # Find matching close paren
        paren_close = rhs.rfind(')')
        if paren_close < 0:
            raise ParseError(
                f"Unmatched '(' in type definition: '{header}'",
                block.line, source)
        params_str = rhs[paren_open + 1:paren_close]
        params = [p.strip() for p in params_str.split(',') if p.strip()]
        # Everything after closing paren = annotations
        after_paren = rhs[paren_close + 1:].strip()
        if after_paren:
            annotations = set(after_paren.split())
    else:
        # No params — constructor is first word, rest are annotations
        parts = rhs.split()
        constructor = parts[0]
        params = []
        annotations = set(parts[1:]) if len(parts) > 1 else set()

    return TypeDef(
        name=name,
        constructor=constructor,
        params=params,
        annotations=annotations,
    )


# ---------------------------------------------------------------------------
# Field parsing helpers
# ---------------------------------------------------------------------------

def _parse_string(text: str) -> str:
    """Extract a quoted string, or return the text as-is if unquoted."""
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    return text


def _parse_type_list(text: str) -> List[str]:
    """Parse a space-separated type list."""
    return text.split()


# ---------------------------------------------------------------------------
# Concept parser
# ---------------------------------------------------------------------------

def _parse_concept(block: Block, source: str = '') -> Tuple[
        Concept, List[Prerequisite], List[Challenge], List[Override]]:
    """Parse a concept block into a Concept, prerequisites, challenges, overrides."""
    name = block.name
    domain = ''
    description = ''
    input_type: List[str] = []
    output_type: List[str] = []
    atomic = False
    reversible = False
    threshold = 0.95
    max_epochs = 100
    process_lines: List[str] = []
    prereqs: List[Prerequisite] = []
    challenges: List[Challenge] = []
    overrides_list: List[Override] = []
    tier = 'theorem'
    assumes: List[str] = []
    defaults: Dict[str, str] = {}

    i = 0
    fields = block.fields
    while i < len(fields):
        ln = fields[i]
        parts = ln.content.split(None, 1)
        keyword = parts[0]
        rest = parts[1] if len(parts) > 1 else ''

        if keyword == 'domain':
            domain = rest.strip()
        elif keyword == 'description':
            description = _parse_string(rest)
        elif keyword == 'input':
            input_type = _parse_type_list(rest)
        elif keyword == 'output':
            output_type = _parse_type_list(rest)
        elif keyword == 'atomic':
            atomic = True
        elif keyword == 'reversible':
            reversible = True
        elif keyword == 'threshold':
            threshold = float(rest.strip())
        elif keyword == 'max_epochs':
            max_epochs = int(rest.strip())
        elif keyword == 'tier':
            tier = rest.strip()
            if tier not in ('axiom', 'theorem', 'conjecture', 'heuristic'):
                raise ParseError(
                    f"Invalid tier '{tier}' — must be axiom, theorem, "
                    f"conjecture, or heuristic", ln.number, source)
        elif keyword == 'assumes':
            assumes = rest.split()
        elif keyword == 'default':
            # default NAME = VALUE
            eq_pos = rest.find('=')
            if eq_pos < 0:
                raise ParseError(
                    f"Default requires '=': '{ln.content}'",
                    ln.number, source)
            prop = rest[:eq_pos].strip()
            val = rest[eq_pos + 1:].strip()
            defaults[prop] = val
        elif keyword == 'requires':
            # requires NAME via "ROLE" [0.95] assuming ASSUMPTION [STATUS]
            # Transfer probability and assuming are both optional
            req_parts = rest.split(' via ', 1)
            req_name = req_parts[0].strip()
            req_role = ''
            transfer_prob = 1.0
            assuming = None
            assumption_status = 'derived'
            if len(req_parts) > 1:
                role_and_rest = req_parts[1]
                # Check for 'assuming' clause
                assuming_match = re.search(
                    r'\bassuming\s+(\S+)(?:\s+\[(\w+)\])?\s*$',
                    role_and_rest)
                if assuming_match:
                    assuming = assuming_match.group(1)
                    if assuming_match.group(2):
                        assumption_status = assuming_match.group(2)
                    role_and_rest = role_and_rest[:assuming_match.start()].strip()
                # Check for trailing [probability]
                prob_match = re.search(r'\[(\d*\.?\d+)\]\s*$', role_and_rest)
                if prob_match:
                    transfer_prob = float(prob_match.group(1))
                    role_and_rest = role_and_rest[:prob_match.start()].strip()
                req_role = _parse_string(role_and_rest)
            prereqs.append(Prerequisite(
                source=req_name,
                target=name,
                role=req_role,
                transfer_probability=transfer_prob,
                assuming=assuming,
                assumption_status=assumption_status,
            ))
        elif keyword == 'challenges':
            # challenges NAME via "REASON"
            ch_parts = rest.split(' via ', 1)
            ch_target = ch_parts[0].strip()
            ch_role = ''
            if len(ch_parts) > 1:
                ch_role = _parse_string(ch_parts[1])
            challenges.append(Challenge(
                source=name,
                target=ch_target,
                role=ch_role,
            ))
        elif keyword == 'overrides':
            # overrides NAME with PROP = VALUE via "REASON"
            ov_parts = rest.split(' with ', 1)
            ov_target = ov_parts[0].strip()
            ov_prop = ''
            ov_val = ''
            ov_reason = ''
            if len(ov_parts) > 1:
                prop_and_rest = ov_parts[1]
                # Split on 'via' for reason
                via_parts = prop_and_rest.split(' via ', 1)
                prop_eq = via_parts[0]
                if len(via_parts) > 1:
                    ov_reason = _parse_string(via_parts[1])
                # Parse PROP = VALUE
                eq_pos = prop_eq.find('=')
                if eq_pos >= 0:
                    ov_prop = prop_eq[:eq_pos].strip()
                    ov_val = prop_eq[eq_pos + 1:].strip()
            overrides_list.append(Override(
                instance=name,
                default_concept=ov_target,
                property=ov_prop,
                value=ov_val,
                reason=ov_reason,
            ))
        elif keyword == 'process':
            # Process can be single-line or multi-line (indented block)
            if rest.strip():
                process_lines.append(rest.strip())
            # Collect any continuation lines (more deeply indented)
            base_indent = ln.indent
            while i + 1 < len(fields) and fields[i + 1].indent > base_indent:
                i += 1
                process_lines.append(fields[i].content)
        else:
            raise ParseError(
                f"Unknown concept field: '{keyword}'",
                ln.number, source)
        i += 1

    concept = Concept(
        name=name,
        description=description,
        domain=domain,
        input_type=input_type,
        output_type=output_type,
        process=process_lines,
        is_atomic=atomic,
        supports_reverse=reversible,
        pass_threshold=threshold,
        max_epochs=max_epochs,
        status='planned',  # DSL-loaded concepts start as planned
        tier=tier,
        assumes=assumes,
        defaults=defaults,
    )

    return concept, prereqs, challenges, overrides_list


# ---------------------------------------------------------------------------
# Functor parser
# ---------------------------------------------------------------------------

def _parse_functor(block: Block, source: str = '') -> Functor:
    """Parse a functor block."""
    name = block.name
    source_domain = ''
    target_domain = ''
    concept_map: Dict[str, str] = {}
    preserves: List[str] = []

    for ln in block.fields:
        parts = ln.content.split(None, 1)
        keyword = parts[0]
        rest = parts[1] if len(parts) > 1 else ''

        if keyword == 'from':
            # from NAME to NAME
            ft_parts = rest.split(' to ', 1)
            source_domain = ft_parts[0].strip()
            target_domain = ft_parts[1].strip() if len(ft_parts) > 1 else ''
        elif keyword == 'map':
            # map NAME -> NAME
            map_parts = rest.split('->', 1)
            if len(map_parts) != 2:
                raise ParseError(
                    f"Functor map requires 'NAME -> NAME', got: '{rest}'",
                    ln.number, source)
            concept_map[map_parts[0].strip()] = map_parts[1].strip()
        elif keyword == 'preserves':
            preserves.append(rest.strip())
        else:
            raise ParseError(
                f"Unknown functor field: '{keyword}'",
                ln.number, source)

    return Functor(
        name=name,
        source_domain=source_domain,
        target_domain=target_domain,
        concept_map=concept_map,
        preserves=preserves,
    )


# ---------------------------------------------------------------------------
# Adjunction parser
# ---------------------------------------------------------------------------

def _parse_adjunction(block: Block, source: str = '') -> Adjunction:
    """Parse an adjunction block."""
    name = block.name
    forward = ''
    inverse = ''
    unit = ''
    counit = ''

    for ln in block.fields:
        parts = ln.content.split(None, 1)
        keyword = parts[0]
        rest = parts[1] if len(parts) > 1 else ''

        if keyword == 'forward':
            forward = rest.strip()
        elif keyword == 'inverse':
            inverse = rest.strip()
        elif keyword == 'unit':
            unit = rest.strip()
        elif keyword == 'counit':
            counit = rest.strip()
        else:
            raise ParseError(
                f"Unknown adjunction field: '{keyword}'",
                ln.number, source)

    return Adjunction(
        name=name,
        forward=forward,
        inverse=inverse,
        unit=unit,
        counit=counit,
    )


# ---------------------------------------------------------------------------
# Interface parser
# ---------------------------------------------------------------------------

def _parse_interface(block: Block, source: str = '') -> Interface:
    """Parse an interface block.

    Syntax:
        interface NAME
          exports types TYPE1 TYPE2 ...
          exports concepts CONCEPT1 CONCEPT2 ...
    """
    name = block.name
    types: List[str] = []
    concepts: List[str] = []

    for ln in block.fields:
        parts = ln.content.split(None, 2)
        keyword = parts[0]
        if keyword == 'exports' and len(parts) >= 3:
            kind = parts[1]
            names = parts[2].split()
            if kind == 'types':
                types.extend(names)
            elif kind == 'concepts':
                concepts.extend(names)
            else:
                raise ParseError(
                    f"Unknown exports kind: '{kind}' "
                    f"(expected 'types' or 'concepts')",
                    ln.number, source)
        else:
            raise ParseError(
                f"Unknown interface field: '{keyword}'",
                ln.number, source)

    return Interface(name=name, types=types, concepts=concepts)


# ---------------------------------------------------------------------------
# Top-level parse function
# ---------------------------------------------------------------------------

def parse(text: str, source: str = '') -> KnowledgeGraph:
    """Parse a .ctkg file into a KnowledgeGraph.

    Args:
        text: contents of a .ctkg file
        source: filename for error messages

    Returns:
        A populated KnowledgeGraph.

    Raises:
        ParseError: on syntax errors (with line number).
    """
    lines = tokenize(text)
    blocks = group_blocks(lines, source)

    graph = KnowledgeGraph()

    for block in blocks:
        if block.kind == 'type':
            typedef = _parse_type_def(block, source)
            graph.add_type(typedef)

        elif block.kind == 'concept':
            concept, prereqs, challenges, overrides_list = _parse_concept(
                block, source)
            graph.add_concept(concept)
            for p in prereqs:
                graph.add_prerequisite(p)
            for ch in challenges:
                graph.add_challenge(ch)
            for ov in overrides_list:
                graph.add_override(ov)

        elif block.kind == 'functor':
            functor = _parse_functor(block, source)
            graph.functors[functor.name] = functor

        elif block.kind == 'adjunction':
            adj = _parse_adjunction(block, source)
            graph.adjunctions[adj.name] = adj

        elif block.kind == 'interface':
            iface = _parse_interface(block, source)
            graph.interfaces[iface.name] = iface

    return graph


def parse_file(path: str) -> KnowledgeGraph:
    """Parse a .ctkg file from disk."""
    with open(path, 'r') as f:
        text = f.read()
    return parse(text, source=path)


def merge(target: KnowledgeGraph, source: KnowledgeGraph) -> None:
    """Merge source graph into target (for loading multiple .ctkg files).

    For sheaf-aware merging with consistency checks, use
    KnowledgeGraph.sheaf_merge() instead.
    """
    for t in source.types.values():
        target.add_type(t)
    for c in source.concepts.values():
        target.add_concept(c)
    for p in source.prerequisites:
        target.add_prerequisite(p)
    for ch in source.challenges:
        target.add_challenge(ch)
    for ov in source.overrides:
        target.add_override(ov)
    target.functors.update(source.functors)
    target.adjunctions.update(source.adjunctions)
    target.interfaces.update(source.interfaces)
