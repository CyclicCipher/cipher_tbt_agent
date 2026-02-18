"""CTKG DSL parser — reads .ctkg files into graph objects.

The DSL is an indentation-based language for defining knowledge graphs.
Top-level blocks: concept, functor, adjunction. Fields are indented
under their parent block.

Example:
    concept addition
      domain arithmetic
      description "Single-digit addition"
      input digit op digit
      output carry digit
      requires counting via "grounds digit semantics"
      reversible
      process carry, ones = apply_op(a, op, b)

See DESIGN.md for the full grammar specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .graph import (
    Adjunction,
    Concept,
    Functor,
    KnowledgeGraph,
    Prerequisite,
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
    """A top-level block (concept, functor, adjunction) with its fields."""
    kind: str          # 'concept', 'functor', 'adjunction'
    name: str          # block name
    fields: List[Line] # indented field lines
    line: int          # line number of the block header


def group_blocks(lines: List[Line], source: str = '') -> List[Block]:
    """Group lines into top-level blocks."""
    blocks: List[Block] = []
    current: Optional[Block] = None

    TOP_KEYWORDS = {'concept', 'functor', 'adjunction'}

    for ln in lines:
        first_word = ln.content.split()[0] if ln.content else ''

        if ln.indent == 0 and first_word in TOP_KEYWORDS:
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
        elif ln.indent == 0 and first_word not in TOP_KEYWORDS:
            raise ParseError(
                f"Unexpected top-level keyword: '{first_word}'",
                ln.number, source)
        else:
            raise ParseError(
                f"Indented line outside any block: '{ln.content}'",
                ln.number, source)

    return blocks


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

def _parse_concept(block: Block, source: str = '') -> Tuple[Concept, List[Prerequisite]]:
    """Parse a concept block into a Concept and its prerequisites."""
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
        elif keyword == 'requires':
            # requires NAME via "ROLE"
            req_parts = rest.split(' via ', 1)
            req_name = req_parts[0].strip()
            req_role = _parse_string(req_parts[1]) if len(req_parts) > 1 else ''
            prereqs.append(Prerequisite(
                source=req_name,
                target=name,
                role=req_role,
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
        is_atomic=atomic,
        supports_reverse=reversible,
        pass_threshold=threshold,
        max_epochs=max_epochs,
        status='planned',  # DSL-loaded concepts start as planned
    )

    return concept, prereqs


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
        if block.kind == 'concept':
            concept, prereqs = _parse_concept(block, source)
            graph.add_concept(concept)
            for p in prereqs:
                graph.add_prerequisite(p)

        elif block.kind == 'functor':
            functor = _parse_functor(block, source)
            graph.functors[functor.name] = functor

        elif block.kind == 'adjunction':
            adj = _parse_adjunction(block, source)
            graph.adjunctions[adj.name] = adj

    return graph


def parse_file(path: str) -> KnowledgeGraph:
    """Parse a .ctkg file from disk."""
    with open(path, 'r') as f:
        text = f.read()
    return parse(text, source=path)


def merge(target: KnowledgeGraph, source: KnowledgeGraph) -> None:
    """Merge source graph into target (for loading multiple .ctkg files)."""
    for c in source.concepts.values():
        target.add_concept(c)
    for p in source.prerequisites:
        target.add_prerequisite(p)
    target.functors.update(source.functors)
    target.adjunctions.update(source.adjunctions)
