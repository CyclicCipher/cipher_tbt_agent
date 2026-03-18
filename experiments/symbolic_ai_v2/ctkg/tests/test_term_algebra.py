"""Tests for ctkg/core/term_algebra.py — Phase I gate."""
import pytest
from experiments.symbolic_ai_v2.ctkg.core.term_algebra import (
    Expr, atom, node, var, size, depth, is_ground, variables,
    match, substitute, anti_unify, anti_unify_list, skeleton,
)
from experiments.symbolic_ai_v2.ctkg.core.node import TOKEN_GRAPH

# ---------------------------------------------------------------------------
# Shorthand constructors for readability
# ---------------------------------------------------------------------------

def _pow(a, b):   return node('pow',  a, b)
def _mul(a, b):   return node('mul',  a, b)
def _add(a, b):   return node('add',  a, b)
def _pred(a):     return node('pred', a)
def _succ(a):     return node('succ', a)
def _d(f, x):     return node('d',    f, x)

x   = atom('x')
y   = atom('y')
c2  = atom('2')
c3  = atom('3')
c4  = atom('4')
c5  = atom('5')
c0  = atom('0')
c1  = atom('1')


# ---------------------------------------------------------------------------
# Expr construction and repr
# ---------------------------------------------------------------------------

class TestConstruct:
    def test_atom(self):
        e = atom('5')
        assert TOKEN_GRAPH.decode(e.head) == '5'
        assert e.args == ()
        assert not e.is_var

    def test_node(self):
        e = _add(c2, c3)
        assert TOKEN_GRAPH.decode(e.head) == 'add'
        assert e.args == (c2, c3)
        assert not e.is_var

    def test_var(self):
        v = var('V0')
        assert TOKEN_GRAPH.decode(v.head) == 'V0'
        assert v.is_var

    def test_repr_atom(self):
        assert repr(atom('7')) == '7'

    def test_repr_var(self):
        assert repr(var('V0')) == '?V0'

    def test_repr_node(self):
        assert repr(_pow(x, c2)) == 'pow(x, 2)'

    def test_frozen(self):
        e = atom('5')
        with pytest.raises(Exception):
            e.head = '6'  # type: ignore[misc]

    def test_hashable(self):
        s = {atom('5'), atom('5'), atom('6')}
        assert len(s) == 2


# ---------------------------------------------------------------------------
# Structural metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_size_leaf(self):
        assert size(atom('5')) == 1

    def test_size_node(self):
        assert size(_pow(x, c2)) == 3  # pow, x, 2

    def test_size_nested(self):
        assert size(_add(_mul(c2, x), c3)) == 5  # add, mul, 2, x, 3

    def test_depth_leaf(self):
        assert depth(atom('5')) == 0

    def test_depth_unary(self):
        assert depth(_pred(c3)) == 1

    def test_depth_nested(self):
        assert depth(_add(_mul(c2, x), c3)) == 2

    def test_is_ground_yes(self):
        assert is_ground(_pow(x, c2))

    def test_is_ground_no(self):
        assert not is_ground(_pow(x, var('V0')))

    def test_variables_empty(self):
        assert variables(_pow(x, c2)) == set()

    def test_variables_present(self):
        e = _mul(var('n'), _pow(var('f'), _pred(var('n'))))
        assert variables(e) == {'n', 'f'}


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

class TestMatch:
    def test_atom_exact(self):
        assert match(atom('5'), atom('5')) == {}

    def test_atom_mismatch(self):
        assert match(atom('5'), atom('6')) is None

    def test_var_binds(self):
        assert match(var('V0'), atom('5')) == {'V0': atom('5')}

    def test_var_binds_subtree(self):
        assert match(var('f'), _pow(x, c2)) == {'f': _pow(x, c2)}

    def test_var_consistent(self):
        # var appears twice; must bind to same value
        pat = _add(var('V'), var('V'))
        assert match(pat, _add(c2, c2)) == {'V': c2}

    def test_var_inconsistent(self):
        # var appears twice but with different values
        pat = _add(var('V'), var('V'))
        assert match(pat, _add(c2, c3)) is None

    def test_node_match(self):
        pat = _pow(x, var('n'))
        result = match(pat, _pow(x, c3))
        assert result == {'n': c3}

    def test_node_mismatch_head(self):
        assert match(_mul(var('a'), var('b')), _add(c2, c3)) is None

    def test_node_mismatch_arity(self):
        assert match(node('f', var('a')), node('f', var('a'), var('b'))) is None

    def test_multi_var(self):
        pat = _mul(var('n'), _pow(var('f'), _pred(var('n'))))
        expr = _mul(c3, _pow(x, _pred(c3)))
        result = match(pat, expr)
        assert result == {'n': c3, 'f': x}

    # Roadmap key test
    def test_roadmap_match(self):
        pat = _pow(var('f'), var('n'))
        result = match(pat, _pow(x, c3))
        assert result == {'f': x, 'n': c3}


# ---------------------------------------------------------------------------
# Substitution
# ---------------------------------------------------------------------------

class TestSubstitute:
    def test_no_vars(self):
        e = _pow(x, c2)
        assert substitute(e, {}) is e  # unchanged, same object

    def test_single_var(self):
        e = var('V0')
        assert substitute(e, {'V0': c5}) == c5

    def test_unbound_var(self):
        e = var('V0')
        assert substitute(e, {}) == var('V0')

    def test_nested(self):
        e = _mul(var('n'), _pow(var('f'), _pred(var('n'))))
        result = substitute(e, {'n': c3, 'f': x})
        expected = _mul(c3, _pow(x, _pred(c3)))
        assert result == expected

    # Roadmap key test
    def test_roadmap_substitute(self):
        e = _mul(var('n'), _pow(var('f'), _pred(var('n'))))
        result = substitute(e, {'n': c3, 'f': x})
        assert result == _mul(c3, _pow(x, _pred(c3)))


# ---------------------------------------------------------------------------
# Anti-unification
# ---------------------------------------------------------------------------

class TestAntiUnify:
    def test_equal(self):
        lgg, s1, s2 = anti_unify(_pow(x, c2), _pow(x, c2))
        assert lgg == _pow(x, c2)
        assert s1 == {}
        assert s2 == {}

    # Roadmap key test 1
    def test_roadmap_pow_exponent(self):
        lgg, s1, s2 = anti_unify(_pow(x, c2), _pow(x, c3))
        assert lgg == _pow(x, var('V0'))
        assert s1 == {'V0': c2}
        assert s2 == {'V0': c3}

    # Roadmap key test 4: x is NOT generalised
    def test_roadmap_mul_coefficient(self):
        lgg, s1, s2 = anti_unify(_mul(c2, x), _mul(c3, x))
        # Position of coefficient varies; position of x is constant
        assert lgg == _mul(var('V0'), x)
        assert s1 == {'V0': c2}
        assert s2 == {'V0': c3}

    def test_different_heads(self):
        lgg, s1, s2 = anti_unify(_add(c2, c3), _mul(c2, c3))
        # Different operators at root → single variable
        assert lgg.is_var
        assert s1[TOKEN_GRAPH.decode(lgg.head)] == _add(c2, c3)
        assert s2[TOKEN_GRAPH.decode(lgg.head)] == _mul(c2, c3)

    def test_same_pair_same_var(self):
        # anti_unify(add(x, x), add(y, y)): pair (x,y) appears twice
        # Both positions should get the SAME variable
        e1 = _add(x, x)
        e2 = _add(y, y)
        lgg, s1, s2 = anti_unify(e1, e2)
        # lgg should be add(V0, V0) — both positions the same var
        assert TOKEN_GRAPH.decode(lgg.head) == 'add'
        assert lgg.args[0] == lgg.args[1]
        assert lgg.args[0].is_var

    def test_roundtrip(self):
        e1 = _mul(c2, _pow(x, c3))
        e2 = _mul(c5, _pow(x, c4))
        lgg, s1, s2 = anti_unify(e1, e2)
        assert substitute(lgg, s1) == e1
        assert substitute(lgg, s2) == e2


# ---------------------------------------------------------------------------
# Anti-unify list
# ---------------------------------------------------------------------------

class TestAntiUnifyList:
    def test_single(self):
        e = _pow(x, c2)
        lgg, substs = anti_unify_list([e])
        assert lgg == e
        assert substs == [{}]

    def test_two_same(self):
        e = _pow(x, c2)
        lgg, substs = anti_unify_list([e, e])
        assert lgg == e
        assert substs == [{}, {}]

    def test_two_different_exponents(self):
        e1 = _pow(x, c2)
        e2 = _pow(x, c3)
        lgg, substs = anti_unify_list([e1, e2])
        assert lgg == _pow(x, var('V0'))
        assert substs[0] == {'V0': c2}
        assert substs[1] == {'V0': c3}

    def test_three_exprs(self):
        # d(pow(x, 2)), d(pow(x, 3)), d(pow(x, 4))
        exprs = [_d(_pow(x, c2), x), _d(_pow(x, c3), x), _d(_pow(x, c4), x)]
        lgg, substs = anti_unify_list(exprs)
        # Exponent position should be a variable; x and 'd' are constant
        assert TOKEN_GRAPH.decode(lgg.head) == 'd'
        assert lgg.args[1] == x          # second arg of d is x, constant
        inner = lgg.args[0]
        assert TOKEN_GRAPH.decode(inner.head) == 'pow'
        assert inner.args[0] == x        # base x is constant
        assert inner.args[1].is_var      # exponent is a variable
        vname = TOKEN_GRAPH.decode(inner.args[1].head)
        assert substs[0][vname] == c2
        assert substs[1][vname] == c3
        assert substs[2][vname] == c4

    def test_roundtrip_all(self):
        exprs = [_mul(c2, x), _mul(c3, x), _mul(c5, x)]
        lgg, substs = anti_unify_list(exprs)
        for e, s in zip(exprs, substs):
            assert substitute(lgg, s) == e

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            anti_unify_list([])


# ---------------------------------------------------------------------------
# Skeleton
# ---------------------------------------------------------------------------

class TestSkeleton:
    def test_atom(self):
        assert skeleton(atom('5')) == atom('_')

    def test_binary(self):
        e = _add(_mul(c2, x), c3)
        sk = skeleton(e)
        expected = _add(_mul(atom('_'), atom('_')), atom('_'))
        assert sk == expected

    def test_same_skeleton_different_atoms(self):
        e1 = _pow(x, c2)
        e2 = _pow(y, c3)
        assert skeleton(e1) == skeleton(e2)

    def test_different_skeleton_different_arity(self):
        e1 = _add(c2, c3)
        e2 = _pred(c3)
        assert skeleton(e1) != skeleton(e2)

    def test_different_skeleton_different_op(self):
        e1 = _add(c2, c3)
        e2 = _mul(c2, c3)
        assert skeleton(e1) != skeleton(e2)
