"""Complete CTKG — all domains from counting to differential equations,
propositional logic to category theory, and the science target.

This is the master graph. Most concepts have status='planned' (no generator
yet). The graph structure defines the COMPLETE prerequisite chain. The
curriculum compiler uses this to:
  1. Validate that no prerequisites are missing
  2. Generate valid training orders for any target concept
  3. Show what needs to be built next (frontier of planned concepts)

Domains:
  arithmetic    — counting, ordinality, single/multi-digit +/-
  arithmetic_ex — multiplication, division, negative numbers, fractions
  algebra       — variables, expressions, equations, polynomials
  functions     — composition, inverses, exp/log/trig
  calculus      — limits, derivatives, integrals
  ode           — ordinary differential equations, Laplace transforms
  logic         — propositional, predicate, set theory, relations
  abstract_alg  — groups, rings, fields, categories
  science       — physics foundations, damped harmonic oscillator

Target calculations:
  1. Analytically solve ODEs up to 3rd order
  2. Solve ODEs via Laplace transform
  3. Derive the impulse response of a damped harmonic oscillator
     (or equivalently, analyze an RLC circuit)
"""

from ..graph import Concept, KnowledgeGraph, Prerequisite


# ---------------------------------------------------------------------------
# Helpers for concise concept/edge definitions
# ---------------------------------------------------------------------------

def _c(name, desc, domain, *, inp=None, out=None, gen=None, n_res=None,
       n_prob=None, atomic=False, reverse=False, status='planned',
       threshold=0.95, max_ep=100):
    return Concept(
        name=name, description=desc, domain=domain,
        input_type=inp or [], output_type=out or [],
        generator_class=gen, n_result=n_res, n_problems=n_prob,
        is_atomic=atomic, supports_reverse=reverse, status=status,
        pass_threshold=threshold, max_epochs=max_ep,
    )


def _p(source, target, role, *, cod=None, dom=None, inv=False):
    return Prerequisite(
        source=source, target=target, role=role,
        codomain_type=cod or [], domain_type=dom or [],
        invertible=inv,
    )


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_full_graph() -> KnowledgeGraph:
    """Build the complete CTKG across all domains."""
    g = KnowledgeGraph()

    # ===================================================================
    # ARITHMETIC (implemented — maps to existing generators)
    # ===================================================================

    for c in [
        _c('query_counting',
           'Count DOTs or TENs given a query',
           'arithmetic',
           inp=['DOT_TEN_sequence', 'query_token'], out=['digit'],
           gen='QueryCountingGenerator', n_res=1, n_prob=100,
           status='implemented'),

        _c('combined_counting',
           'Count both DOTs and TENs via count-up process with STOP',
           'arithmetic',
           inp=['DOT_TEN_sequence'], out=['count_up_sequence'],
           gen='CombinedCountingGenerator', n_res=20, n_prob=100,
           status='implemented'),

        _c('single_digit_arithmetic',
           'Single-digit +/- producing carry + ones',
           'arithmetic',
           inp=['digit', 'op', 'digit'], out=['carry', 'digit'],
           gen='SingleDigitArithmeticGenerator', n_res=2, n_prob=155,
           atomic=True, reverse=True, status='implemented'),

        _c('two_digit_single_arithmetic',
           'Two-digit +/- single-digit with column scratchpad',
           'arithmetic',
           inp=['digit_pair', 'op', 'digit'], out=['column_scratchpad'],
           gen='TwoDigitSingleArithmeticGenerator', n_res=21, n_prob=1800,
           status='implemented'),

        _c('two_digit_arithmetic',
           'Two-digit +/- two-digit with column scratchpad',
           'arithmetic',
           inp=['digit_pair', 'op', 'digit_pair'], out=['column_scratchpad'],
           gen='TwoDigitArithmeticGenerator', n_res=21, n_prob=12195,
           status='implemented'),
    ]:
        g.add_concept(c)

    # ===================================================================
    # ARITHMETIC — planned stages from CONTINUATION.md
    # ===================================================================

    for c in [
        _c('ordinality',
           'Successor/predecessor: SUCC(4)=5, PRED(7)=6 (mod-10)',
           'arithmetic',
           inp=['digit', 'SUCC_or_PRED'], out=['digit'],
           n_prob=20, atomic=True),

        _c('comparison',
           'Digit comparison: GT, LT, EQ',
           'arithmetic',
           inp=['digit', 'digit'], out=['comparison_token'],
           n_prob=100),

        _c('counting_addition',
           'Addition as counting-up: 3+4 WORK 4 5 6 7 = 0 7',
           'arithmetic',
           inp=['digit', 'plus', 'digit'], out=['count_sequence', 'carry', 'digit'],
           n_prob=100, reverse=True),

        _c('counting_subtraction',
           'Subtraction as counting-down: 7-3 WORK 6 5 4 = 0 4',
           'arithmetic',
           inp=['digit', 'minus', 'digit'], out=['count_sequence', 'borrow', 'digit'],
           n_prob=55),
    ]:
        g.add_concept(c)

    # ===================================================================
    # EXTENDED ARITHMETIC
    # ===================================================================

    for c in [
        _c('place_value',
           'Positional notation: ones, tens, hundreds columns',
           'arithmetic_ex',
           inp=['multi_digit_number'], out=['digit_columns'],
           n_prob=90),

        _c('multiplication_concept',
           'Multiplication as repeated addition: 3*4 = 4+4+4',
           'arithmetic_ex',
           inp=['digit', 'times', 'digit'], out=['repeated_sum', 'product'],
           n_prob=100),

        _c('multiplication_table',
           'Single-digit multiplication facts',
           'arithmetic_ex',
           inp=['digit', 'times', 'digit'], out=['carry', 'digit'],
           n_prob=100, atomic=True),

        _c('long_multiplication',
           'Multi-digit multiplication via partial products',
           'arithmetic_ex',
           inp=['number', 'times', 'number'], out=['partial_products', 'product']),

        _c('division_concept',
           'Division as inverse of multiplication: 12/3 = ? because 3*? = 12',
           'arithmetic_ex',
           inp=['number', 'div', 'digit'], out=['quotient', 'remainder'],
           n_prob=90, reverse=True),

        _c('long_division',
           'Multi-digit division algorithm',
           'arithmetic_ex',
           inp=['number', 'div', 'number'], out=['division_scratchpad']),

        _c('negative_numbers',
           'Extending the number line below zero; sign rules',
           'arithmetic_ex',
           inp=['signed_number'], out=['signed_number']),

        _c('signed_arithmetic',
           'Addition and subtraction with negative numbers',
           'arithmetic_ex',
           inp=['signed_number', 'op', 'signed_number'], out=['signed_number']),

        _c('fraction_concept',
           'Fractions as parts of a whole: a/b',
           'arithmetic_ex',
           inp=['numerator', 'denominator'], out=['fraction']),

        _c('fraction_equivalence',
           'Equivalent fractions: a/b = (ak)/(bk)',
           'arithmetic_ex',
           inp=['fraction'], out=['fraction']),

        _c('fraction_arithmetic',
           'Add, subtract, multiply, divide fractions',
           'arithmetic_ex',
           inp=['fraction', 'op', 'fraction'], out=['fraction']),

        _c('decimal_representation',
           'Fractions as decimal numbers',
           'arithmetic_ex',
           inp=['fraction'], out=['decimal']),
    ]:
        g.add_concept(c)

    # ===================================================================
    # ALGEBRA
    # ===================================================================

    for c in [
        _c('variable_concept',
           'Letters represent unknown numbers; substitution',
           'algebra',
           inp=['variable', 'value'], out=['evaluated_expression']),

        _c('algebraic_expression',
           'Combining variables with arithmetic operations',
           'algebra',
           inp=['expression_tokens'], out=['expression']),

        _c('expression_simplification',
           'Combining like terms, distributing, factoring out',
           'algebra',
           inp=['expression'], out=['simplified_expression']),

        _c('linear_equation',
           'Solve ax + b = c for x',
           'algebra',
           inp=['linear_equation'], out=['solution'],
           reverse=True),

        _c('quadratic_formula',
           'Solve ax^2 + bx + c = 0 via completing the square or formula',
           'algebra',
           inp=['quadratic_equation'], out=['solutions']),

        _c('polynomial_arithmetic',
           'Add and multiply polynomials',
           'algebra',
           inp=['polynomial', 'op', 'polynomial'], out=['polynomial']),

        _c('polynomial_factoring',
           'Factor polynomials into irreducibles',
           'algebra',
           inp=['polynomial'], out=['factored_form']),

        _c('exponents',
           'Power rules: a^m * a^n = a^(m+n), (a^m)^n = a^(mn)',
           'algebra',
           inp=['base', 'exponent'], out=['power_expression'],
           atomic=True),
    ]:
        g.add_concept(c)

    # ===================================================================
    # FUNCTIONS
    # ===================================================================

    for c in [
        _c('function_concept',
           'Functions as input-output mappings; f(x) notation',
           'functions',
           inp=['function_def', 'input'], out=['output']),

        _c('function_composition',
           'Composing functions: (f . g)(x) = f(g(x))',
           'functions',
           inp=['function', 'function', 'input'], out=['output']),

        _c('inverse_function',
           'Finding f^-1 such that f(f^-1(x)) = x',
           'functions',
           inp=['function'], out=['inverse_function'],
           reverse=True),

        _c('exponential_function',
           'a^x, e^x; growth/decay behavior',
           'functions',
           inp=['base', 'variable'], out=['exponential_expression']),

        _c('logarithm',
           'log_a(x), ln(x); inverse of exponential',
           'functions',
           inp=['exponential_expression'], out=['exponent'],
           reverse=True),

        _c('trigonometric_functions',
           'sin, cos, tan; unit circle definition',
           'functions',
           inp=['angle'], out=['trig_value'],
           atomic=True),

        _c('trig_identities',
           'sin^2 + cos^2 = 1, double-angle, sum formulas',
           'functions',
           inp=['trig_expression'], out=['simplified_trig']),
    ]:
        g.add_concept(c)

    # ===================================================================
    # CALCULUS
    # ===================================================================

    for c in [
        _c('limit_concept',
           'lim x->a f(x); approaching a value; epsilon-delta intuition',
           'calculus',
           inp=['function', 'point'], out=['limit_value']),

        _c('derivative_definition',
           "f'(x) = lim h->0 (f(x+h)-f(x))/h; rate of change",
           'calculus',
           inp=['function', 'point'], out=['derivative_value']),

        _c('power_rule',
           'd/dx x^n = n*x^(n-1)',
           'calculus',
           inp=['power_expression'], out=['derivative_expression'],
           atomic=True),

        _c('product_rule',
           "d/dx (f*g) = f'*g + f*g'",
           'calculus',
           inp=['product_expression'], out=['derivative_expression']),

        _c('quotient_rule',
           "d/dx (f/g) = (f'g - fg') / g^2",
           'calculus',
           inp=['quotient_expression'], out=['derivative_expression']),

        _c('chain_rule',
           "d/dx f(g(x)) = f'(g(x)) * g'(x)",
           'calculus',
           inp=['composite_function'], out=['derivative_expression']),

        _c('higher_derivatives',
           "f'', f''', f^(n); repeated differentiation",
           'calculus',
           inp=['function', 'order'], out=['derivative_expression']),

        _c('antiderivative',
           'F(x) such that F\'(x) = f(x); reverse of differentiation',
           'calculus',
           inp=['function'], out=['antiderivative_expression'],
           reverse=True),

        _c('integration_by_substitution',
           'integral f(g(x))g\'(x)dx = integral f(u)du',
           'calculus',
           inp=['integral_expression'], out=['integrated_expression']),

        _c('integration_by_parts',
           'integral u dv = uv - integral v du',
           'calculus',
           inp=['integral_expression'], out=['integrated_expression']),

        _c('partial_fractions',
           'Decompose P(x)/Q(x) into sum of simpler fractions for integration',
           'calculus',
           inp=['rational_expression'], out=['partial_fraction_form']),

        _c('definite_integral',
           'integral_a^b f(x)dx = F(b) - F(a); fundamental theorem',
           'calculus',
           inp=['function', 'bounds'], out=['numeric_value']),
    ]:
        g.add_concept(c)

    # ===================================================================
    # ORDINARY DIFFERENTIAL EQUATIONS
    # ===================================================================

    for c in [
        _c('ode_concept',
           'Equations involving derivatives: dy/dx = f(x,y)',
           'ode',
           inp=['ode_expression'], out=['solution_concept']),

        _c('separable_ode',
           'dy/dx = f(x)g(y); separate and integrate both sides',
           'ode',
           inp=['separable_ode'], out=['general_solution']),

        _c('first_order_linear_ode',
           'dy/dx + P(x)y = Q(x); integrating factor method',
           'ode',
           inp=['first_order_linear_ode'], out=['general_solution']),

        _c('characteristic_equation',
           'For ay\'\' + by\' + cy = 0: ar^2 + br + c = 0 gives r1, r2',
           'ode',
           inp=['linear_ode_coefficients'], out=['characteristic_roots']),

        _c('second_order_linear_ode',
           'ay\'\' + by\' + cy = f(x); homogeneous + particular',
           'ode',
           inp=['second_order_ode'], out=['general_solution']),

        _c('third_order_linear_ode',
           'ay\'\'\' + by\'\' + cy\' + dy = f(x); extending characteristic method',
           'ode',
           inp=['third_order_ode'], out=['general_solution']),

        _c('laplace_transform_definition',
           'L{f(t)} = integral_0^inf e^(-st) f(t) dt = F(s)',
           'ode',
           inp=['time_function'], out=['s_domain_function']),

        _c('laplace_transform_properties',
           'Linearity, shifting, derivative/integral transforms, convolution',
           'ode',
           inp=['s_domain_expression'], out=['s_domain_expression']),

        _c('inverse_laplace',
           'L^-1{F(s)} = f(t); partial fractions + table lookup',
           'ode',
           inp=['s_domain_function'], out=['time_function'],
           reverse=True),

        _c('ode_via_laplace',
           'Transform ODE to algebra in s-domain, solve, invert',
           'ode',
           inp=['ode_with_initial_conditions'], out=['particular_solution']),
    ]:
        g.add_concept(c)

    # ===================================================================
    # LOGIC — propositional through predicate
    # ===================================================================

    for c in [
        _c('proposition',
           'A statement that is true or false',
           'logic',
           inp=['statement'], out=['truth_value'],
           atomic=True),

        _c('logical_connectives',
           'AND, OR, NOT; combining propositions',
           'logic',
           inp=['proposition', 'connective', 'proposition'],
           out=['compound_proposition']),

        _c('truth_tables',
           'Systematically evaluate compound propositions',
           'logic',
           inp=['compound_proposition'], out=['truth_table']),

        _c('logical_equivalence',
           'De Morgan\'s laws, double negation, distribution',
           'logic',
           inp=['compound_proposition'], out=['equivalent_proposition']),

        _c('implication',
           'P -> Q; if-then statements; modus ponens',
           'logic',
           inp=['proposition', 'proposition'], out=['implication']),

        _c('contrapositive',
           'P -> Q iff ~Q -> ~P; proof by contrapositive',
           'logic',
           inp=['implication'], out=['contrapositive_implication'],
           reverse=True),

        _c('predicate_logic',
           'Statements with variables: P(x), Q(x,y)',
           'logic',
           inp=['predicate', 'variable'], out=['open_sentence']),

        _c('quantifiers',
           'For all (forall x P(x)) and there exists (exists x P(x))',
           'logic',
           inp=['predicate', 'quantifier'], out=['quantified_statement']),

        _c('quantifier_manipulation',
           'Negation of quantifiers, nesting, scope rules',
           'logic',
           inp=['quantified_statement'], out=['equivalent_statement']),

        _c('proof_by_induction',
           'Base case + inductive step => for all n',
           'logic',
           inp=['predicate_over_naturals'], out=['proof']),
    ]:
        g.add_concept(c)

    # ===================================================================
    # SET THEORY AND RELATIONS
    # ===================================================================

    for c in [
        _c('set_membership',
           'x in S; defining sets by property or enumeration',
           'logic',
           inp=['element', 'set'], out=['membership_truth']),

        _c('set_operations',
           'Union, intersection, complement, difference',
           'logic',
           inp=['set', 'set_op', 'set'], out=['set']),

        _c('subset_relations',
           'A subset B iff forall x: x in A -> x in B',
           'logic',
           inp=['set', 'set'], out=['subset_truth']),

        _c('relations',
           'R subset A x B; domain, codomain, image',
           'logic',
           inp=['set', 'set'], out=['relation']),

        _c('equivalence_relations',
           'Reflexive + symmetric + transitive; equivalence classes',
           'logic',
           inp=['relation'], out=['equivalence_classes']),

        _c('partial_orders',
           'Reflexive + antisymmetric + transitive; Hasse diagrams',
           'logic',
           inp=['relation'], out=['poset']),
    ]:
        g.add_concept(c)

    # ===================================================================
    # ABSTRACT ALGEBRA
    # ===================================================================

    for c in [
        _c('group_concept',
           '(G, *, e, inv): closure, associativity, identity, inverse',
           'abstract_alg',
           inp=['set', 'operation'], out=['group']),

        _c('group_properties',
           'Subgroups, cosets, Lagrange\'s theorem, homomorphisms',
           'abstract_alg',
           inp=['group'], out=['group_property']),

        _c('ring_concept',
           '(R, +, *): two operations, distributive law',
           'abstract_alg',
           inp=['set', 'addition', 'multiplication'], out=['ring']),

        _c('field_concept',
           'Ring where every nonzero element has multiplicative inverse',
           'abstract_alg',
           inp=['ring'], out=['field']),

        _c('category_concept',
           'Objects + morphisms + composition; identity morphisms',
           'abstract_alg',
           inp=['objects', 'morphisms'], out=['category']),

        _c('functor_concept',
           'Structure-preserving map between categories',
           'abstract_alg',
           inp=['category', 'category'], out=['functor']),

        _c('natural_transformation',
           'Map between functors preserving commutativity',
           'abstract_alg',
           inp=['functor', 'functor'], out=['natural_transformation']),
    ]:
        g.add_concept(c)

    # ===================================================================
    # SCIENCE — damped harmonic oscillator
    # ===================================================================

    for c in [
        _c('physical_quantity',
           'Number + unit; dimensional consistency',
           'science',
           inp=['number', 'unit'], out=['quantity']),

        _c('dimensional_analysis',
           'Units must match in equations; checking correctness',
           'science',
           inp=['equation_with_units'], out=['dimensional_check']),

        _c('newtons_second_law',
           'F = ma; force equals mass times acceleration',
           'science',
           inp=['force', 'mass'], out=['acceleration'],
           atomic=True),

        _c('hookes_law',
           'F = -kx; restoring force proportional to displacement',
           'science',
           inp=['spring_constant', 'displacement'], out=['force'],
           atomic=True),

        _c('damping_force',
           'F = -cv; damping force proportional to velocity',
           'science',
           inp=['damping_coefficient', 'velocity'], out=['force'],
           atomic=True),

        _c('spring_mass_damper_setup',
           'Combine forces: mx\'\' + cx\' + kx = f(t) via Newton\'s law',
           'science',
           inp=['mass', 'damping', 'spring_constant', 'forcing'],
           out=['second_order_ode']),

        _c('damped_oscillator_solution',
           'Solve mx\'\' + cx\' + kx = delta(t) for impulse response',
           'science',
           inp=['second_order_ode', 'initial_conditions'],
           out=['time_domain_solution']),

        _c('oscillator_analysis',
           'Classify: underdamped (oscillation), critically damped, overdamped',
           'science',
           inp=['characteristic_roots'], out=['behavior_classification']),
    ]:
        g.add_concept(c)

    # ===================================================================
    # PREREQUISITES — Arithmetic
    # ===================================================================

    for p in [
        _p('query_counting', 'combined_counting',
           'Individual counting composes into dual counting'),
        _p('combined_counting', 'single_digit_arithmetic',
           'Counting grounds digit semantics'),
        _p('single_digit_arithmetic', 'two_digit_single_arithmetic',
           'Single-digit ops are column operations'),
        _p('two_digit_single_arithmetic', 'two_digit_arithmetic',
           'Bridge from one-variable to two-variable columns'),
        # Planned arithmetic stages
        _p('query_counting', 'ordinality',
           'Counting establishes digits; ordinality orders them'),
        _p('ordinality', 'comparison',
           'Successor ordering enables comparison'),
        _p('ordinality', 'counting_addition',
           'Successor function enables counting-up for addition'),
        _p('ordinality', 'counting_subtraction',
           'Predecessor function enables counting-down for subtraction'),
    ]:
        g.add_prerequisite(p)

    # ===================================================================
    # PREREQUISITES — Extended Arithmetic
    # ===================================================================

    for p in [
        _p('two_digit_arithmetic', 'place_value',
           'Multi-digit arithmetic teaches positional notation'),
        _p('single_digit_arithmetic', 'multiplication_concept',
           'Multiplication is repeated addition'),
        _p('multiplication_concept', 'multiplication_table',
           'Concept grounds the fact table'),
        _p('multiplication_table', 'long_multiplication',
           'Single-digit multiplication is the column operation'),
        _p('place_value', 'long_multiplication',
           'Positional notation enables column decomposition'),
        _p('multiplication_table', 'division_concept',
           'Division is the inverse of multiplication'),
        _p('division_concept', 'long_division',
           'Single-digit division is the sub-operation'),
        _p('place_value', 'long_division',
           'Positional notation enables column decomposition'),
        _p('comparison', 'negative_numbers',
           'Comparison is prerequisite for understanding sign'),
        _p('single_digit_arithmetic', 'negative_numbers',
           'Arithmetic on positive numbers extends to signed'),
        _p('negative_numbers', 'signed_arithmetic',
           'Sign concept enables signed operations'),
        _p('division_concept', 'fraction_concept',
           'Fractions generalize division'),
        _p('fraction_concept', 'fraction_equivalence',
           'Fractions must be established before equivalence'),
        _p('fraction_equivalence', 'fraction_arithmetic',
           'Equivalence (common denominators) enables fraction ops'),
        _p('multiplication_table', 'fraction_arithmetic',
           'Fraction ops require multiplication for cross-multiply'),
        _p('fraction_arithmetic', 'decimal_representation',
           'Decimals are fractions with power-of-10 denominators'),
    ]:
        g.add_prerequisite(p)

    # ===================================================================
    # PREREQUISITES — Algebra
    # ===================================================================

    for p in [
        _p('signed_arithmetic', 'variable_concept',
           'Arithmetic with signed numbers enables algebraic substitution'),
        _p('variable_concept', 'algebraic_expression',
           'Variables combine with operations to form expressions'),
        _p('signed_arithmetic', 'algebraic_expression',
           'Arithmetic operations apply to expressions'),
        _p('algebraic_expression', 'expression_simplification',
           'Expressions must exist before simplification'),
        _p('expression_simplification', 'linear_equation',
           'Simplification is the core operation in equation solving'),
        _p('linear_equation', 'quadratic_formula',
           'Linear solving techniques extend to quadratic'),
        _p('multiplication_table', 'exponents',
           'Exponents generalize repeated multiplication'),
        _p('exponents', 'polynomial_arithmetic',
           'Powers are the building blocks of polynomials'),
        _p('expression_simplification', 'polynomial_arithmetic',
           'Polynomial ops require simplification'),
        _p('polynomial_arithmetic', 'polynomial_factoring',
           'Must know polynomial arithmetic to factor'),
        _p('quadratic_formula', 'polynomial_factoring',
           'Quadratic formula finds roots for factoring'),
    ]:
        g.add_prerequisite(p)

    # ===================================================================
    # PREREQUISITES — Functions
    # ===================================================================

    for p in [
        _p('algebraic_expression', 'function_concept',
           'Expressions become function definitions'),
        _p('function_concept', 'function_composition',
           'Functions must exist before composing them'),
        _p('function_concept', 'inverse_function',
           'Functions must exist before finding inverses'),
        _p('linear_equation', 'inverse_function',
           'Equation solving finds the inverse'),
        _p('exponents', 'exponential_function',
           'Exponent rules ground exponential functions'),
        _p('function_concept', 'exponential_function',
           'Function concept frames exponential as a function'),
        _p('exponential_function', 'logarithm',
           'Logarithm is the inverse of exponential'),
        _p('inverse_function', 'logarithm',
           'Inverse function concept applies to exponential'),
        _p('function_concept', 'trigonometric_functions',
           'Trig functions are specific function types'),
        _p('trigonometric_functions', 'trig_identities',
           'Must know trig functions before their identities'),
        _p('expression_simplification', 'trig_identities',
           'Simplification skills apply to trig expressions'),
    ]:
        g.add_prerequisite(p)

    # ===================================================================
    # PREREQUISITES — Calculus
    # ===================================================================

    for p in [
        _p('function_concept', 'limit_concept',
           'Limits are defined for functions'),
        _p('fraction_arithmetic', 'limit_concept',
           'Limit evaluation often involves fraction manipulation'),
        _p('limit_concept', 'derivative_definition',
           'Derivative is defined as a limit'),
        _p('derivative_definition', 'power_rule',
           'Power rule derived from derivative definition'),
        _p('derivative_definition', 'product_rule',
           'Product rule derived from derivative definition'),
        _p('derivative_definition', 'quotient_rule',
           'Quotient rule derived from derivative definition'),
        _p('function_composition', 'chain_rule',
           'Chain rule applies to composed functions'),
        _p('derivative_definition', 'chain_rule',
           'Chain rule is a derivative rule'),
        _p('power_rule', 'higher_derivatives',
           'Higher derivatives apply rules repeatedly'),
        _p('product_rule', 'higher_derivatives',
           'Higher derivatives apply rules repeatedly'),
        _p('chain_rule', 'higher_derivatives',
           'Higher derivatives apply rules repeatedly'),
        _p('derivative_definition', 'antiderivative',
           'Antiderivative reverses differentiation'),
        _p('power_rule', 'antiderivative',
           'Power rule for integration: reverse of power rule'),
        _p('function_composition', 'integration_by_substitution',
           'Substitution reverses the chain rule'),
        _p('antiderivative', 'integration_by_substitution',
           'Substitution is an integration technique'),
        _p('product_rule', 'integration_by_parts',
           'By-parts reverses the product rule'),
        _p('antiderivative', 'integration_by_parts',
           'By-parts is an integration technique'),
        _p('polynomial_factoring', 'partial_fractions',
           'Factoring the denominator enables decomposition'),
        _p('fraction_arithmetic', 'partial_fractions',
           'Partial fractions are fraction manipulations'),
        _p('antiderivative', 'definite_integral',
           'Definite integral uses antiderivative (FTC)'),
    ]:
        g.add_prerequisite(p)

    # ===================================================================
    # PREREQUISITES — ODEs and Laplace
    # ===================================================================

    for p in [
        _p('derivative_definition', 'ode_concept',
           'ODEs are equations involving derivatives'),
        _p('ode_concept', 'separable_ode',
           'Separable is a specific ODE type'),
        _p('antiderivative', 'separable_ode',
           'Solving separable ODEs requires integration'),
        _p('ode_concept', 'first_order_linear_ode',
           'First-order linear is a specific ODE type'),
        _p('integration_by_parts', 'first_order_linear_ode',
           'Integrating factor method uses integration'),
        _p('exponential_function', 'first_order_linear_ode',
           'Integrating factor is an exponential'),
        _p('quadratic_formula', 'characteristic_equation',
           'Characteristic equation is a polynomial equation'),
        _p('ode_concept', 'characteristic_equation',
           'Characteristic equation comes from ODE theory'),
        _p('characteristic_equation', 'second_order_linear_ode',
           'Characteristic roots give homogeneous solution'),
        _p('exponential_function', 'second_order_linear_ode',
           'Solutions are exponentials (real or complex)'),
        _p('trigonometric_functions', 'second_order_linear_ode',
           'Complex roots give sin/cos solutions'),
        _p('second_order_linear_ode', 'third_order_linear_ode',
           'Third order extends the characteristic method'),
        _p('definite_integral', 'laplace_transform_definition',
           'Laplace transform is an improper integral'),
        _p('exponential_function', 'laplace_transform_definition',
           'Laplace kernel is e^(-st)'),
        _p('laplace_transform_definition', 'laplace_transform_properties',
           'Must know the transform before its properties'),
        _p('higher_derivatives', 'laplace_transform_properties',
           'Derivative property: L{f\'} = sF(s) - f(0)'),
        _p('laplace_transform_properties', 'inverse_laplace',
           'Properties enable inversion'),
        _p('partial_fractions', 'inverse_laplace',
           'Partial fractions decompose F(s) for inversion'),
        _p('inverse_laplace', 'ode_via_laplace',
           'Inversion recovers time-domain solution'),
        _p('laplace_transform_properties', 'ode_via_laplace',
           'Transform converts ODE to algebraic equation'),
        _p('linear_equation', 'ode_via_laplace',
           'Solving the algebraic equation in s-domain'),
    ]:
        g.add_prerequisite(p)

    # ===================================================================
    # PREREQUISITES — Logic
    # ===================================================================

    for p in [
        _p('proposition', 'logical_connectives',
           'Propositions are combined by connectives'),
        _p('logical_connectives', 'truth_tables',
           'Truth tables evaluate compound propositions'),
        _p('truth_tables', 'logical_equivalence',
           'Truth tables prove equivalences'),
        _p('logical_connectives', 'implication',
           'Implication is a connective'),
        _p('implication', 'contrapositive',
           'Contrapositive is equivalent to implication'),
        _p('logical_equivalence', 'contrapositive',
           'Equivalence reasoning proves contrapositive'),
        _p('proposition', 'predicate_logic',
           'Predicates generalize propositions with variables'),
        _p('variable_concept', 'predicate_logic',
           'Predicates use variables'),
        _p('predicate_logic', 'quantifiers',
           'Quantifiers bind variables in predicates'),
        _p('quantifiers', 'quantifier_manipulation',
           'Must know quantifiers before manipulating them'),
        _p('logical_equivalence', 'quantifier_manipulation',
           'Equivalence rules extend to quantifiers'),
        _p('quantifiers', 'proof_by_induction',
           'Induction proves universal statements over naturals'),
        _p('implication', 'proof_by_induction',
           'Induction step is an implication'),
    ]:
        g.add_prerequisite(p)

    # ===================================================================
    # PREREQUISITES — Set Theory and Relations
    # ===================================================================

    for p in [
        _p('predicate_logic', 'set_membership',
           'Sets defined by predicates'),
        _p('set_membership', 'set_operations',
           'Operations combine sets'),
        _p('set_operations', 'subset_relations',
           'Subset defined via set operations'),
        _p('set_operations', 'relations',
           'Relations are subsets of Cartesian products'),
        _p('relations', 'equivalence_relations',
           'Equivalence relations are specific relations'),
        _p('relations', 'partial_orders',
           'Partial orders are specific relations'),
    ]:
        g.add_prerequisite(p)

    # ===================================================================
    # PREREQUISITES — Abstract Algebra
    # ===================================================================

    for p in [
        _p('set_operations', 'group_concept',
           'Groups are sets with structure'),
        _p('signed_arithmetic', 'group_concept',
           'Integers under addition are the prototypical group'),
        _p('group_concept', 'group_properties',
           'Must know groups before studying their properties'),
        _p('equivalence_relations', 'group_properties',
           'Cosets use equivalence relations'),
        _p('group_concept', 'ring_concept',
           'Rings add a second operation to groups'),
        _p('multiplication_concept', 'ring_concept',
           'Ring multiplication generalizes number multiplication'),
        _p('ring_concept', 'field_concept',
           'Fields are rings with division'),
        _p('function_composition', 'category_concept',
           'Morphism composition generalizes function composition'),
        _p('group_concept', 'category_concept',
           'Groups are single-object categories'),
        _p('partial_orders', 'category_concept',
           'Posets are thin categories'),
        _p('category_concept', 'functor_concept',
           'Functors map between categories'),
        _p('functor_concept', 'natural_transformation',
           'Natural transformations map between functors'),
    ]:
        g.add_prerequisite(p)

    # ===================================================================
    # PREREQUISITES — Science (Damped Harmonic Oscillator)
    # ===================================================================

    for p in [
        _p('algebraic_expression', 'physical_quantity',
           'Physical quantities are numerical expressions with units'),
        _p('physical_quantity', 'dimensional_analysis',
           'Dimensional analysis checks unit consistency'),
        _p('physical_quantity', 'newtons_second_law',
           'F=ma relates physical quantities'),
        _p('physical_quantity', 'hookes_law',
           'F=-kx relates spring force to displacement'),
        _p('physical_quantity', 'damping_force',
           'F=-cv relates damping force to velocity'),
        _p('newtons_second_law', 'spring_mass_damper_setup',
           'Newton\'s law combines forces into equation of motion'),
        _p('hookes_law', 'spring_mass_damper_setup',
           'Spring force term in the ODE'),
        _p('damping_force', 'spring_mass_damper_setup',
           'Damping force term in the ODE'),
        _p('higher_derivatives', 'spring_mass_damper_setup',
           'Acceleration is second derivative of position'),
        _p('spring_mass_damper_setup', 'damped_oscillator_solution',
           'The ODE to solve'),
        _p('ode_via_laplace', 'damped_oscillator_solution',
           'Laplace transform method solves the ODE'),
        _p('second_order_linear_ode', 'damped_oscillator_solution',
           'Characteristic equation method also applies'),
        _p('damped_oscillator_solution', 'oscillator_analysis',
           'Solution determines behavior classification'),
        _p('comparison', 'oscillator_analysis',
           'Comparing discriminant to zero classifies damping'),
    ]:
        g.add_prerequisite(p)

    return g
