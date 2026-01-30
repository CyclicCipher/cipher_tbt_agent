"""
Concrete specification of abstract reasoning architecture.

Answers:
- How big is it?
- One subnet or multiple?
- Input/output dimensions?
- How does it connect to other brain regions?
"""

from architecture_validator import NetworkConfig

print("=" * 70)
print("ABSTRACT REASONING: CONCRETE ARCHITECTURE SPECIFICATION")
print("=" * 70)
print()

# ============================================================================
# CONTEXT: WHERE REASONING FITS IN THE FULL AGENT
# ============================================================================

print("POSITION IN FULL ARCHITECTURE:")
print()
print("  Position 0: Vision + Motor (sensory/motor)")
print("    Vision: Conv layers → (Ventral × Dorsal)")
print("    Output: 768 dims (512 ventral + 256 dorsal)")
print()
print("  Position 1: Association (multimodal integration)")
print("    Input: 768 dims from vision")
print("    Output: ~512 dims (integrated features)")
print()
print("  Position 2: ABSTRACT REASONING (compositional inference)")
print("    Input: 512 dims from association + language + memory")
print("    Output: Actions, memory updates, attention control")
print()
print("  Lateral: Working Memory (persistent state)")
print("    Bidirectional connections to reasoning")
print("    Stores evidence, hypotheses, conclusions")
print()

# ============================================================================
# BUILD THE REASONING ARCHITECTURE
# ============================================================================

config = NetworkConfig()

print("=" * 70)
print("BUILDING REASONING SUBNET SPECIFICATIONS")
print("=" * 70)
print()

# ----------------------------------------------------------------------------
# INPUTS TO REASONING SYSTEM
# ----------------------------------------------------------------------------

print("INPUTS:")
print()

# From association cortex (vision processing)
association_input = config.add_subnet(
    "association_features", "simple",
    input_size=768,   # From vision (ventral × dorsal)
    output_size=512   # Compressed for reasoning
)
print(f"  Association features: {association_input.output_size} dims")

# From language processing (dialogue, text)
language_input = config.add_subnet(
    "language_encoding", "simple",
    input_size=1024,  # From language encoder (not yet built)
    output_size=512   # Compressed for reasoning
)
print(f"  Language encoding: {language_input.output_size} dims")

# From working memory (persistent state)
memory_read = config.add_subnet(
    "memory_read", "simple",
    input_size=2048,  # Working memory capacity
    output_size=512   # Retrieved relevant content
)
print(f"  Memory read: {memory_read.output_size} dims")

# Combined input to reasoning
combined_input = 512 + 512 + 512  # 1536 dims total
print(f"  TOTAL INPUT to reasoning: {combined_input} dims")
print()

# ----------------------------------------------------------------------------
# EVIDENCE PROCESSING (Product: Testimony × Physical × Timeline)
# ----------------------------------------------------------------------------

print("EVIDENCE PROCESSING (Product structure):")
print()

# Three evidence streams process different modalities
testimony = config.add_subnet(
    "testimony_processor", "simple",
    input_size=512,   # From language (what was said)
    output_size=512
)
print(f"  Testimony processor: {testimony.input_size} → {testimony.output_size}")
print(f"    Params: {testimony.input_size * testimony.output_size:,}")

physical = config.add_subnet(
    "physical_processor", "simple",
    input_size=512,   # From vision (objects, locations)
    output_size=512
)
print(f"  Physical processor: {physical.input_size} → {physical.output_size}")
print(f"    Params: {physical.input_size * physical.output_size:,}")

timeline = config.add_subnet(
    "timeline_processor", "simple",
    input_size=512,   # Temporal sequencing
    output_size=512
)
print(f"  Timeline processor: {timeline.input_size} → {timeline.output_size}")
print(f"    Params: {timeline.input_size * timeline.output_size:,}")

# Product: Combines all evidence
evidence = config.add_subnet(
    "evidence", "product",
    input_size=512,
    output_size=1536,  # 512 + 512 + 512
    components=[testimony, physical, timeline]
)
print(f"  Evidence (product): {evidence.output_size} dims")
print(f"  Evidence processing SUBTOTAL: ~750K params")
print()

# ----------------------------------------------------------------------------
# ABSTRACTION (Exponential: Evidence → Theory)
# ----------------------------------------------------------------------------

print("ABSTRACTION (Recurrent reasoning):")
print()

# Theory space (abstract hypothesis representation)
theory_space = config.add_subnet(
    "theory_space", "simple",
    input_size=256,
    output_size=256
)

# Abstraction subnet: Maps evidence to theory
abstraction = config.add_subnet(
    "abstraction", "exponential",
    input_size=1536,  # Evidence
    output_size=256,  # Theory
    domain=evidence,
    codomain=theory_space,
    is_recurrent=True  # Iterative refinement
)

print(f"  Abstraction subnet: {abstraction.input_size} → {abstraction.output_size}")
print(f"    Layers (estimated): 1536 → 768 → 512 → 256")
print(f"    Params: ~3M (multilayer with recurrence)")
print(f"    Recurrent: YES (composes with itself for multi-step inference)")
print()

# ----------------------------------------------------------------------------
# HYPOTHESIS TRACKING (Coproduct: Alternative Theories)
# ----------------------------------------------------------------------------

print("HYPOTHESIS TRACKING (Coproduct structure):")
print()

# Individual hypotheses (one per suspect)
# For Danganronpa: typically 6-10 suspects
hypotheses_list = []
for i in range(6):  # 6 suspects for example
    hyp = config.add_subnet(
        f"hypothesis_{i}", "simple",
        input_size=256,
        output_size=256
    )
    hypotheses_list.append(hyp)

print(f"  Number of hypotheses: {len(hypotheses_list)}")
print(f"  Each hypothesis: 256 → 256")
print(f"  Params per hypothesis: ~65K")
print(f"  Hypothesis tracking SUBTOTAL: ~400K params")

# Coproduct: Select one active hypothesis
hypotheses = config.add_subnet(
    "hypotheses", "coproduct",
    input_size=256,
    output_size=256,
    components=hypotheses_list
)
print()

# ----------------------------------------------------------------------------
# CONCRETIZATION (Exponential: Theory → Predictions)
# ----------------------------------------------------------------------------

print("CONCRETIZATION (Theory → Predictions):")
print()

# Predicted evidence (what theory predicts we should see)
evidence_predictions = config.add_subnet(
    "evidence_predictions", "simple",
    input_size=1536,
    output_size=1536
)

concretization = config.add_subnet(
    "concretization", "exponential",
    input_size=256,   # Theory
    output_size=1536, # Predicted evidence
    domain=theory_space,
    codomain=evidence_predictions
)

print(f"  Concretization subnet: {concretization.input_size} → {concretization.output_size}")
print(f"    Layers (estimated): 256 → 512 → 768 → 1536")
print(f"    Params: ~3M")
print(f"    Forms adjunction with abstraction (predictive coding for reasoning!)")
print()

# ----------------------------------------------------------------------------
# WORKING MEMORY (State Monad)
# ----------------------------------------------------------------------------

print("WORKING MEMORY (Persistent state):")
print()

working_memory = config.add_subnet(
    "working_memory", "simple",
    input_size=2048,
    output_size=2048,
    is_recurrent=True
)

memory_write = config.add_subnet(
    "memory_write", "simple",
    input_size=256,   # From theory space
    output_size=2048  # Update memory
)

print(f"  Memory capacity: {working_memory.output_size} dims")
print(f"  Memory read: 2048 → 512 (retrieval)")
print(f"  Memory write: 256 → 2048 (storage)")
print(f"  Params: ~2M")
print(f"  Recurrent: YES (state persists across timesteps)")
print()

# ----------------------------------------------------------------------------
# ACTION SELECTION (Exponential: Theory → Action)
# ----------------------------------------------------------------------------

print("ACTION SELECTION:")
print()

# Different action types
present_evidence_action = config.add_subnet(
    "present_evidence", "exponential",
    input_size=256,
    output_size=1536,  # Pointer into evidence space
    domain=theory_space,
    codomain=evidence_predictions
)

advance_dialogue_action = config.add_subnet(
    "advance_dialogue", "simple",
    input_size=256,
    output_size=1  # Binary: continue or not
)

print(f"  Present evidence: {present_evidence_action.input_size} → {present_evidence_action.output_size}")
print(f"  Advance dialogue: {advance_dialogue_action.input_size} → {advance_dialogue_action.output_size}")
print(f"  Action selection SUBTOTAL: ~500K params")
print()

# ============================================================================
# CONNECTION TOPOLOGY
# ============================================================================

print("=" * 70)
print("CONNECTION TOPOLOGY")
print("=" * 70)
print()

# Forward connections
config.connect(language_input, testimony)
config.connect(association_input, physical)
config.connect(evidence, abstraction)
config.connect(abstraction, hypotheses)
config.connect(hypotheses, concretization)
config.connect(hypotheses, present_evidence_action)

# Recurrent connections
# (Not explicitly modeled as connections in validator, but noted as is_recurrent)

print("Forward flow:")
print("  Association → Physical evidence")
print("  Language → Testimony")
print("  Memory → (all components)")
print("  Evidence → Abstraction → Hypotheses → Concretization")
print("  Hypotheses → Action selection")
print()

print("Recurrent/Feedback:")
print("  Abstraction → Abstraction (iterative refinement)")
print("  Predictions → Evidence (error signal)")
print("  Theory → Memory write → Memory → Theory (state persistence)")
print("  Gaze control → Vision (foveal attention)")
print()

# ============================================================================
# SIZE SUMMARY
# ============================================================================

print("=" * 70)
print("SIZE SUMMARY")
print("=" * 70)
print()

sizes = {
    "Evidence processing": "~750K",
    "Abstraction": "~3M",
    "Hypothesis tracking": "~400K",
    "Concretization": "~3M",
    "Working memory": "~2M",
    "Action selection": "~500K"
}

total_params = 9.65  # Million

print("Parameter breakdown:")
for component, params in sizes.items():
    print(f"  {component:.<30} {params:>10} params")

print()
print(f"  {'TOTAL REASONING SYSTEM':.<30} ~{total_params:.1f}M params")
print()

print("Comparison to other systems:")
print(f"  Vision system:    ~10M params (conv layers)")
print(f"  Reasoning system: ~10M params (this architecture)")
print(f"  Motor system:     ~1M params (simple mapping)")
print(f"  {'FULL AGENT':.30} ~21M params")
print()

print("Memory footprint (FP16):")
print(f"  Parameters: {total_params * 2:.1f} MB")
print(f"  Activations: ~20 MB (during inference)")
print(f"  Working memory state: 8 MB (2048 dims × FP32)")
print(f"  {'TOTAL':.30} ~40 MB for reasoning system")
print()

# ============================================================================
# STRUCTURAL ANALYSIS
# ============================================================================

print("=" * 70)
print("STRUCTURAL ANALYSIS")
print("=" * 70)
print()

print("Is it one subnet or multiple?")
print("  → MULTIPLE SUBNETS (12 total) organized categorically")
print()

print("Subnets in reasoning system:")
print("  1. Testimony processor (evidence component)")
print("  2. Physical processor (evidence component)")
print("  3. Timeline processor (evidence component)")
print("  4. Evidence (product of above)")
print("  5. Abstraction (recurrent, exponential)")
print("  6-11. Six hypotheses (coproduct components)")
print("  12. Hypotheses (coproduct)")
print("  13. Concretization (exponential)")
print("  14. Working memory (state monad)")
print("  15. Memory read")
print("  16. Memory write")
print("  17. Present evidence action")
print("  18. Advance dialogue action")
print()

print("How big is it?")
print(f"  → ~10M parameters (similar to vision system)")
print(f"  → ~40 MB memory footprint")
print()

print("Input dimensions?")
print(f"  → 1536 dims total")
print(f"    • 512 from association (vision processing)")
print(f"    • 512 from language (dialogue/text)")
print(f"    • 512 from working memory (retrieved content)")
print()

print("Output dimensions?")
print(f"  → Multiple outputs:")
print(f"    • 1536 dims to actions (present evidence)")
print(f"    • 1 dim to dialogue control (advance/wait)")
print(f"    • 2048 dims to memory write (state update)")
print(f"    • 256 dims to attention (gaze control)")
print()

print("How does it connect to other regions?")
print()
print("  INPUTS from:")
print("    • Association cortex (vision → features)")
print("    • Language encoder (dialogue → semantics)")
print("    • Working memory (state → relevant facts)")
print()
print("  OUTPUTS to:")
print("    • Motor system (actions: keyboard, mouse)")
print("    • Working memory (updates: new evidence, conclusions)")
print("    • Attention system (gaze: where to look)")
print("    • Self (recurrent: iterative refinement)")
print()

print("=" * 70)
print("VALIDATE")
print("=" * 70)
print()

report = config.validate()
report.print_report()

print()
print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print()

print("The abstract reasoning system is:")
print()
print("  • ~10M parameters (similar scale to vision)")
print("  • 18 subnets organized categorically (not arbitrary)")
print("  • Deeply recurrent (abstraction iterates)")
print("  • Stateful (working memory persists)")
print("  • Structured by products, coproducts, exponentials, adjunctions")
print()
print("This is NOT a single monolithic 'reasoning layer'.")
print("It's a SYSTEM of interconnected subnets with categorical structure.")
print()
print("Key architectural principle:")
print("  Category theory PRESCRIBES how these subnets must be organized.")
print("  The structure isn't arbitrary - it's mathematically necessary.")
print()

if report.is_valid():
    print("✓ Architecture is categorically sound and ready to implement.")
else:
    print("✗ Architecture needs fixes.")

print("=" * 70)
