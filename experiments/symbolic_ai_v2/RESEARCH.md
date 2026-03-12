# symbolic_ai_v2 — Research Survey

Compiled from literature search conducted 2026-03-09.
Focus: unsupervised discovery of relationships, with priority on category-theoretic
approaches and O(n) algorithms.

---

## The central open problem

No existing algorithm discovers functors from data in O(n). This is the gap v2
fills. The three pieces that combine to solve it are already in the literature:

1. **SEQUITUR** — discovers a context-free grammar (= free monoidal category
   presentation) from a sequence in O(n).
2. **FCA** — discovers all adjunctions (Galois connections) in a binary relation
   in O(K³) where K = number of concepts.
3. **Merge = Hopf algebra** — the cross-level functor (segmentation / coproduct)
   falls out of the Hopf algebra structure automatically; no separate discovery
   step needed.

Combining these three gives the first O(n) unsupervised functor discovery
algorithm. That is the primary algorithmic contribution of v2.

---

## 1. Category Theory Foundations

### 1.1 Formal Concept Analysis — adjunction discovery from data

**Wille, R. (1982). "Restructuring Lattice Theory: An Approach Based on
Hierarchies of Concepts."**

The foundational paper. A formal context is a triple (G, M, I): objects G,
attributes M, incidence relation I ⊆ G × M. A formal concept is a closed pair
(A, B) where A ⊆ G, B ⊆ M, A = B'' (objects sharing all attributes in B) and
B = A' (attributes shared by all objects in A). The set of all formal concepts,
ordered by extension inclusion, forms a complete lattice: the concept lattice.

**Key mathematical fact**: Every binary relation R ⊆ X × Y induces a Galois
connection between powersets 2^X and 2^Y. A Galois connection is exactly an
adjunction between poset-categories. Therefore **FCA is adjunction discovery
from data** — it finds all adjunctions entailed by the observed relation.

No free parameters. No thresholds. The concept lattice is uniquely determined
by the data.

Actionable for v2: Apply FCA to the cluster-level transition matrix (K×K per
relation) to discover all closed biclusters. Each closed bicluster is an
adjunction — a pair of inverse operations. This is how the CTKG's adjunction
objects should be populated: not by hand, but by running FCA on the observed
data.

URL: https://en.wikipedia.org/wiki/Formal_concept_analysis
URL: https://www.researchgate.net/publication/256846108_Formal_Contexts_Formal_Concept_Analysis_and_Galois_Connections

---

### 1.2 Merge is a Hopf algebra

**Marcolli, M., Chomsky, N., & Berwick, R.C. (2023/2025). "Mathematical
Structure of Syntactic Merge: An Algebraic Model for Generative Linguistics."**
arXiv:2305.18278. MIT Press (Linguistic Inquiry Monographs), August 2025.

Proves that the operation of syntactic Merge (combining two syntactic objects
into one) is described by a **Hopf algebra** — the same algebraic structure
that appears in the renormalization of quantum field theories (Connes-Kreimer
Hopf algebra of rooted trees) and in the combinatorial Dyson-Schwinger
equations of physics.

The Hopf algebra has two operations:
- **Product** (Merge): combine two chunks into one. m: H ⊗ H → H.
- **Coproduct** (Segment): split a chunk into its constituents. Δ: H → H ⊗ H.
These satisfy compatibility conditions (bialgebra axioms + antipode).

**Critical implication for v2**: The multi-level hierarchy (characters →
morphemes → words → phrases → ...) is not a free architectural choice — it is
the canonical structure of the Merge Hopf algebra. The coproduct Δ is
exactly the cross-level functor: it maps a level-N chunk to a tensor product
of level-(N-1) chunks. This means the cross-level structure is algebraically
entailed by the merge operations; it does not need to be discovered separately.

The AtomVocabulary in v1 was implicitly building a Hopf algebra. The
coassociativity of Δ (((Δ⊗id)∘Δ = (id⊗Δ)∘Δ)) is the associativity of
hierarchical parsing: it does not matter whether you split a 3-chunk into
(2+1) or (1+2) first.

URL: https://arxiv.org/abs/2305.18278
URL: https://mitpress.mit.edu/9780262552523/mathematical-structure-of-syntactic-merge/

---

### 1.3 Enriched category theory of language — Yoneda grounds semantics in syntax

**Bradley, T-D., Terilla, J., & Vlassopoulos, Y. (2021/2022). "An Enriched
Category Theory of Language: From Syntax to Semantics."**
arXiv:2106.07890. La Matematica 1, 551–580 (2022).

Probability distributions on texts form a category enriched over [0,1] (the
unit interval with multiplication). Objects = linguistic expressions. The
hom-object hom(w1, w2) = P(w2 extends w1) — the conditional probability that
w2 is a continuation of w1. This is the syntactic enriched category L.

Via the **Yoneda embedding** ℓ: L → [L^op, [0,1]], pass to the category of
enriched copresheaves. This is the semantic category. The meaning of a
linguistic expression = all of its conditional probability relationships to
all other expressions. Semantics is read off from syntax via Yoneda; no
separate semantic representation is needed.

**Theoretical justification for the entire v1 E0-E6 approach**: The
_atom_bigrams dict P(next | atom, rel) is exactly the hom-object of the
syntactic enriched category. E0 clustering = finding which objects in this
category are isomorphic (have the same hom-objects, up to approximation). R3
composition = computing hom-object composition in the enriched category.

**For v2**: The prediction module does not need a separate semantic
representation layer. P(next | context) is the hom-object, and the Yoneda
lemma guarantees that all semantic content is recoverable from it.

URL: https://arxiv.org/abs/2106.07890
URL: https://link.springer.com/article/10.1007/s44007-022-00021-2

---

### 1.4 Functorial data migration — schemas as categories

**Spivak, D.I. (2010/2012). "Functorial Data Migration."**
arXiv:1009.1166. Information and Computation 217, 31–48.

A database schema = a small category C. A database instance = a set-valued
functor F: C → Set. A schema morphism φ: C → D induces three data migration
functors: Σ_φ (left Kan extension), Π_φ (right Kan extension), Δ_φ (pullback).
These generalize SQL projections, unions, and joins.

**Implication**: Discovering that two schemas are related = discovering a
functor between them. The build_functor() method in v1 does this. The v2
system should represent all discovered schema-level structure as functors
between CTKG categories.

URL: https://arxiv.org/abs/1009.1166

**Spivak, D.I. & Kent, R.E. (2012). "Ologs: A Categorical Framework for
Knowledge Representation."**
PLOS ONE 7(1): e24274.

An olog (ontology log) is a category: objects are types (noun phrases),
morphisms are aspects (functional relations between types). A functor between
ologs = a consistent translation between two knowledge representations. Any
domain of knowledge can be represented as an olog; the functor gives the
systematic translation between domains.

URL: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0024274

**Fong, B. & Spivak, D.I. (2018/2019). "Seven Sketches in Compositionality:
An Invitation to Applied Category Theory."**
arXiv:1803.05316. Cambridge University Press 2019.

Seven applications: orders/Galois connections, databases/functors,
resources/monoidal categories, codesign/profunctors, circuits/operads,
dynamical systems/sheaves, probability/Markov categories. Chapter 2 (Galois
connections) and Chapter 3 (functorial databases) are directly relevant.

URL: https://arxiv.org/abs/1803.05316

---

### 1.5 Backprop as functor — learning is functorial

**Fong, B., Spivak, D.I., & Tuyéras, R. (2019). "Backprop as Functor: A
Compositional Perspective on Supervised Learning."**
arXiv:1711.10455. LICS 2019.

Gradient descent with fixed step size defines a monoidal functor
Para(Euc) → Learn, from parametrized Euclidean maps to a category of learners.
Learners are symmetric lenses (forward pass = map, backward pass = parameter
update). Learn ≅ Para(SLens).

**Implication for v2**: The belief update in active inference is the backward
pass of this functor. Forward pass = predict next token from current model.
Backward pass = update model parameters given prediction error. The functorial
structure guarantees that composing learners gives a learner for the composed
task.

URL: https://arxiv.org/abs/1711.10455

**Gavranović, B., Lessard, P., Dudzik, A., et al. (2024). "Position:
Categorical Deep Learning is an Algebraic Theory of All Architectures."**
arXiv:2402.15332. ICML 2024.

Universal algebra of monads valued in a 2-category of parametric maps subsumes
all known architectures (CNNs, RNNs, GNNs, Transformers). Equivariance =
monad algebra homomorphism. Discovering the symmetry structure of data =
discovering the monad.

URL: https://arxiv.org/abs/2402.15332

---

### 1.6 Compositional Markov processes — functorial belief propagation

**Baez, J.C., Fong, B., & Pollard, B.S. (2016). "A Compositional Framework
for Markov Processes."**
arXiv:1508.06448. Journal of Mathematical Physics 57, 033301.

Open Markov processes = morphisms of a dagger compact category. Composition =
wiring processes together at their interfaces. A black-boxing functor maps
detailed-balanced Markov processes to Lagrangian relations between symplectic
vector spaces.

**Implication**: The ContextBeliefState in v1 is a Bayes filter; this paper
shows such filters are functorial. The multi-level belief cascade = composition
of open Markov processes. In v2, all belief propagation should be expressed as
morphism composition in the appropriate Markov category (already built in
experiments/ctkg/).

URL: https://arxiv.org/abs/1508.06448

---

### 1.7 DisCoCat — functors from grammar to semantics

**Coecke, B., Sadrzadeh, M., & Clark, S. (2010). "Mathematical Foundations for
a Compositional Distributional Model of Meaning."**
Linguistic Analysis 36, 345–384.

Grammar type reductions live in a compact closed category P (pregroup grammar).
Word meanings are vectors in FVect. DisCoCat is a functor F: P → FVect mapping
grammatical derivations to linear maps on word vectors. Sentence meaning =
F(derivation) applied to the tensor product of word meanings. Visualized via
string diagrams.

**Relation to v2**: DisCoCat solves half the problem — it defines a functor
from grammar to semantics — but assumes the grammar is given (pregroup/Lambek).
V2 must discover the grammar and the functor simultaneously and unsupervised.
DisCoCat is what the system converges to; the algorithm that gets there is
what v2 contributes.

URL: https://en.wikipedia.org/wiki/DisCoCat
URL: https://ncatlab.org/nlab/show/categorical+compositional+distributional+semantics

**de Felice, G. (2022). "Categorical Tools for Natural Language Processing."
Oxford PhD thesis.**

String diagrams provide a unified model of syntactic structures across formal
grammars. Functors compute semantics from syntax. Morphisms of grammars =
graph homomorphisms preserving structure.

URL: https://ora.ox.ac.uk/objects/uuid:07b94121-5634-439c-9ece-051f3ffd6c81

---

### 1.8 Yoneda at scale — foundation models as representable functors

**Yuan, Y. et al. (2023). "On the Power of Foundation Models."**
ICML 2023, PMLR 202.

Applies the Yoneda lemma to prompt tuning: a task T is a functor in C∧; the
Yoneda lemma gives Hom(hC(X), T) ≅ T(X). A foundation model can solve task T
iff T is representable. The optimal prompt = the representing object.

**Implication**: A sufficiently capable symbolic AI (one that has learned enough
morphisms) can solve any representable task via pattern matching against its
learned morphisms. The benchmark for v2 is whether the tasks in GOALS.md are
representable in the discovered CTKG.

URL: https://proceedings.mlr.press/v202/yuan23b/yuan23b.pdf

---

## 2. O(n) Grammar Induction Algorithms

### 2.1 SEQUITUR — the baseline O(n) algorithm

**Nevill-Manning, C.G. & Witten, I.H. (1997). "Identifying Hierarchical
Structure in Sequences: A Linear-Time Algorithm."**
arXiv:cs/9709102. Journal of Artificial Intelligence Research 7, 67–82.

Two invariants maintained throughout processing:
1. **Digram uniqueness**: no pair of adjacent symbols appears more than once
   in the grammar.
2. **Rule utility**: every grammar rule is used at least twice.

Algorithm: maintain a hash table of digrams → (rule, position). When a new
digram is seen for the second time, create a new rule replacing both
occurrences. Enforce rule utility by folding single-use rules back into their
call sites. Running time and space: **O(n)** in input length.

Applied to: 40MB digital library text, DNA sequences, Bible (English/French/
German), genealogical databases. Domain-agnostic.

**Relation to v2**: SEQUITUR is the O(n) backbone algorithm. The v1 PCH used
a soft surprise threshold (free parameter); SEQUITUR uses digram uniqueness
(MDL-entailed, zero free parameters). The "second occurrence" criterion is
identical to MDL: a pair reduces description length iff it occurs ≥ 2 times.
SEQUITUR extends to arbitrary graphs by replacing "adjacent symbols" with
"adjacent nodes in the input graph" — the digram hash table becomes an edge-pair
hash table, and the topology is an input parameter.

URL: https://arxiv.org/abs/cs/9709102
URL: http://www.sequitur.info/Nevill-Manning.pdf

---

### 2.2 ADIOS — statistical motif extraction from digraphs

**Solan, Z., Horn, D., Ruppin, E., & Edelman, S. (2005). "Unsupervised
Learning of Natural Languages."**
PNAS 102(33), 11629–11634.

Data structure: directed pseudograph where nodes = tokens, paths = sequences.
A "pattern" (Motif) is a maximal equivalence class of paths that share a
statistically significant prefix bundle. The MEX criterion (Mutual Exclusivity
and eXhaustiveness) defines significance without free parameters: a path is a
pattern iff replacing its context with a random context changes the path
probability significantly.

Evaluated on: English, Chinese, artificial CFGs with thousands of rules,
protein sequences — fully domain-agnostic.

**Relation to v2**: ADIOS operates on the same digraph structure as v2's input
topology abstraction. The MEX criterion is a statistical version of SEQUITUR's
digram uniqueness — both are MDL-motivated, both zero free parameters. ADIOS is
not linear-time but is practically efficient. It explicitly handles
non-deterministic and probabilistic grammars, which SEQUITUR does not.

URL: https://www.pnas.org/doi/10.1073/pnas.0409746102
URL: https://shimon-edelman.github.io/SolanHornRuppinEdelman-PNAS05.pdf

---

## 3. Unsupervised Relation Discovery from Text

### 3.1 Foundational NLP approaches

**Hasegawa, T., Sekine, S., & Grishman, R. (2004). "Discovering Relations Among
Named Entities from Large Corpora."**
ACL 2004.

Method: extract named entity pairs, represent by context bag-of-words, apply
hierarchical agglomerative clustering (HAC), label clusters by most frequent
context words. First major unsupervised relation discovery paper.

**Davidov, D. & Rappoport, A. (2008). "Unsupervised Discovery of Generic
Relationships Using Pattern Clusters and its Evaluation by Automatically
Generated SAT Analogy Questions."**
ACL 2008.

Extract lexico-syntactic patterns between word pairs, cluster patterns, evaluate
via automatically generated SAT analogy questions. The patterns between entity
pairs are proxies for relation types; clustering patterns = clustering relation
types. Structurally identical to v1's R1 (role signature clustering).

URL: https://aclanthology.org/P08-1079/

**Yao, L. et al. (2012). "Unsupervised Relation Discovery with Sense
Disambiguation."**
ACL 2012.

URL: https://ciir-publications.cs.umass.edu/getpdf.php?id=1032

---

### 3.2 Open Information Extraction

**Survey: "A Survey on Open Information Extraction from Rule-based Model to
Large Language Model."**
arXiv:2208.08690. EMNLP 2024 Findings.

Covers TextRunner, OLLIE, ClausIE, OpenIE4, neural OpenIE, LLM-based IE. Four
generations of OIE systems. Key unresolved problem: canonicalization — mapping
surface relation strings to a unified schema. Universal Schema (below) is the
principled solution.

URL: https://arxiv.org/abs/2208.08690

---

### 3.3 Universal Schema — matrix factorization over all relation types

**Riedel, S., Yao, L., McCallum, A., & Marlin, B.M. (2013). "Relation
Extraction with Matrix Factorization and Universal Schemas."**
NAACL-HLT 2013, pp. 74–84. arXiv:1301.4293.

Takes the union of all relation schemas: surface patterns from OpenIE plus
structured KB relations (Freebase). Jointly factorizes the entity-pair ×
relation matrix, learning latent vectors for entity pairs and for relation
types. Learns asymmetric implicature P(r2 | r1, entity-pair) — which surface
patterns imply which structured relations.

**Relation to v2's R3**: This is the probabilistic / approximate version of
what R3 does exactly (composition of transition matrices). Universal Schema
uses matrix factorization (approximate, finds low-rank structure); R3 discovers
exact algebraic compositions. V2 should use MDL-guided exact composition where
possible and factorization as a fallback for high-V levels.

URL: https://arxiv.org/abs/1301.4293

---

## 4. Knowledge Graph Embedding — relation discovery as tensor decomposition

### 4.1 RESCAL

**Nickel, M., Tresp, V., & Kriegel, H-P. (2011). "A Three-Way Model for
Collective Learning on Multi-Relational Data."**
ICML 2011.

Three-way tensor X[i,j,k] = 1 iff entity_i has relation_k with entity_j.
RESCAL factorizes X_k ≈ A · R_k · A^T. Entity embedding matrix A is shared
across relations; per-relation matrix R_k captures the relational structure.

### 4.2 TransE

**Bordes, A., et al. (2013). "Translating Embeddings for Modeling
Multi-Relational Data."**
NeurIPS 2013.

Relations as translations in embedding space: h + r ≈ t. Simple, linear.
Fails for symmetric, 1-N, N-1, N-N relations.

### 4.3 TuckER — most expressive bilinear model

**Balažević, I., Allen, C., & Hospedales, T.M. (2019). "TuckER: Tensor
Factorization for Knowledge Graph Completion."**
EMNLP 2019. arXiv:1901.09590.

Tucker decomposition: score(h, r, t) = W ×₁ e_h ×₂ w_r ×₃ e_t where W is a
core tensor. RESCAL, DistMult, ComplEx, and TransE are all special cases of
TuckER with different sparsity patterns on W.

**Implication for v2**: The K×K transition matrices per relation in v1 (type-
level transition model) are a coarsened Tucker decomposition where entity
identities are replaced by type membership. TuckER proves this is the most
expressive bilinear model; there is no strictly more general bilinear approach.
The symbolic v2 system should beat TuckER on structured data because it finds
exact algebraic structure rather than approximate low-rank approximations.

URL: https://arxiv.org/abs/1901.09590

Curated list of KGE papers: https://github.com/MIRALab-USTC/KGEPapers

---

## 5. MDL and Information-Theoretic Structure Discovery

### 5.1 Minimum description length

**Rissanen, J. (1978). "Modeling by Shortest Data Description."**
Automatica 14, 465–471.

The MDL principle: the best model of data is the one that produces the shortest
description of the data + the model. Equivalent to Bayesian model selection
with a universal prior. The theoretical foundation for all zero-parameter
criteria in v2.

URL: https://mitpress.mit.edu/books/minimum-description-length-principle

**Survey: "The minimum description length principle for pattern mining."**
Data Mining and Knowledge Discovery (Springer, 2022).

Reviews MDL methods for transactional data (Krimp), sequential data, relational
data, and graph data. Directly applicable to the pattern discovery layer of v2.

URL: https://link.springer.com/article/10.1007/s10618-022-00846-7

### 5.2 GraphMDL — MDL for graph patterns

**Bariatti, F., Cellier, P., & Ferré, S. (2020). "GraphMDL: Graph Pattern
Selection Based on Minimum Description Length."**
IDA 2020, Springer.

MDL applied to labeled graphs. Selects a compact set of graph patterns that
minimally describe the data. Uses "ports" to encode how pattern instances
connect to the rest of the graph.

**Relation to v2**: After the morphism discovery layer finds all recurring
patterns, GraphMDL selects the most informative subset. This gives principled
compression of the CTKG: only patterns that reduce description length are kept.

URL: https://link.springer.com/chapter/10.1007/978-3-030-44584-3_5

### 5.3 SUBDUE — MDL graph grammar induction

**Cook, D.J. & Holder, L.B. (1994). "Substructure Discovery Using Minimum
Description Length and Background Knowledge."**
Journal of Artificial Intelligence Research 1, 231–255.

Discovers subgraphs that maximally compress the input graph (MDL criterion).
With the graph grammar option, iterates: find best subgraph, replace all
instances with a single node, repeat. Produces a hierarchical graph grammar.

**Relation to v2**: SUBDUE with iterative grammar mode is the graph-structured
analogue of SEQUITUR — it applies Merge to a graph, guided by MDL, producing
the same hierarchical structure but for arbitrary graph topologies.

### 5.4 Solomonoff induction — the theoretical ceiling

**Solomonoff, R. (1964). "A Formal Theory of Inductive Inference."**
Information and Control 7(1), 1–22.

The universal prior over programs. Completeness: if any describable regularity
exists, Solomonoff induction finds it. Incomputable in general.

**Relation to v2**: MDL (via SEQUITUR + GraphMDL) is a computable approximation
to Solomonoff induction. The v2 system is doing tractable Solomonoff induction:
finding the shortest grammar that generates the observed data.

URL: https://en.wikipedia.org/wiki/Solomonoff%27s_theory_of_inductive_inference

---

## 6. Graph-Based Substructure Mining

### 6.1 gSpan

**Yan, X. & Han, J. (2002). "gSpan: Graph-Based Substructure Pattern Mining."**
ICDM 2002.

DFS lexicographic canonical form for graphs. Mines frequent subgraphs without
candidate generation. Substantially faster than prior algorithms (e.g. AGM,
FSG).

URL: https://sites.cs.ucsb.edu/~xyan/papers/gSpan-short.pdf

### 6.2 Small-world network structure

**Watts, D.J. & Strogatz, S.H. (1998). "Collective Dynamics of 'Small-World'
Networks."**
Nature 393, 440–442.

Two structural properties: high clustering coefficient + low average path
length. The interpolation between regular lattice (p=0) and random graph (p=1)
via rewiring probability p.

**Relation to v2**: The topology of the discovered morphism graph characterizes
what kind of structure the system has found. Regular lattice = rigid grammar.
Random graph = no structure. Small-world = hierarchical, modular structure
(what we expect from natural language and mathematics). The geometry detector
in v1's R0 measures this; v2 should use it to characterize the CTKG topology.

URL: https://www.nature.com/articles/30918

---

## 7. Inductive Logic Programming — supervised relation discovery

**Cropper, A., et al. (2021). "Inductive Logic Programming at 30."**
Machine Learning (Springer). arXiv:2002.11002.

ILP discovers first-order logic rules (Horn clauses) from positive/negative
examples plus background knowledge. Modern variants: ILASP (ASP-based),
∂ILP (differentiable), NeuralLP, Popper (meta-level search).

**Limitation**: ILP requires labelled examples and background knowledge.
It is supervised or semi-supervised. The v2 system is fully unsupervised.

**Use in v2**: ILP can serve as a verifier or post-hoc interpreter. Once
the morphism graph is built, ILP can extract interpretable logical rules from
it given a small set of examples. This is the interface between unsupervised
discovery (v2 core) and symbolic reasoning (CTKG interpreter).

URL: https://link.springer.com/article/10.1007/s10994-021-06089-1

---

## 8. Word Embeddings as Implicit Relation Discovery

**Mikolov, T. et al. (2013). "Distributed Representations of Words and
Phrases and their Compositionality." (Word2Vec)**
NeurIPS 2013.

Analogy arithmetic: king - man + woman ≈ queen. Linear algebraic structure
in embedding space is discovered unsupervised from raw text. The skip-gram
objective with negative sampling implicitly factorizes the shifted PMI matrix
(Levy & Goldberg 2014).

**Relation to v2**: Word2Vec finds approximate relation composition via vector
arithmetic. V2 finds exact relation composition via morphism graph traversal.
The symbolic approach is interpretable, composable, and generalizes to
non-linguistic domains; the vector approach is opaque and domain-specific.

**Pennington, J., Socher, R., & Manning, C.D. (2014). "GloVe: Global Vectors
for Word Representation."**
EMNLP 2014.

Global co-occurrence statistics. Objective: w_i · w_j + b_i + b_j ≈ log(X_ij).
Linear substructure in embedding space. Same approximate relation composition.

URL: https://nlp.stanford.edu/projects/glove/

---

## 9. Curated Reading Lists

**bgavran/Category_Theory_Machine_Learning (GitHub)**
Curated list of papers studying ML through the lens of category theory.
URL: https://github.com/bgavran/Category_Theory_Machine_Learning

**jbrkr/Category_Theory_Natural_Language_Processing_NLP (GitHub)**
Papers at the intersection of category theory and NLP.
URL: https://github.com/jbrkr/Category_Theory_Natural_Language_Processing_NLP

---

## 10. Summary: What to use, what to discard

| Result | Use in v2 | Reason |
|--------|-----------|--------|
| FCA / Galois connections (Wille 1982) | **Yes — adjunction discovery** | Zero-parameter, finds all adjunctions from data; O(K³) on cluster-level matrix |
| Merge = Hopf algebra (Marcolli et al. 2023) | **Yes — algebraic structure** | Cross-level functor = Hopf coproduct; falls out for free from merge operations |
| SEQUITUR (Nevill-Manning & Witten 1997) | **Yes — O(n) core algorithm** | Zero-parameter, MDL-entailed, extends to arbitrary graph topologies |
| Enriched CT of language (Bradley et al. 2021) | **Yes — theoretical foundation** | Proves P(next\|context) is the right hom-object; Yoneda gives semantics from syntax |
| DisCoCat (Coecke et al. 2010) | **Reference target** | What v2 converges to; not an algorithm for unsupervised discovery |
| Functorial data migration (Spivak 2012) | **Yes — schema alignment** | Principled framework for cross-domain functor discovery |
| Backprop as functor (Fong et al. 2019) | **Yes — active inference** | Belief updates are the backward pass of a functorial learner |
| ADIOS (Solan et al. 2005) | **Yes — probabilistic motifs** | Statistical version of SEQUITUR; handles non-determinism; domain-agnostic |
| Universal Schema (Riedel et al. 2013) | **Partial — high-V fallback** | Use matrix factorization when exact composition is too expensive |
| TuckER (Balažević et al. 2019) | **Reference only** | Shows K×K transition matrices are a coarsened Tucker decomposition; v2 beats it on structured data |
| GraphMDL (Bariatti et al. 2020) | **Yes — graph compression** | MDL-based selection of most informative graph patterns |
| SUBDUE (Cook & Holder 1994) | **Yes — graph grammar** | Graph-structured analogue of SEQUITUR; iterative Merge on arbitrary graphs |
| ILP (Cropper et al. 2021) | **Post-hoc verifier only** | Supervised; use to extract interpretable rules after unsupervised discovery |
| Word2Vec / GloVe | **Baseline only** | Approximate, opaque; v2 should exceed them on structured benchmarks |
| Solomonoff induction | **Theoretical ceiling** | V2 is tractable approximation via SEQUITUR + MDL |

---

## 8. Gary Marcus — Algebraic Rules and Variable Binding

*Research compiled 2026-03-11 (web search + primary sources).*

### 8.1 Core claim

Marcus (2001, *The Algebraic Mind*) argues that human cognition is driven by **algebraic rules** — operations over abstract variables — rather than by statistical association. The minimum mechanism requires:

1. **Variables**: abstract placeholders (slots), not specific items or weighted features.
2. **Variable binding**: instantiating a variable with a specific token via pointer indirection, not via distributed weighted sum.
3. **Operations over bound variables**: rules defined over the slots, not over the slot contents.

The critical diagnostic: **can the system produce correct output for inputs with zero statistical overlap with training data?** Pure statistical systems cannot; systems with true variables can.

### 8.2 Empirical foundation — infant A-B-A study

Marcus, Vijayan, Bandi Rao & Vishton (1999, *Science*): 7-month-old infants habituated to ABA-structured nonsense syllable sequences, then tested on completely novel syllables arranged in ABA vs. ABB structure. Infants discriminated — demonstrating that they had extracted the abstract structural rule (`slot[0] == slot[2]`) and applied it to items with zero phonological overlap with training.

**Computational claim**:
- **Negative**: The result cannot be explained by transitional probabilities or simple recurrent network interpolation.
- **Positive**: Infants extracted the relational rule as an algebraic predicate over variables, not as a pattern over specific syllables.

*Note*: A 2022 multi-lab replication (Geambasu et al., *Developmental Science*, N=96) failed to replicate. The computational argument stands independently of the empirical result.

### 8.3 Two-level structure: innate machinery vs. learned rules

Marcus's position is not hard nativism:
- **The specific rules** (ABA, English past tense suffix) are **learned** from data.
- **The machinery** (variable binding capacity, structural abstraction substrate) is **innate** — present before learning begins because it is what makes rule-learning possible.

Parallel to Chomsky's UG: specific grammars are learned; the capacity for recursive phrase structure is innate. Marcus makes the same argument one level deeper — the algebraic abstraction capacity is the innate part.

### 8.4 Neural implementation: PFC/BG pointer indirection

Marcus's preferred biological implementation (Marcus, Marblestone & Dean 2014, "The Atoms of Neural Computation"):

- Activity patterns in PFC area A encode **pointers/addresses** that instruct the basal ganglia to gate operations on PFC area B, **regardless of B's contents**.
- A variable IS an address, not a value.
- This is pointer indirection — the standard implementation of a universally quantified variable in von Neumann architectures.

Alternative implementations Marcus acknowledges as valid:
- **Smolensky tensor products** (1990): outer product of role vector × filler vector encodes variable-value binding in a distributed, mathematically structured tensor.
- **Binding by synchrony** (Shastri & Ajjanagadde 1993): variables and values synchronized in neural firing patterns; mathematically equivalent to tensor products over temporal role vectors.

### 8.5 Relationship to classical symbolic AI

Marcus's rules are **functionally equivalent to condition-action production rules** (ACT-R, Soar): if variable X is bound to an item satisfying type condition C, produce output O over X. He does not claim the mind uses full first-order logic with proof search — only that it implements the representational substrate (variables, binding, constituent structure) that makes algebraic rules possible.

**Key difference from lookup tables**: A lookup table `σ(arg_id) = result_id` only covers seen pairs. An algebraic rule `M = N + "ed"` covers *any* N, including novel inputs. Both are implementable in a connectionist network; only the latter satisfies Marcus's criterion.

### 8.6 Past tense debate (Rumelhart-McClelland 1986 vs. Pinker-Prince 1988 vs. Marcus 1995/2001)

The English past tense is the central test case:
- **Regular verbs** (`walk → walked`): rule-governed — suffix concatenation over variable X. Generalizes to novel verbs (`rick → ricked`) with zero degradation.
- **Irregular verbs** (`go → went`): lexically stored — each pair is a memory entry. Does not generalize to novel forms.

Marcus's prediction: statistical models fail on **rare regular verbs** (which the rule handles perfectly) and show spurious sensitivity to **phonological similarity** (which the rule should be insensitive to). Both failures are documented.

### 8.7 Implications for variable_binding.py / Phase 19

The hypothesis enumeration approach (`_fit_suffix`, `_fit_prefix`, `_fit_ordinal`, ...) is **wrong in design**, not just incomplete. It puts the intelligence in the engineer (enumerating hypothesis classes) rather than in a general mechanism.

**Marcus-aligned design** for our architecture:

| Layer | Marcus concept | v2 implementation |
|-------|---------------|------------------|
| Variables | Typed slots | Edge types in MorphismGraph |
| Variable binding | Pointer indirection | `mg.rules_inv[(ctx_id, etype, arg_id)] → comp_id` |
| Operations over variables | Rules defined over slots, not contents | AlgebraicRule with `fn` operating on slot contents |
| Type constraint | Variables are typed; rules respect types | Topology edge types constrain which hypothesis space applies |
| Lexical memory | Irregular verbs, arbitrary mappings | MorphismGraph edge counts (statistical associations) |
| Rule learning | Discover rules from data | Symbolic regression within type-appropriate hypothesis space |

**The correct Phase 19 design**: pass topology type information into `fit_rule`; only attempt arithmetic hypotheses for atoms appearing via `num`-typed edges. For atoms appearing via other edge types, return `None` — the MorphismGraph statistical associations are the complete and correct representation. No string suffix hypotheses needed: if no algebraic rule exists, the system correctly falls back to associative memory (MorphismGraph edges).

**The derivatives problem** is not solvable by extending the hypothesis space. The arguments to `d` are composition nodes (structured objects), not atoms. Solving derivatives requires `f∘g` compositional rules — rules over structured variables — which is a separate, larger extension.

### 8.8 Relevant implementations satisfying Marcus's requirements

| System | Mechanism | Status |
|--------|-----------|--------|
| Smolensky TPR (1990) | Outer product variable-value binding | Theoretical; requires designed role structure |
| Holographic Reduced Representations (Plate 1995) | Circular convolution approximation to TPR | Lossy but practical |
| Neural Theorem Provers / DeepProbLog | Differentiable logic programming | Gradient-learnable rule discovery |
| NS-CL (Mao et al. 2019, MIT) | Neural perception + symbolic programs | Outperforms pure DL on CLEVR |
| Lake, Ullman, Tenenbaum & Gershman (2017) | Probabilistic program induction, causal models | Most aligned with Marcus; Marcus/Davis: "a good start" |
| PFC/BG indirection (Kriete et al. 2013) | Biologically grounded pointer indirection | Demonstrates neural variable binding |
| MorphismGraph (v2, this work) | Edge-type variables + Graph-SEQUITUR composition | Satisfies Marcus at composition level; `AlgebraicRule` satisfies at rule level |

### 8.9 Key sources

- Marcus, G.F. (2001). *The Algebraic Mind*. MIT Press.
- Marcus et al. (1999). Rule learning by seven-month-old infants. *Science* 283(5398), 77–80.
- Marcus, Marblestone & Dean (2014). The atoms of neural computation. *Science* 346(6209), 551–552. arXiv:1410.8826.
- Marcus, G.F. (2018). Deep learning: A critical appraisal. arXiv:1801.00631.
- Marcus, G.F. (2020). The next decade in AI. arXiv:2002.06177.
- Geambasu et al. (2022/2023). Robustness of the rule-learning effect: replication failure. *Developmental Science*.
- Lake, Ullman, Tenenbaum & Gershman (2017). Building machines that learn and think like people. *Behavioral and Brain Sciences* 40, e253. arXiv:1604.00289.
- Smolensky, P. (1990). Tensor product variable binding and the representation of symbolic structures in connectionist systems. *Artificial Intelligence* 46(1–2), 159–216.
- Belle & Marcus. The future is neuro-symbolic. AAAI.

---

## 9. Phase 24–26 Implementation Notes (2026-03-11)

### 9.1 Phase 24 — Domain-Agnostic Adjunction Detection

`reasoning/adjunction_detect.py` implements `detect_adjunctions(mg, topo)`.  Given
a trained MorphismGraph and Topology, it scans all pairs of edge types (r1, r2) and
computes unit/counit coverage: what fraction of "r1-then-r2" roundtrips return to
the origin (unit) and vice versa (counit).  Returns `AdjunctionCandidate` objects
sorted by `score = (unit_coverage + counit_coverage) / 2`.

Key design decisions:
- Works entirely on `mg._out` (no knowledge of arithmetic, grammar, or any other domain).
- `is_exact` if both coverages ≥ 0.95 and score > 0.9.
- Handles single-edge-type topologies correctly (returns empty list).
- 8 tests, all pass.  Runtime < 0.5 s on 10K edges.

### 9.2 Phase 25 — Template-Based Rule Matcher

`reasoning/templates.py` implements `build_templates(mg, topo)` and
`predict_via_template(mg, atom_buf, templates)`.

Algorithm:
1. `_collect_observations`: expand every node in `mg._out` to base atoms via
   `mg.generate(src, target_level=0)`.  Record `(ctx_atom_ids, etype, tgt_atom_id)`.
2. Group by `(etype, ctx_len)`.  Skip groups with < 2 distinct contexts.
3. Anti-unify context sequences with `lgg_all` (Phase 22).  Keep only templates
   with ≥ 1 Variable (otherwise nothing to generalise over).
4. Build lookup: `{var_value_tuple: most_common_tgt_id}`.
5. Try `fold_detect` (Phase 23) on variable portions — gives generalisation to
   unseen inputs if a base case exists.
6. Annotate with adjunction info from `detect_adjunctions` (Phase 24).
7. Sort by `(len(lhs), coverage)` descending — longer (more specific) templates
   first, then by coverage within same length.

**Key insight**: Templates are sorted by specificity (`len(lhs)`) first, then
coverage.  Without this, a 1-atom template with high coverage defeats a 4-atom
template that is actually the right answer.  E.g., `['?0'] etype=1` (coverage=732)
must NOT fire before `['add', '?0', '?1', 'eq'] etype=1` (coverage=260).

10 tests, all pass.  Performance < 2 s on math corpus.

### 9.3 Phase 26 — Full Reasoning Layer Replacement

Wired templates into `predict.py` as step 0 of both back-off chains:

1. `_predict_via_templates(mg, atom_buf)` — reads `mg._templates` (set by
   `build_templates_and_store`), calls `predict_via_template`, wraps as `{id: 1.0}`.
2. Inserted at step 0 in `perplexity_multilevel` back-off chain.
3. Inserted at step 0 in `generate_until_eos` back-off chain (before frame match).
4. Added `build_templates_and_store(mg, topo)` convenience function to `templates.py`.

**The original failure fixed**: `rule_store.py` (Phase 17a) yielded 0 endofunctors
for language-only models because it required arithmetic-specific patterns.  After
Phase 26, `build_templates_and_store` on a morphology corpus (dog→dogs, cat→cats,
etc.) yields > 0 templates.  The domain-agnostic anti-unification correctly
discovers `[?0] → ?0+'s'` patterns without any domain knowledge.

6 integration tests, all pass.
