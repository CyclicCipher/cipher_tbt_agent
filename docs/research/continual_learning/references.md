# Continual Learning References

## Core Methods (Applicable to Our Use Case)

### Elastic Weight Consolidation (EWC)
- **Paper**: Kirkpatrick et al. 2017 — "Overcoming catastrophic forgetting in neural networks"
- **arXiv**: https://arxiv.org/abs/1612.00796
- **Key idea**: Diagonal Fisher Information penalty protects important weights
- **Relevance**: Soft constraints allow compositional reuse, unlike hard freezing

### Dark Experience Replay++ (DER++)
- **Paper**: Buzzega et al. 2020 — "Dark Experience for General Continual Learning: a Strong, Simple Baseline"
- **arXiv**: https://arxiv.org/abs/2004.07211
- **Key idea**: Store logits alongside replay samples; MSE on logits preserves circuit behavior
- **Relevance**: Prevents representational drift, not just accuracy drift

### La-MAML (Look-Ahead Meta-Learning for Continual Learning)
- **Paper**: Gupta et al. 2020 — "La-MAML: Look-ahead Meta Learning for Continual Learning"
- **arXiv**: https://arxiv.org/abs/2007.13904
- **Key idea**: Per-parameter learning rates based on gradient alignment between old/new tasks
- **Relevance**: Automatically identifies composition (aligned gradients) vs. conflict

### Synaptic Intelligence (SI)
- **Paper**: Zenke et al. 2017 — "Continual Learning Through Synaptic Intelligence"
- **arXiv**: https://arxiv.org/abs/1703.04200
- **Key idea**: Online importance tracking (path integral of per-parameter contributions to loss)
- **Relevance**: Like EWC but computed online, no post-hoc Fisher computation

## Architecture-Level Approaches

### Google's Nested Learning / HOPE
- **Paper**: Behrouz et al. 2025 — "Nested Learning: Multi-Level Optimization"
- **arXiv**: https://arxiv.org/abs/2512.24695
- **Blog**: https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/
- **Key idea**: Multi-timescale memory (CMS); different components update at different rates
- **Relevance**: Core insight = differential LR by layer; full HOPE needs 1.3B+ scale

### Progressive Neural Networks
- **Paper**: Rusu et al. 2016 — "Progressive Neural Networks"
- **arXiv**: https://arxiv.org/abs/1606.04671
- **Key idea**: New frozen column per task with lateral connections
- **Relevance**: NOT viable — 12x model size, prevents composition within columns

### PackNet
- **Paper**: Mallya & Lazebnik 2018 — "PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning"
- **arXiv**: https://arxiv.org/abs/1711.05769
- **CVPR**: https://openaccess.thecvf.com/content_cvpr_2018/papers/Mallya_PackNet_Adding_Multiple_CVPR_2018_paper.pdf
- **Key idea**: Prune after each task, freeze surviving weights, use pruned weights for next task
- **Relevance**: NOT viable — disjoint subnetworks prevent composition

### SupSup (Supermasks in Superposition)
- **Paper**: Wortsman et al. 2020 — "Supermasks in Superposition"
- **arXiv**: https://arxiv.org/abs/2006.14769
- **Key idea**: Fixed random base weights + learned binary masks per task
- **Relevance**: NOT viable — random base, independent masks, no composition

### Winning SubNetworks (WSN)
- **Paper**: Kang et al. 2022 — "Forget-free Continual Learning with Winning Subnetworks"
- **ICML**: https://proceedings.mlr.press/v162/kang22b/kang22b.pdf
- **Key idea**: Task-adaptive binary masks over trained (not random) weights; weight sharing allowed
- **Relevance**: Partially viable — allows weight reuse but composition is implicit

### SoftNet
- **Paper**: Kang et al. 2023 — "Soft-Masking for Cost-Constrained Group Fairness"
- **arXiv**: https://arxiv.org/abs/2303.14962
- **Key idea**: Continuous (non-binary) masks for smoother optimization
- **Relevance**: Extension of WSN with better optimization properties

## Compositional Learning (Validating Our Approach)

### Skill Composition from Examples
- **Paper**: Zhao et al. 2024 — "Can Models Learn Skill Composition from Examples?"
- **NeurIPS 2024**: https://proceedings.neurips.cc/paper_files/paper/2024/hash/b99fd30559b520cc97447ba905040677-Abstract-Conference.html
- **arXiv**: https://arxiv.org/abs/2409.19808
- **Key finding**: Fine-tuning on k=2-3 skill compositions teaches a META-SKILL for composition generalizing to k=4-5

### Compositional Curricula in ICL
- **Paper**: Lee et al. 2025 — Compositional Curricula for In-Context Learning
- **arXiv**: https://arxiv.org/abs/2506.13253
- **Key finding**: Curriculum-trained models develop internal representations of intermediate values

### Skills-in-Context (SKiC)
- **Paper**: Chen et al. 2024 — "Skills-in-Context Prompting"
- **EMNLP**: https://aclanthology.org/2024.findings-emnlp.812/
- **Key finding**: LLMs compose skills when given explicit definitions + examples

### Two-System Compositional CL
- **Paper**: Shan et al. 2025 — "What and How: Two-System Approach for Compositional CL"
- **arXiv**: https://arxiv.org/abs/2510.20709
- **NeurIPS 2025**: https://neurips.cc/virtual/2025/poster/117578
- **Key idea**: Low-rank RNN with modular components; W = Σₖ αₖ uₖvₖᵀ
- **Relevance**: Promising for Mamba3 adaptation (modular recurrence)

### ICLR 2025 Workshop — Modular Networks Warning
- **Key finding**: Modular networks DON'T compose unless task structure is explicitly provided
- **Implication**: Our explicit interleaved counting structure may be essential

## LoRA-based Continual Learning (Not Directly Applicable)

### C-LoRA
- **arXiv**: https://arxiv.org/abs/2502.17920
- **Note**: Designed for fine-tuning pre-trained LLMs, not training from scratch

### CL-LoRA
- **CVPR 2025**: https://openaccess.thecvf.com/content/CVPR2025/papers/He_CL-LoRA_Continual_Low-Rank_Adaptation_for_Rehearsal-Free_Class-Incremental_Learning_CVPR_2025_paper.pdf
- **Note**: Task-shared + task-specific adapters; requires pre-trained backbone

### KeepLoRA (January 2026)
- **Key idea**: Project gradients onto subspace orthogonal to pre-trained + previous task directions
- **Note**: Requires pre-trained backbone

## Differential Learning Rates

### LeRaC (Learning Rate Curriculum)
- **Paper**: Springer 2024 — "LeRaC: Learning Rate Curriculum"
- **Link**: https://link.springer.com/article/10.1007/s11263-024-02186-5
- **Key idea**: Different learning rates for different layers, converging over training

## Surveys

### Modular Deep Learning
- **arXiv**: https://arxiv.org/abs/2302.11529

### Continual Learning in Foundation Models
- **arXiv**: https://arxiv.org/abs/2506.03320

### Continual Learning Meets Compositionality
- **NeurIPS 2023**: https://neurips.cc/virtual/2023/poster/73708

### Compositional Generalization in NLI
- **TACL 2024**: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00680/123881

## Modular Compositional Networks
- **Google Research**: https://arxiv.org/abs/2107.10963
  Isometric ResNets with interchangeable modules; modules reusable with small param increase
