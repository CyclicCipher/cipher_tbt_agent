# Gradient-free feature learning and non-learned denoisers — what the literature says about a block that generalises by shared structure, without a gradient

Written 2026-09-21 for question (2) of the standing rule of 2026-09-22 (no gradient patch for the two places that seem
to need one; research the mathematics first). The question, on E35's numbers (`RESULTS.md` E35, `runs/e35/e35_strict.json`):
a written denoising block is a table from a radius-1 window (9 cells, 16 colours) to the cleaner centre cell, learned
by counting; with every window kept and a Hamming-NEAREST stored window for an unseen one (`e35.py: NearestRule
.predict_nearest`), the chain repairs held-out frames at t = 0.25 / 0.5 / 0.75 to 0.946 / 0.880 / 0.800 (one shot
0.940 / 0.854 / 0.732); under the description-length price every block collapses to the identity. How can a block
generalise by FEATURES (structure shared across windows) rather than by LOOKUP, with no gradient — and what do the
gradient-free feature literature and the non-learned denoisers say?

**Sources.** Abstracts of arXiv papers are quoted verbatim (arXiv releases abstracts as CC0 metadata); for journal and
conference papers not on arXiv (BM3D, K-SVD, NLM, Coates & Ng, Rahimi & Recht, Levin & Nadler) the mechanism and the
numbers are given in our words with short attributed fragments; every number below was read from the paper's PDF on
2026-09-21 unless marked "from memory". No PDFs are copied beside this note (licenses not checked).

**Feel-experiment beside this note:** `research/r_window_neighbours.py` → `research/r_window_neighbours.json` (117 s,
CPU), described in §3. It reproduces E35's one-shot numbers and measures what the lookup actually does on the held-out
frames.

---

## 1. Gradient-free feature learning — what each method fits, how, and what it cannot do

### 1.1 PCANet — Chan, Jia, Gao, Lu, Zeng, Ma, "PCANet: A Simple Deep Learning Baseline for Image Classification?", arXiv:1404.3606 (Apr 2014; IEEE TIP 2015)

Abstract (verbatim): "In this work, we propose a very simple deep learning network for image classification which
comprises only the very basic data processing components: cascaded principal component analysis (PCA), binary
hashing, and block-wise histograms. In the proposed architecture, PCA is employed to learn multistage filter banks. It
is followed by simple binary hashing and block histograms for indexing and pooling. This architecture is thus named as
a PCA network (PCANet) and can be designed and learned extremely easily and efficiently. For comparison and better
understanding, we also introduce and study two simple variations to the PCANet, namely the RandNet and LDANet. They
share the same topology of PCANet but their cascaded filters are either selected randomly or learned from LDA. We have
tested these basic networks extensively on many benchmark visual datasets for different tasks, such as LFW for face
verification, MultiPIE, Extended Yale B, AR, FERET datasets for face recognition, as well as MNIST for hand-written
digits recognition. Surprisingly, for all tasks, such a seemingly naive PCANet model is on par with the state of the
art features, either prefixed, highly hand-crafted or carefully learned (by DNNs). Even more surprisingly, it sets new
records for many classification tasks in Extended Yale B, AR, FERET datasets, and MNIST variations. Additional
experiments on other public datasets also demonstrate the potential of the PCANet serving as a simple but highly
competitive baseline for texture classification and object recognition."

**What is fitted, and how (paper §2.1).** Three stages, all closed form:
1. *PCA filters, stage 1.* Every k1×k2 patch of every image, patch mean removed, as a column of X ∈ R^{k1k2 × Nmn}.
   The L1 filters are the L1 leading eigenvectors of X Xᵀ (their eq. 2–3: min_V ‖X − V Vᵀ X‖²_F s.t. VᵀV = I). Stage 2
   repeats this on the L1 filtered images (eq. 5–6), giving L1·L2 output maps. There is NO non-linearity between the
   stages; the paper tried an absolute-value rectifier after stage 1 and saw no gain.
2. *Binary hashing.* The L2 stage-2 outputs at a pixel are thresholded at 0 (Heaviside) and read as an L2-bit integer,
   T = Σ_ℓ 2^{ℓ−1} H(I ∗ W²_ℓ) (eq. 8) — an integer in [0, 2^{L2} − 1] which "we here treat ... as a distinct 'word'".
3. *Block histograms.* Each integer image is split into B blocks (overlapping for digits/textures/objects,
   non-overlapping for faces) and the 2^{L2}-bin histogram of each block is concatenated (eq. 9): the feature is a
   vector of COUNTS of discrete codes. A linear SVM reads it.
   Settings used everywhere: L1 = L2 = 8 (the count of Gabor orientations), k1 = k2 = 7 on MNIST (5 on textures and
   CIFAR-10); the histogram block size is the one tuned parameter (7×7 on MNIST). Patch-mean removal matters ("we have
   tested the PCANet without patch-mean removal and the performance degrades significantly", footnote 3).

**Numbers (their Tables 9, 10, 12; error rates %).** MNIST (60k train, no augmentation): PCANet-2 0.66, RandNet-2 0.63,
LDANet-2 0.62, PCANet-1 with a 13×13 filter 0.62; ScatNet-2 with an RBF SVM 0.43; ConvNet 0.53; Conv. Maxout + Dropout
0.45 — "the difference is not so statistically meaningful". MNIST *basic* (10k train, 50k test): PCANet-2 1.06,
RandNet-2 1.25, ScatNet-2 1.27, CAE-2 2.48; *bg-img* 10.95 (previous best 12.25); *rot* 7.37 (ScatNet-2 7.48, TIRBM
4.20); *bg-img-rot* 35.48 (TIRBM 35.50). CIFAR-10 (no augmentation, accuracy %): PCANet-2 77.14, with two filter sizes
combined 78.67; K-means triangle 4000 features 79.60; Cuda-convnet2 82.00; NIN 89.59 — "around 11% accuracy
degradation in comparison to state-of-the-art".

**What it cannot do (their §4).** Depth: "two-stage PCANet is in general sufficient ... a deeper architecture does not
necessarily lead to further improvement", and the stated bottleneck is that "the dimension of the resulted feature
would increase exponentially with the number of stages" (L1·L2·… maps, each hashed). Variability: for ImageNet-scale
pose/scale variation "PCANet might not be sufficient", and "some preprocessing of pose alignment and scale
normalization might be needed". RandNet's near-parity (random Gaussian filters, same hashing + histograms) holds only
at 60k training examples (MNIST: RandNet-2 0.63 vs PCANet-2 0.66); at 10k (MNIST basic) RandNet-2 is 1.25 vs
PCANet-2 1.06, 18% relatively worse (their own Table 10). So "the discrete code + counting do the work" is a
60k-example statement; at E35's 30k-window scale the fitted projections are expected to matter (corrected 2026-09-22).

**Reading for us.** PCANet is literally "features → discrete code → count table": a linear projection fitted in closed
form (an eigendecomposition of a 49×49 covariance), a sign quantiser, and histograms (counts) that a linear reader
consumes. Its discrete code is the analogue of E35's window key, except that two windows that differ in a cell can
share a code (the projections' signs agree), which is where its generalisation comes from.

### 1.2 Scattering networks — Bruna & Mallat, "Invariant Scattering Convolution Networks", arXiv:1203.1513 (Mar 2012; IEEE TPAMI 2013); Oyallon & Mallat, arXiv:1412.8659 (2015); Oyallon, Belilovsky, Zagoruyko, arXiv:1703.08961 (2017)

Abstract (Bruna & Mallat, verbatim): "A wavelet scattering network computes a translation invariant image
representation, which is stable to deformations and preserves high frequency information for classification. It
cascades wavelet transform convolutions with non-linear modulus and averaging operators. The first network layer
outputs SIFT-type descriptors whereas the next layers provide complementary invariant information which improves
classification. The mathematical analysis of wavelet scattering networks explains important properties of deep
convolution networks for classification. A scattering representation of stationary processes incorporates higher order
moments and can thus discriminate textures having the same Fourier power spectrum. State of the art classification
results are obtained for handwritten digits and texture discrimination, using a Gaussian kernel SVM and a generative
PCA classifier."

**What is fitted: nothing.** The filters are fixed complex Morlet wavelets ψ_λ at J scales and 8 orientations; a path
p = (λ1, …, λm) gives U[p]x = | ‖x ⋆ ψ_λ1| ⋆ ψ_λ2| … ⋆ ψ_λm| (their §2.2), and the windowed scattering coefficient is
S_J[p]x = U[p]x ⋆ φ_{2^J} — a low-pass average, so it is invariant to translations ≪ 2^J and (their §3.1, the theorem)
Lipschitz-stable to small deformations, which the Fourier modulus is not. Only the classifier is fitted (a PCA affine
model per class, or an RBF SVM). The design knobs are J, the orientations, and the maximum path length m_max.

**Numbers (their Tables 4, 7, 9; error %).** MNIST, full 60k: scattering m_max = 2 + SVM 0.43 (their Conv. Net column
0.53); with 300 training samples: scattering + PCA 4.7 vs Conv. Net 7.18; with 5,000: 1.03 vs 1.52 — "below 5·10³
training samples, the scattering PCA classifier improves results of a deep-learning convolutional network". Going from
m_max = 1 to 2 "reduces errors by about 30%"; order 3 "brings marginal classification improvements". Rotated MNIST:
scattering m_max = 2 PCA 4.4 vs Conv. Net 8.8. CUReT textures (46 training images per class): scattering m_max = 2 PCA
0.2 (Fourier spectrum 1, textons SVM 1.53, MRF 2.46). Oyallon & Mallat 2015 (Table 1, 4; accuracy %): roto-translation
scattering, order 2, + OLS feature reduction, CIFAR-10 82.3 (translation-only order 1: 72.6; order 2: 80.3), Caltech-101
79.9; the supervised CNN row 91.8; CIFAR-100 56.8 vs CNN 65.4. Oyallon et al. 2017 (abstract, verbatim fragment): with a
scattering front-end fixed and a learned ResNet behind it, "a single-crop top 5 error of 11.4% on imagenet ILSVRC2012,
comparable to the Resnet-18 architecture, while utilizing only 10 layers", and "hybrid architectures can yield
excellent performance in the small sample regime, exceeding their end-to-end counterparts".

**What it cannot do.** It is a fixed prior for a KNOWN group (translations, rotations, small deformations): the ~10-point
CIFAR-10 gap to the supervised CNN never closes with a scattering-only representation; on ImageNet everything above
the first few layers must be learned. Nothing in the representation is data-dependent, so it cannot discover which
cells of a game frame co-vary — it presupposes that the invariances are geometric.

### 1.3 K-means features and the encoding — Coates, Lee, Ng, "An Analysis of Single-Layer Networks in Unsupervised Feature Learning" (AISTATS 2011); Coates & Ng, "The Importance of Encoding Versus Training with Sparse Coding and Vector Quantization" (ICML 2011)

**What is fitted.** A dictionary of K centroids c^(k) over whitened 6×6 colour patches (108 dims), by k-means (hard
assignment, no learning rate — each step is a mean, i.e. counting); then every patch is ENCODED by a fixed
non-linearity and the codes are average-pooled over 4 quadrants and read by a linear SVM. The AISTATS encoding is the
"triangle": f_k(x) = max{0, μ(z) − z_k}, z_k = ‖x − c^(k)‖₂, μ(z) the mean of z over k (their eq. 3) — a competition
against the average distance, no threshold to tune. The ICML encoding is the soft threshold f_j = max{0, D^(j)ᵀx − α}
(and its negative half), α fixed.

**Numbers.** AISTATS (Table 2, accuracy %): CIFAR-10 K-means triangle 1600 features 77.9, hard-assignment 68.6, sparse
autoencoder 73.4, sparse RBM 72.4, triangle with 4000 features 79.6 (the previous best, a convolutional RBM, 78.9);
NORB 97.0 (CNN 93.4, DBN 95.0). The finding, in the abstract: "large numbers of hidden nodes and dense feature
extraction are as critical to achieving high performance as the choice of algorithm itself"; whitening, stride 1, a
6-pixel receptive field. ICML (Table 2): CIFAR-10 with the dictionary made of RANDOMLY SAMPLED PATCHES and the soft
threshold: 79.1; sparse coding trained + encoded: 78.8; OMP-1 dictionary + soft threshold 79.4; with 6000 basis vectors
81.5 ("the best known result on CIFAR" then; the deep NN row 80.49). NORB: random patches + soft threshold 95.0 (CNN
94.4). Caltech-101 (30 per class) is the exception: the soft threshold falls to 64–68 vs sparse coding 72.6 — with few
labels "sparse coding excels", the encoder matters when data is scarce.

**What it cannot do.** One layer; stacking k-means layers does not compound (the SoftHebb paper below reports that two
stacked WTA layers do WORSE than one on CIFAR-10/STL-10: 32.9 / 31.5 vs 43.9 / 36.9 — the reason SoftHebb needed
convolutions + pooling + a width factor). The dictionary is a metric quantiser in the INPUT space: it can only group
patches that are close in Euclidean distance after whitening; it has no notion of a target.

**Reading for us.** The claim that matters is "training [the dictionary] hardly matters; the encoder does": random
exemplars = a set of stored windows, exactly E35's memory block; what the paper adds is a SOFT code over many stored
exemplars (distances to all centroids, thresholded) instead of one nearest, and a pooling of codes over a region.
k-means on 9-cell categorical windows is k-modes (Huang 1998, from memory): Hamming distance and the per-cell mode as
the centroid.

### 1.4 Random features — Rahimi & Recht, "Random Features for Large-Scale Kernel Machines" (NeurIPS 2007)

**What is fitted.** Nothing in the features: z(x) = √(2/D) cos(ωᵀx + b) with ω drawn from the Fourier transform of a
shift-invariant kernel (Bochner's theorem, their Theorem 1; Gaussian kernel → Gaussian ω) or random binning; then a
LINEAR machine on z, solved as ridge regression — least squares, closed form. Their abstract: "linear machine learning
algorithms applied to these features outperform state-of-the-art large-scale kernel machines". Table 1 (test error /
training time): Adult 14.9% in 9 s (exact SVM 15.1% in 7 min), Forest Cover 11.6% Fourier / 2.2% binning (exact 2.2% in
44 h), CPU regression 3.6% (exact 11%). Related, from memory: the extreme learning machine (Huang, Zhu, Siew 2006) is the
same recipe with a random hidden layer and a pseudo-inverse readout.

**What it cannot do.** It approximates a FIXED kernel: the similarity between two windows is whatever the kernel says
(Euclidean, or Hamming for a categorical alphabet), not something learned from which windows predict the same target.
The number of random features needed grows with the complexity of the target (their Fig. 3, middle: "Error decays
quickly as P grows", P the number of random features). It
is the cleanest statement that "features + least squares" is a gradient-free pipeline: the readout is a linear solve.

### 1.5 Dictionary learning — Aharon, Elad, Bruckstein, "K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation", IEEE TSP 54(11), 2006

Abstract fragment (Technion listing): "K-SVD is an iterative method that alternates between sparse coding of the
examples based on the current dictionary and a process of updating the dictionary atoms to better fit the data" and
"generalizing the K-means clustering process". **What is fitted:** an overcomplete dictionary D (n × K) such that each
signal y ≈ D x with ‖x‖₀ ≤ T₀; sparse coding by a pursuit (OMP), each atom updated by the rank-1 SVD of the residual
restricted to the signals that use it. No gradient — alternating closed-form steps (k-means is the T₀ = 1, binary-x
special case). Denoising use (Elad & Aharon 2006, from memory): a dictionary trained on the noisy image's own 8×8
patches, sparse-code each patch with the error bound set by σ, average the overlaps. Standing vs BM3D: Levin & Nadler
2011 (§2.4 below) show one image at σ = 75 with KSVD 22.41 dB vs BM3D 23.86 dB; the BM3D paper's Fig. 4 puts K-SVD below
BM3D on every test image ("uniformly outperforms").

**What it cannot do.** The generalisation is LINEAR-sparse: a patch is a few atoms added; a categorical frame cell is
not a sum. Sparse coding at test time is itself an iterative pursuit per patch.

### 1.6 Hebbian deep learning — Journé, Garcia Rodriguez, Guo, Moraitis, "Hebbian Deep Learning Without Feedback", arXiv:2209.11883 (ICLR 2023); Krotov & Hopfield, "Unsupervised Learning by Competing Hidden Units", arXiv:1806.10181 (PNAS 2019)

SoftHebb abstract (verbatim): "Recent approximations to backpropagation (BP) have mitigated many of BP's computational
inefficiencies and incompatibilities with biology, but important limitations still remain. Moreover, the approximations
significantly decrease accuracy in benchmarks, suggesting that an entirely different approach may be more fruitful.
Here, grounded on recent theory for Hebbian learning in soft winner-take-all networks, we present multilayer SoftHebb,
i.e. an algorithm that trains deep neural networks, without any feedback, target, or error signals. As a result, it
achieves efficiency by avoiding weight transport, non-local plasticity, time-locking of layer updates, iterative
equilibria, and (self-) supervisory or other feedback signals -- which were necessary in other approaches. Its
increased efficiency and biological compatibility do not trade off accuracy compared to state-of-the-art bio-plausible
learning, but rather improve it. With up to five hidden layers and an added linear classifier, accuracies on MNIST,
CIFAR-10, STL-10, and ImageNet, respectively reach 99.4%, 80.3%, 76.2%, and 27.3%. In conclusion, SoftHebb shows with a
radically different approach from BP that Deep Learning over few layers may be plausible in the brain and increases the
accuracy of bio-plausible machine learning. Code is available at this https URL."

**What is fitted, and how (their §3).** Per layer of K neurons, a softmax competition y_k = e^{u_k/τ} / Σ_l e^{u_l/τ}
over the weighted inputs u_k (eq. 1), and the local rule Δw_ik = η · y_k · (x_i − u_k · w_ik) (eq. 2), "all variables
... temporally and spatially local to the synapse"; the rule "provably optimizes the model to perform Bayesian inference
of the hidden causes" (Moraitis et al. 2021) — its fixed point is a soft k-means / mixture centroid. Additions that made
it work deep: anti-Hebbian sign for every neuron except the winner; convolutions with pooling, width ×4 per layer; a
per-neuron rate η_i = η (r_i − 1)^q that goes to zero as the weight norm reaches 1; a triangle activation (Coates) for
forward propagation; ONE epoch of unsupervised training; then a linear classifier trained on labels.

**Numbers (their §4, Table 2).** SoftHebb 4–5 hidden conv layers + linear classifier: MNIST 99.35, CIFAR-10 80.31,
STL-10 76.23, ImageNet 27.3, ImageNette 80.98; the SAME architectures trained end-to-end supervised with BP: 99.45,
83.97, 74.51, 85.30 — SoftHebb beats BP on STL-10 (few labels) and trails by 3.7 points on CIFAR-10 and 4.3 on
ImageNette. Random (untrained) weights in the same architecture: STL-10 68.2, ImageNet 14.0. Hard-WTA Hebbian: 54.8 on
STL-10. Krotov & Hopfield (their §4): a 784→2000 hidden layer trained by the "biological" rule (global inhibition,
ranking by current, an anti-Hebbian push on the k = 2 runner-up), then a 2000→10 top layer trained by SGD: MNIST test
error 1.52%, the same as the end-to-end SGD network (≈1.5%); CIFAR-10 (fully connected, 3072→2000→10): 49.25% error vs
44.74% for end-to-end SGD.

**What they cannot do.** Both are ONLINE ITERATIVE rules with a learning rate — not a gradient of a global loss, but
also not closed form; SoftHebb's authors say plainly "BP in more complex models significantly outperforms SoftHebb",
and shallow fully-connected Hebbian layers do not stack (32.9 / 31.5 for two layers vs 43.9 / 36.9 for one). The
features are unsupervised: the target enters only through the final linear reader, which in both papers is trained by
SGD (a linear solve or a count table would do the same job without one). For a mixture-of-categoricals model the same
fixed point is reachable by BATCH EM — a few passes of soft-assignment counting, no η.

---

## 2. Non-learned denoisers, and where the memory block sits among them

### 2.1 Non-local means — Buades, Coll, Morel, "A non-local algorithm for image denoising", CVPR 2005 (60–65); the IPOL 2011 article "Non-Local Means Denoising" (doi 10.5201/ipol.2011.bcm_nlm)

**Mechanism (IPOL §2, pixelwise).** The restored value at p is a weighted average over a search window B(p, r):
û(p) = (1/C(p)) Σ_{q ∈ B(p,r)} u(q) w(p, q), with the weight a function of the squared Euclidean distance d² between
the (2f+1)×(2f+1) PATCHES around p and q: w(p, q) = exp(−max(d² − 2σ², 0) / h²), h = kσ. Patches closer than the noise
floor 2σ² get weight 1; farther ones decay; the reference pixel's own weight is capped at the largest neighbour weight.
Parameters (their Table 1, grey): σ ≤ 15 → 3×3 patches, 21×21 search, h = 0.40σ; up to σ = 100 → 11×11 patches, 35×35
search, h = 0.30σ. The patchwise variant averages the (2f+1)² overlapping estimates of every pixel, "the gain on PSNR
by the patchwise implementation, due to the larger noise reduction of the final aggregation". The 2005 abstract's
principle: "replacing the color of a pixel with an average of the colors of similar pixels", and the "method noise"
(what a denoiser removes) as the yardstick. The IPOL text's own caveat: "the term 'semi-local' would have been more
appropriate" — the search is a 21×21 window, not the image.

**Standing.** NLM is the ancestor of every patch-similarity denoiser; BM3D (below) is what happens when the similar
patches are filtered JOINTLY instead of averaged; DnCNN's Table II has no NLM column because BM3D superseded it. NLM
with kernel width h → 0 is the single nearest patch; with a Hamming distance on a discrete alphabet, weights
exp(−d/h), and the stored windows' count vectors as the values, it is EXACTLY E35's `NearestRule` at h → 0 — a
Nadaraya-Watson regression P(y | w) = Σ_s K(d(w,s)) n_s(y) / Σ_s K(d(w,s)) n_s over stored windows s. §3 measures
this.

### 2.2 BM3D — Dabov, Foi, Katkovnik, Egiazarian, "Image denoising by sparse 3D transform-domain collaborative filtering", IEEE TIP 16(8), 2007

**Mechanism (their §II, the two-step algorithm).** Step 1, basic estimate: for every reference block, *grouping* —
block-match similar blocks (Euclidean distance under a threshold) and stack them in a 3-D group; *collaborative
hard-thresholding* — a 3-D transform (2-D DCT/wavelet across the block, 1-D Haar along the stack), hard-threshold the
spectrum, invert, return every block's estimate to its position; *aggregation* — a weighted average of all overlapping
block estimates (weights ∝ 1 / number of retained coefficients). Step 2, final estimate: group again using the BASIC
estimate for the matching, then *collaborative Wiener filtering* of the noisy group with the basic estimate's energy
spectrum as the pilot, and aggregate. Their two motivations, in their words: the basic estimate "allows to improve the
grouping by block-matching", and Wiener filtering with it as pilot "is much more effective and accurate than the
simple hard-thresholding". Nothing is learned from data; the transforms are fixed; the "sparsity" is enforced by the
group being similar.

**Numbers (their Table III, output PSNR dB, grey).** Lena 512²: σ = 10 → 35.93, σ = 25 → 32.08, σ = 50 → 28.86.
Barbara σ = 25 → 30.72, House 32.86, Cameraman 29.45. The paper's comparison figure puts BM3D above BLS-GSM, K-SVD,
exemplar-based and SA-DCT on every image, most on House and Barbara (edges and textures "enable a very effective
grouping").

### 2.3 The learned denoiser it is measured against — Zhang, Zuo, Chen, Meng, Zhang, "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising", arXiv:1608.03981 (IEEE TIP 2017)

DnCNN: 17–20 conv layers, residual learning (the net predicts the NOISE), batch norm, trained by SGD on 400 images.
Their Table II, average PSNR on BSD68: σ = 15: BM3D 31.07, DnCNN-S 31.73; σ = 25: 28.57 vs 29.23; σ = 50: 25.62 vs
26.23 — "our DnCNN-S model outperforms BM3D by 0.6dB on all the three noise levels", against the folklore they cite
that "few methods can outperform BM3D by more than 0.3dB on average". Set12 σ = 25: BM3D 29.969 vs DnCNN-S 30.436.
DnCNN is not the ceiling (corrected 2026-09-22): DRUNet — Zhang, Li, Zuo, Zhang, Van Gool, Timofte, "Plug-and-Play
Image Restoration with Deep Denoiser Prior", IEEE TPAMI 2021, arXiv:2008.13751, Table 1 (read) — reaches on BSD68
σ = 15 / 25 / 50: 31.91 / 29.48 / 26.59 vs BM3D 31.08 / 28.57 / 25.60 and DnCNN 31.73 / 29.23 / 26.23, i.e.
+0.83 / +0.91 / +0.99 dB over BM3D (Set12: 33.25 / 30.94 / 27.90 vs BM3D 32.37 / 29.97 / 26.72). So the
learned-vs-fixed gap in Gaussian denoising is about 1 dB (DnCNN 2017 +0.6, DRUNet 2021 +0.9), not 0.6. That part of
the gap is the receptive field (DnCNN sees 35–41 pixels; BM3D matches 8×8 blocks within a 39×39 window; DRUNet's
U-Net sees more) is this note's own conjecture, not a result of either paper.

### 2.4 The bound that says why lookup runs out — Levin & Nadler, "Natural Image Denoising: Optimality and Inherent Bounds", CVPR 2011

They represent the natural-image prior non-parametrically by 10¹⁰ clean patches and compute the Bayesian MMSE for a
denoiser that sees a k×k window — exactly a huge lookup table with a Gaussian kernel: their optimum (eqs. 10–12) is
the KERNEL average over the database, μ̂(y) = Σ_i p(y | x_i) x_i / Σ_i p(y | x_i), non-local means with bandwidth σ,
NOT a nearest-neighbour rule; and their neighbour-density statement below is about when this lower bound is TIGHT
(enough neighbours within the kernel's reach), not about the accuracy of a hard nearest estimator (clarified
2026-09-22 — the synthesis note's F1 had read it as a density limit on E35's hard rule, which its own F2 refutes: the
soft kernel over the same stored windows beats the hard rule by 0.05–0.10). Findings (their §3–4): for small
windows "state of the art denoising algorithms are approaching optimality and cannot be further improved beyond
∼0.1dB"; BM3D is within 0.1 dB of the k-window optimum for small k. The neighbour density is the limit: at σ = 18, "for
3×3 patches, 99% of the examples had more than 2,000 neighbors. In contrast, for a 9×9 patch size, 13% of the examples
had no neighbors within this distance" (their Fig. 3), and "increasing the support size of non-parametric algorithms
might lead to a dead-end, and parametric approaches are required". That is the E35 question in their vocabulary: the
window is where the lookup's density gives out, and a bigger window needs a model with shared structure.

---

## 3. What the reader's feel-experiment measured — `research/r_window_neighbours.py`

One-shot block only (corrupted window at t → clean centre), radius 1, E35's frames (115 training, 135 held out), 4
corruption draws per training frame (30,640 pairs), seed 0. Numbers from `research/r_window_neighbours.json`:

| t | distinct training windows | held-out windows UNSEEN | Hamming distance to the nearest stored window (share) | input | hard nearest (E35's rule) | soft Hamming kernel, h by leave-one-out | naive Bayes over the 9 cells | linear mix of the 9 per-cell tables |
|---|---|---|---|---|---|---|---|---|
| 0.25 | 21,029 | 0.626 | d=0 0.37, 1 0.36, 2 0.20, 3 0.06, 4+ 0.01 | 0.767 | 0.941 | **0.954** (h = 0.5; 0.35 bits/cell) | 0.926 (0.63 bits) | 0.889 (0.97 bits) |
| 0.50 | 28,944 | 0.923 | d=0 0.08, 1 0.22, 2 0.31, 3 0.28, 4 0.10, 5+ 0.01 | 0.530 | 0.849 | **0.900** (h = 0.75; 0.69 bits) | 0.870 (0.77 bits) | 0.822 (1.06 bits) |
| 0.75 | 30,509 | 0.995 | d=0 0.01, 1 0.08, 2 0.24, 3 0.25, 4 0.38, 5 0.05 | 0.285 | 0.730 | **0.829** (h = 0.75; 0.87 bits) | 0.811 (0.91 bits) | 0.792 (1.14 bits) |

Hard-nearest accuracy by distance at t = 0.5: d = 0 → 1.000, 1 → 0.980, 2 → 0.888, 3 → 0.748, 4 → 0.641, 5 → 0.505.

What it establishes. (i) The apparatus reproduces E35's one shot (0.941 / 0.849 / 0.730 vs E35's 0.940 / 0.854 /
0.732). (ii) At t ≥ 0.5 the "table" is not a lookup at all: 92–99.5% of held-out windows were never stored, and
30,509 of 30,640 training windows are distinct at t = 0.75 — the block is nearest-neighbour regression, and its
accuracy falls with the distance it has to bridge (Levin & Nadler's neighbour-density limit at the scale of a 9-cell
window). (iii) The SOFT kernel over all stored windows, with the one width h picked by leave-one-out likelihood on a
2,000-window training subsample (a grid of 8 values, no gradient), beats the hard rule by +0.013 / +0.051 / +0.099,
and one soft block beats E35's whole 8-block hard chain (0.946 / 0.880 / 0.800). The parallel run
`research/expE_nlm_block.log` (a vote of the k nearest windows' majorities) agrees: k = all, h = 0.5, one shot
0.903 / 0.816 and chain 0.923 / 0.851 at t = 0.5 / 0.75. (iv) Naive Bayes — the closed-form PRODUCT of the 9 per-cell
likelihoods P(w_i | y) with unit exponents — is second (0.870 / 0.811), well above the hard rule at t ≥ 0.5, and
generalises to every window; the linear mixture of the same 9 tables is last (0.822 / 0.792). The ordering
product > linear is E34's (1.801 vs 1.820 bits/char), but here the product does NOT over-sharpen (0.77 bits/cell at
t = 0.5), because the 9 cells given the centre are near conditionally independent — the case where unit exponents are
right — whereas E34's 18 text contexts are nested and overlapping.

---

## 4. Where the "learning" sits in each method — the table the design needs

| method | features fitted by | readout fitted by | what a gradient would add |
|---|---|---|---|
| PCANet | eigenvectors of the patch covariance (closed form); sign quantiser; block histograms (counts) | linear SVM (convex) | the ~11 points on CIFAR-10 (77 → 88–90) |
| ScatNet | nothing (fixed wavelets, modulus, averaging) | PCA affine model or SVM | ~10 points on CIFAR-10 (82.3 → 91.8); everything above layer 2 on ImageNet |
| Coates k-means | k-means (means = counting) or random exemplars; a fixed encoder (triangle / soft threshold) | linear SVM | ~1–2 points on CIFAR-10 at the time; more with depth |
| Rahimi–Recht | random draws from the kernel's spectrum | ridge regression (closed form) | nothing at fixed kernel; the kernel itself |
| K-SVD | alternating OMP + rank-1 SVD (closed-form steps, iterated) | the sparse code | ~1 dB vs BM3D in denoising |
| SoftHebb / Krotov–Hopfield | a local Hebbian rule with a learning rate (fixed point = soft k-means) | linear layer by SGD | 3.7 points on CIFAR-10, 4.3 on ImageNette; ~4.5 on CIFAR-10 fully connected |
| NLM / BM3D | nothing (patch distance; fixed transforms) | — | ~1 dB (DnCNN 2017 +0.6, DRUNet 2021 +0.9 over BM3D on BSD68) |
| E35 memory block | stored windows (counting) | the nearest window's majority | not measured; a soft kernel already gives +0.05–0.10 |

Two regularities. First, in every "gradient-free features" paper the readout is still SOLVED — by least squares, a
convex SVM, or SGD on a linear layer; the analogue in ZipLearn is the count table keyed on the feature code (PCANet's
histogram is that table). Second, the residual gap to backprop is the same shape everywhere: roughly 4–10 points on
natural-image classification (features the data-independent or unsupervised method cannot invent: the target never
shapes them), and ~1 dB on denoising (DnCNN +0.6, DRUNet +0.9; that the receptive field is the cause is this note's
conjecture). In classification the gap is not "the feature", it is "the feature chosen for the target".

---

## 5. Candidates for a gradient-free block that generalises by features

Each names what is fitted, by what closed-form or counting step, and what refutes it. None is designed beyond what
is written here.

### (a) Discrete game frames (E35's block: 9 cells × 16 colours → centre cell)

1. **The soft Hamming kernel (non-local means on the stored windows) — measured, §3.** Keep E35's memory block as is;
   replace `predict_nearest`'s argmin by P(y | w) ∝ Σ_s exp(−d_H(w, s)/h) n_s(y); h by leave-one-out likelihood on the
   stored windows (one scalar, a grid). Gain +0.05 / +0.10 at t = 0.5 / 0.75 one shot. Cost: a Hamming distance to
   every stored window per cell (30k × 9 comparisons; fine at 64×64, 122k cells per frame ≈ 3·10⁹ byte compares, a
   second in numpy). Not a feature — the kernel is fixed — but the honest baseline any feature block must beat. What
   the price must count: the leave-one-out likelihood IS a description length of the stored windows under the
   kernel, so the sleep pass can price a memory by the bits it saves on its neighbours, not by its entries (E35's open
   point).
2. **Naive Bayes / Chow–Liu factoring of the window — the closed-form product, measured, §3.** Nine per-cell tables
   P(w_i | y) and P(y), 9 × 17 × 16 counts, generalise to every window at 0.87 / 0.81 (t = 0.5 / 0.75). The next step
   up with no gradient is a Chow–Liu tree (Chow & Liu 1968, from memory): the maximum-spanning tree of pairwise mutual
   informations among the 9 cells + target, giving P(y, w) = Π P(node | parent) from pairwise counts — exact
   normalisation, no exponent to learn, dependence between cells captured to first order. Refuted if it does not beat
   naive Bayes' 0.870 at t = 0.5.
3. **PCANet on one-hot windows: closed-form projections → sign code → count table.** The window as a 9×17 one-hot
   vector (153 dims); the L leading eigenvectors of its covariance (from the same 30k windows, one 153×153
   eigendecomposition); the sign bits of the L projections as the table key (2^L keys, L = 8 → 256, L = 12 → 4096)
   instead of the raw 9-cell key; counts of the centre colour per key. Two windows differing in a noise cell share a
   key when the projections agree. RandNet says random projections would do nearly as well at 60k examples (not at
   10k, §1.1 — at E35's 30k windows the PCA fit is expected to matter) — the cheapest test of
   whether a shared discrete code beats the raw key. Refuted if the key's table at t = 0.5 is below the soft kernel's
   0.900 — then the code loses more than it shares.
4. **k-modes centroids + a soft assignment (Coates on categorical data).** K centroids over the 9-cell windows by
   k-modes (per-cell mode, Hamming distance; each step is counting); the code is the triangle activation over
   distances to all K centroids, or simply the nearest centroid as the table key (a coarser key than 3.). Coates'
   result that random exemplars ≈ trained centroids says the memory block's stored windows ARE the dictionary
   already; what k-modes adds is fewer, more general keys — the sleep pass's goal by another route, and priced the
   same way (bits of the pairs under the merged table).
5. **The receptive field, not the block.** The learned denoisers' ~1 dB over BM3D (DnCNN 0.6, DRUNet 0.9; the receptive-field
   attribution is this note's conjecture) and Levin & Nadler's support-size argument both say
   the next gain is context; a radius-2 window (25 cells) makes the raw key hopeless (16^25) but is exactly where
   features 2–4 keep generalising and the soft kernel degrades (the Hamming distance at 25 cells is dominated by noise
   cells). Test the crossover: radius 2 with candidates 2–4 against radius 1 with candidate 1.

### (b) Character text (E33's tables; E35's ±3 masked-character block; the E34 library)

1. **Similarity-weighted contexts = non-local means on text (Dagan, Lee, Pereira, "Similarity-Based Methods for Word
   Sense Disambiguation", arXiv cmp-lg/9708010, verified; the 1999 Machine Learning paper from memory).** Their abstract:
   the similarity-based estimators "perform up to 40% better" than back-off on a controlled pseudo-word task. For an
   unseen order-k context c, P(y | c) = Σ_{c'} K(c, c') P(y | c') over the SEEN contexts c' of the same order, with a
   distributional distance between contexts (the KL or Jensen–Shannon distance between their next-character
   distributions, from counts) — the text analogue of candidate (a)1, and the piece E33's blended back-off lacks:
   back-off shrinks the context; similarity swaps it for a look-alike. One kernel width, chosen by prequential bits.
2. **Class-based contexts (Brown et al. 1992, "Class-based n-gram models of natural language", from memory).**
   Agglomerative clustering of characters (or of the current-word expert's words) maximising the mutual information
   between adjacent classes — greedy merges scored by counts, no gradient — then n-gram tables keyed on CLASS
   sequences run beside the character tables and are mixed in as more experts (E34's library). This is PCANet's
   discrete code for text: two contexts that differ in a character of the same class share a key. Refuted if the
   class-keyed expert adds nothing over the chain's 1.823 in the E34 mixer.
3. **Closed-form embeddings: SVD of the (context, next-character) count matrix (Levy & Goldberg 2014, NeurIPS,
   "Neural Word Embedding as Implicit Matrix Factorization" — that SGNS factorises the shifted-PMI matrix and that an
   SVD of it matches SGNS on word-similarity tasks; from the paper's abstract as summarised, not fetched).** A
   character or a context becomes a dense vector by one SVD of counts; the soft kernel of (b)1 then runs on cosine
   distance in that space; or k-means over the vectors gives the classes of (b)2 (this is what the parallel
   `research/expF_cluster_features_text.py` is trying: k-means k = 256–4096 and 16–32 PCA directions over one-hot ±3
   windows as the block's key).
4. **What text says against features at radius 3.** E35's own null (chain = one shot at ±3) and expF's early rows
   (a mutual information of only 0.43 bits between the nearest neighbour position and the target) say the ±3 window
   carries too little information for ANY key to make a denoising chain pay; the gain on text must come from the
   long contexts E33/E34 already use, which is where (b)1–2 operate. A feature block on text is not yet worth a
   diffusion chain; it is worth an expert.

---

## 6. Open questions

- The price. The two-part code as written (`arcgames.py: LocalRule._price`) prices a memory by its entries; the
  leave-one-out likelihood under the kernel prices it by what it predicts for its neighbours. Which price makes the
  sleep pass keep exactly the windows the soft kernel needs — and does the merged-key table of candidate (a)3–4 then
  compress on its own? Not designed.
- Whether candidate (a)2's product stays honest at radius 2 (25 near-independent cells) or over-sharpens as E34's
  18 nested contexts did; the Chow–Liu tree is the closed-form answer if it does.
- SoftHebb's fixed point by batch EM: does a mixture of categoricals over 9-cell windows (K components, soft
  assignment counting, 3–5 passes) give better keys than k-modes' hard ones? Coates' data says the encoder matters
  more than the centroids; untested here.
- None of the surveyed methods fits the features TO the target without a gradient (PCA/k-means/scattering are
  target-blind; the target enters at the readout). LDANet (PCANet's supervised variant, closed-form LDA filters) is the
  one exception in the survey and gained nothing over PCA (0.62 vs 0.66 on MNIST). Whether a target-aware closed form
  (Chow–Liu with the target as the root; information-gain splits as in a decision tree, all counts) closes any of
  the 4–10-point gap is the open question for both frames and text.
