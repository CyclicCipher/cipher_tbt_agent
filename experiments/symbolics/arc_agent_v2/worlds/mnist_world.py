"""MNIST (docs/LAB.md part 12): handwritten digits, shown to the brain as patches at gaze positions, as SDRs.

**The data:** the idx files at `C:/Users/julia/GitHub/data/MNIST/raw/`, outside the repository, read with numpy.
Per seed, a random `n_train` of the 60,000 training images and a random `n_test` of the 10,000 test images.

**What the brain is shown of an image:** an episode of its 16 patches of 7 × 7 pixels, on a 4 × 4 grid.
- **The "retina" sensor:** one patch, transduced to an SDR.
  - **The transduction:** the patch's 49 pixels as ±1 (on above 127), projected by a fixed Gaussian matrix (d × 49, the
    world's seed), with the top w of the d sums kept.
  - **What it is:** random projection and winner-take-all (Dasgupta, Stevens & Navlakha 2017, *Science* 358:793).
    Similar patches overlap, and a blank patch always gives one SDR.
- **The "gaze" sensor:** the efference copy of the gaze's shift to the next patch. It is the shift's code in the
  region's innate sheet (`config.retinotopy`): one genome for the body and the cortex.
- **The order:** the patch at (0, 0) first (every episode starts at one place, P1), then the other 15 in a random
  order, each after the gaze shift that brings it.

**Labels** live only here, to score. The brain never sees a label, a pixel or a position.
"""
import os

import numpy as np

from arc_agent_v2.architecture.HDClibrary import SDR

DATA = "C:/Users/julia/GitHub/data/MNIST/raw"
PATCH, GRID = 7, 4
_cache = {}


def _idx(name):
    """An idx file as a numpy array (cached)."""
    if name not in _cache:
        with open(os.path.join(DATA, name), "rb") as f:
            raw = f.read()
        nd = raw[3]
        dims = [int.from_bytes(raw[4 + 4 * i:8 + 4 * i], "big") for i in range(nd)]
        _cache[name] = np.frombuffer(raw, np.uint8, offset=4 + 4 * nd).reshape(dims)
    return _cache[name]


class MnistWorld:
    def __init__(self, sheet, d=2048, w=40, seed=0, n_train=3000, n_test=1000):
        self.sheet, self.d, self.w, self.seed = sheet, d, w, seed
        rng = np.random.default_rng((seed, 28))
        tr, te = _idx("train-images-idx3-ubyte"), _idx("t10k-images-idx3-ubyte")
        tri = np.sort(rng.choice(len(tr), n_train, replace=False))
        tei = np.sort(rng.choice(len(te), n_test, replace=False))
        self.train_labels = _idx("train-labels-idx1-ubyte")[tri].astype(int)
        self.test_labels = _idx("t10k-labels-idx1-ubyte")[tei].astype(int)
        self.R = np.random.default_rng((seed, 4949)).standard_normal((d, PATCH * PATCH))
        self.train = self._encode(tr[tri])              # [image][position r * GRID + c] -> SDR
        self.test = self._encode(te[tei])
        self.positions = [(r, c) for r in range(GRID) for c in range(GRID)]

    def _encode(self, images):
        n = len(images)
        p = images.reshape(n, GRID, PATCH, GRID, PATCH).transpose(0, 1, 3, 2, 4).reshape(n * GRID * GRID, -1)
        v = np.where(p > 127, 1.0, -1.0)
        scores = v @ self.R.T
        top = np.argpartition(-scores, self.w, axis=1)[:, :self.w]
        sdrs = [SDR._sorted(self.d, np.sort(t).astype(np.int32)) for t in top]
        return [sdrs[i * GRID * GRID:(i + 1) * GRID * GRID] for i in range(n)]

    def episode(self, split, k):
        """Image k of "train" or "test" as an episode: [(sensor, SDR)]."""
        patches = (self.train if split == "train" else self.test)[k]
        rng = np.random.default_rng((self.seed, 77, 0 if split == "train" else 1, k))
        order = [(0, 0)] + [self.positions[1 + i] for i in rng.permutation(len(self.positions) - 1)]
        out, here = [], (0, 0)
        for r, c in order:
            if (r, c) != here:
                out.append(("gaze", self.sheet.shift(r - here[0], c - here[1])))
                here = (r, c)
            out.append(("retina", patches[r * GRID + c]))
        return out

    def nearest_neighbour(self):
        """The encoder's check (U4): each test image's nearest training image by the summed overlap of their patch
        SDRs at each position; the accuracy of its label."""
        from scipy.sparse import csr_matrix
        score = np.zeros((len(self.test), len(self.train)))
        for pos in range(GRID * GRID):
            def mat(images):
                rows = np.repeat(np.arange(len(images)), self.w)
                cols = np.concatenate([im[pos].idx for im in images])
                return csr_matrix((np.ones(cols.size), (rows, cols)), shape=(len(images), self.d))
            score += (mat(self.test) @ mat(self.train).T).toarray()
        return float((self.train_labels[np.argmax(score, 1)] == self.test_labels).mean())
