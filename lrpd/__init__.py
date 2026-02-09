"""
LRPD: Low-Rank Plus Diagonal matrix operations.

Efficient operations for matrices of the form A = diag(d) + U U^T,
including Woodbury inversion, log-determinant, Schur complements,
online updates, and optimal decomposition.

Core operations are O(n*k^2) or O(n*k) instead of O(n^3).

References:
  - Woodbury identity: Max A. Woodbury (1950)
  - Matrix determinant lemma
  - Yeon & Anitescu 2025: Alt algorithm for optimal LRPD decomposition
"""

from .woodbury import (
    woodbury_inverse_components,
    lrpd_solve,
    lrpd_matvec,
    lrpd_inv_diag,
)

from .log_det import lrpd_log_det

from .schur import (
    schur_complement_diag,
    schur_complement_full_diag,
)

from .online_update import (
    online_lrpd_update,
)

from .alt_decompose import (
    alt_decompose,
    nystrom_alt_decompose,
    alt_decompose_from_factors,
)

from .psd_bound import (
    psd_upper_bound,
    psd_upper_bound_from_eigendecomp,
)

__all__ = [
    "woodbury_inverse_components",
    "lrpd_solve",
    "lrpd_matvec",
    "lrpd_inv_diag",
    "lrpd_log_det",
    "schur_complement_diag",
    "schur_complement_full_diag",
    "online_lrpd_update",
    "alt_decompose",
    "nystrom_alt_decompose",
    "alt_decompose_from_factors",
    "psd_upper_bound",
    "psd_upper_bound_from_eigendecomp",
]
