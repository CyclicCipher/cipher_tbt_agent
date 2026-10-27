"""
test_attention_residuals_symbolic.py
Rigorous benchmark: Standard Additive Residuals vs. Attention Residuals (AttnRes)
on a Piecewise / Domain-Gated Target Function.

Target:
    f(x) = x^2      for x < 0
    f(x) = sin(x)   for x >= 0
"""

import numpy as np


class StandardAdditiveRegressor:
    """
    Standard Additive Residual Model:
    y_pred(x) = sum_i c_i * phi_i(x)
    Each layer adds its contribution with a fixed uniform skip-weight (1.0).
    Solved via Ordinary Least Squares (the theoretical lower bound for additive models).
    """
    def __init__(self, vocab_names, vocab_funcs):
        self.vocab_names = vocab_names
        self.vocab_funcs = vocab_funcs
        self.coef_ = None

    def fit(self, X, y):
        # Design matrix: shape (N, M)
        Phi = np.column_stack([f(X) for f in self.vocab_funcs])
        # Optimal OLS fit
        self.coef_, residuals, rank, s = np.linalg.lstsq(Phi, y, rcond=None)
        return self

    def predict(self, X):
        Phi = np.column_stack([f(X) for f in self.vocab_funcs])
        return Phi @ self.coef_


class AttentionResidualRegressor:
    """
    Attention Residuals (AttnRes):
    Instead of rigid uniform addition, each term's contribution is routed
    via input-dependent depth attention:
        alpha_i(x) = softmax(q(x)^T k_i)
        y_pred(x)  = sum_i alpha_i(x) * (c_i * phi_i(x))

    Attention keys (k_i) and coefficients (c_i) are co-optimized with Adam.
    """
    def __init__(self, vocab_names, vocab_funcs, lr=0.05, epochs=600):
        self.vocab_names = vocab_names
        self.vocab_funcs = vocab_funcs
        self.lr = lr
        self.epochs = epochs
        self.K = None      # Attention keys: shape (M, 2)
        self.c = None      # Coefficients: shape (M,)

    def fit(self, X, y):
        N = len(X)
        M = len(self.vocab_funcs)

        # Context features for query q(x) = [x, 1]
        X_aug = np.column_stack([X, np.ones_like(X)])  # (N, 2)
        Phi = np.column_stack([f(X) for f in self.vocab_funcs])  # (N, M)

        # Initialize parameters
        np.random.seed(42)
        self.K = np.random.randn(M, 2) * 0.1
        self.c = np.ones(M) * 0.5

        # Adam optimizer buffers
        m_K, v_K = np.zeros_like(self.K), np.zeros_like(self.K)
        m_c, v_c = np.zeros_like(self.c), np.zeros_like(self.c)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for epoch in range(1, self.epochs + 1):
            # 1. Forward Pass
            # Attention logits: S = X_aug @ K.T -> shape (N, M)
            S = X_aug @ self.K.T
            S_max = np.max(S, axis=1, keepdims=True)
            exp_S = np.exp(S - S_max)
            alpha = exp_S / np.sum(exp_S, axis=1, keepdims=True)  # (N, M)

            # Modulated outputs: V = c * Phi -> shape (N, M)
            V = self.c * Phi
            y_pred = np.sum(alpha * V, axis=1)  # (N,)

            # Loss: MSE
            loss = np.mean((y_pred - y) ** 2)

            # 2. Backward Pass (Analytical Gradients)
            dL_dpred = 2.0 * (y_pred - y) / N                      # (N,)
            dL_dV = dL_dpred[:, None] * alpha                     # (N, M)
            dL_dc = np.sum(dL_dV * Phi, axis=0)                   # (M,)

            dL_dalpha = dL_dpred[:, None] * V                     # (N, M)
            sum_dalpha_alpha = np.sum(dL_dalpha * alpha, axis=1, keepdims=True)
            dL_dS = alpha * (dL_dalpha - sum_dalpha_alpha)        # (N, M)
            dL_dK = dL_dS.T @ X_aug                               # (M, 2)

            # 3. Adam Update
            # Keys (K)
            m_K = beta1 * m_K + (1 - beta1) * dL_dK
            v_K = beta2 * v_K + (1 - beta2) * (dL_dK ** 2)
            m_K_hat = m_K / (1 - beta1 ** epoch)
            v_K_hat = v_K / (1 - beta2 ** epoch)
            self.K -= self.lr * m_K_hat / (np.sqrt(v_K_hat) + eps)

            # Coefficients (c)
            m_c = beta1 * m_c + (1 - beta1) * dL_dc
            v_c = beta2 * v_c + (1 - beta2) * (dL_dc ** 2)
            m_c_hat = m_c / (1 - beta1 ** epoch)
            v_c_hat = v_c / (1 - beta2 ** epoch)
            self.c -= self.lr * m_c_hat / (np.sqrt(v_c_hat) + eps)

        return self

    def get_attention_weights(self, X):
        X_aug = np.column_stack([X, np.ones_like(X)])
        S = X_aug @ self.K.T
        S_max = np.max(S, axis=1, keepdims=True)
        exp_S = np.exp(S - S_max)
        return exp_S / np.sum(exp_S, axis=1, keepdims=True)

    def predict(self, X):
        X_aug = np.column_stack([X, np.ones_like(X)])
        Phi = np.column_stack([f(X) for f in self.vocab_funcs])
        alpha = self.get_attention_weights(X)
        V = self.c * Phi
        return np.sum(alpha * V, axis=1)


def run_experiment():
    print("=" * 70)
    print("ATTENTION RESIDUALS (AttnRes) vs. STANDARD ADDITIVE RESIDUALS")
    print("=" * 70)

    np.random.seed(42)
    X = np.linspace(-3.0, 3.0, 400)
    # Target: Piecewise function
    y = np.where(X < 0, X**2, np.sin(X))

    vocab_names = ["1.0", "x", "x^2", "sin(x)", "cos(x)"]
    vocab_funcs = [
        lambda x: np.ones_like(x),
        lambda x: x,
        lambda x: x**2,
        lambda x: np.sin(x),
        lambda x: np.cos(x),
    ]

    # 1. Standard Additive Residual Model
    std_model = StandardAdditiveRegressor(vocab_names, vocab_funcs)
    std_model.fit(X, y)
    std_preds = std_model.predict(X)
    std_mse = np.mean((y - std_preds) ** 2)

    # 2. Attention Residual Model
    attn_model = AttentionResidualRegressor(vocab_names, vocab_funcs, lr=0.08, epochs=700)
    attn_model.fit(X, y)
    attn_preds = attn_model.predict(X)
    attn_mse = np.mean((y - attn_preds) ** 2)

    # -------------------------------------------------------------
    # DIAGNOSTIC PROFILE: WHAT DID THEY LEARN?
    # -------------------------------------------------------------
    print("\n--- [MODEL 1: Standard Additive Model (Fixed Skip Connections)] ---")
    print("Formula: y_pred = sum_i c_i * phi_i(x)")
    for name, coef in zip(vocab_names, std_model.coef_):
        print(f"  Term [{name:<7}]: Coefficient = {coef:+.4f}")
    print(f"  -> Final Standard Additive MSE: {std_mse:.6f}")

    print("\n--- [MODEL 2: Attention Residuals (Learned Depth-Wise Attention)] ---")
    print("Formula: y_pred = sum_i alpha_i(x) * c_i * phi_i(x)")
    for name, coef in zip(vocab_names, attn_model.c):
        print(f"  Term [{name:<7}]: Learned Base Coefficient = {coef:+.4f}")

    # Inspect attention routing across domain
    probe_points = np.array([-3.0, -2.0, -1.0, -0.1, 0.1, 1.0, 2.0, 3.0])
    probe_alphas = attn_model.get_attention_weights(probe_points)

    print("\n--- [Learned Attention Routing Map: alpha_i(x)] ---")
    header = f"{'x':<6} | " + " | ".join([f"{name:^7}" for name in vocab_names])
    print(header)
    print("-" * len(header))
    for x_val, alpha_row in zip(probe_points, probe_alphas):
        row_str = " | ".join([f"{w:7.2%}" for w in alpha_row])
        print(f"{x_val:+6.1f} | {row_str}")

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"[*] Standard Additive Residual MSE : {std_mse:.6f}")
    print(f"[*] Attention Residuals (AttnRes) MSE: {attn_mse:.6f}")
    speedup = std_mse / attn_mse
    print(f"[+] Error Reduction Factor         : {speedup:.1f}x lower error with AttnRes!")


if __name__ == "__main__":
    run_experiment()