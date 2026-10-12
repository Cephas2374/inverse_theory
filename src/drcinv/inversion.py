"""Regularised potential-field inversion: Tikhonov, fabric-graph, compact and TV norms.

Objective (Eq. 2 of the paper, single property):
    Phi(m) = ||Wd (G m - d)||^2
           + beta * ( ||W_s Rs m||^2 + ||W_r Rr m||^2 )
           + ||Wa (m - m_arch)||^2
    Rs = sqrt(alpha_s) Wz              smallness
    Rr = D Wz (isotropic)  or  sqrt(lambda w) B Wz (fabric graph)   smoothness
W_s, W_r are identity for l2 (Tikhonov / graph-Laplacian) and IRLS weights for lp norms:
    w = eps^((2-p)/2) (x^2 + eps^2)^((p-2)/4),  x = residual of the block,
which gives a minimum-support (compact, p = 0; Last & Kubik 1983; Portniaguine & Zhdanov
1999) or total-variation-like (p = 1) solution. All blocks are stacked into one whitened
least-squares system K m ~ r and solved with LSQR. Posterior samples use
randomise-then-optimise: r -> r + N(0, I). That is exact for the l2 (Gaussian) cases and a
Laplace-type approximation about the converged IRLS weights for the lp cases.
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, lsqr


def depth_weights(mesh, z_obs, exponent=1.0):
    """Li & Oldenburg (1998) style depth weighting w(z) = (z + z0)^(-exponent), normalised."""
    _, _, Z = mesh.centers()
    z0 = mesh.dz / 2
    w = (z_obs - Z + z0) ** (-exponent)
    return w / w.max()


def sensitivity_weights(G, sd, exponent=0.5):
    """Integrated-sensitivity weighting (Li & Oldenburg, 2000): w = (sum_i (G_ij/sd_i)^2)^(exponent/2)."""
    w = np.sqrt(np.sum((G / sd[:, None]) ** 2, axis=0)) ** exponent
    return w / w.max()


def lp_weights(x, p, eps):
    """IRLS (Lawson) weights for an lp penalty, scaled to 1 at x = 0."""
    return eps ** ((2 - p) / 2) * (x ** 2 + eps ** 2) ** ((p - 2) / 4)


@dataclass
class Problem:
    G: np.ndarray                  # (n_data, n_cells)
    d: np.ndarray                  # observed data
    sd: np.ndarray                 # data standard deviations
    Wz: np.ndarray                 # depth/sensitivity weights (n_cells,)
    D: list                        # first-difference operators
    alpha_s: float = 1e-3
    B: sp.spmatrix = None          # graph incidence (n_edges x n_cells)
    w_graph: np.ndarray = None     # edge weights
    graph_ratio: float = 0.0       # lambda / beta
    arch_idx: np.ndarray = None    # cells with archive constraints
    arch_val: np.ndarray = None
    arch_sd: np.ndarray = None

    def __post_init__(self):
        n = self.G.shape[1]
        Wz = sp.diags(self.Wz)
        self.Rs0 = (np.sqrt(self.alpha_s) * Wz).tocsr()
        if self.B is not None and self.graph_ratio > 0:
            # the fabric graph replaces isotropic first-difference smoothing
            self.Rr0 = (sp.diags(np.sqrt(self.graph_ratio * self.w_graph)) @ self.B @ Wz).tocsr()
        else:
            self.Rr0 = sp.vstack([Di @ Wz for Di in self.D]).tocsr()
        self.A = None
        if self.arch_idx is not None and len(self.arch_idx):
            k = len(self.arch_idx)
            self.A = sp.csr_matrix((1.0 / self.arch_sd, (np.arange(k), self.arch_idx)), shape=(k, n))
        self.set_weights(None, None)

    def set_weights(self, ws, wr):
        """IRLS row weights for the smallness and smoothness blocks (None = l2)."""
        Rs = self.Rs0 if ws is None else sp.diags(ws) @ self.Rs0
        Rr = self.Rr0 if wr is None else sp.diags(wr) @ self.Rr0
        self.R = sp.vstack([Rs, Rr]).tocsr()

    def operator(self, beta):
        Gw = self.G / self.sd[:, None]
        sb = np.sqrt(beta)
        rows = [Gw.shape[0], self.R.shape[0], self.A.shape[0] if self.A is not None else 0]
        n = self.G.shape[1]
        cuts = np.cumsum([0] + rows)

        def mv(m):
            out = [Gw @ m, sb * (self.R @ m)]
            if self.A is not None:
                out.append(self.A @ m)
            return np.concatenate(out)

        def rmv(r):
            x = Gw.T @ r[cuts[0]:cuts[1]] + sb * (self.R.T @ r[cuts[1]:cuts[2]])
            if self.A is not None:
                x += self.A.T @ r[cuts[2]:cuts[3]]
            return x

        K = LinearOperator((cuts[-1], n), matvec=mv, rmatvec=rmv, dtype=float)
        rhs = np.zeros(cuts[-1])
        rhs[cuts[0]:cuts[1]] = self.d / self.sd
        if self.A is not None:
            rhs[cuts[2]:cuts[3]] = self.arch_val / self.arch_sd
        return K, rhs

    def misfit(self, m):
        return float(np.sum(((self.G @ m - self.d) / self.sd) ** 2))


def solve_map(prob, beta, rhs_noise=None, x0=None, iters=2000, tol=1e-7):
    K, rhs = prob.operator(beta)
    if rhs_noise is not None:
        rhs = rhs + rhs_noise
    return lsqr(K, rhs, atol=tol, btol=tol, iter_lim=iters, x0=x0)[0]


def choose_beta(prob, target=None, lo=1e-2, hi=1e7, n_iter=14):
    """Discrepancy principle: find beta with chi^2 = target (default n_data) by bisection in log."""
    target = target or len(prob.d)
    a, b = np.log10(lo), np.log10(hi)
    m = None
    for _ in range(n_iter):
        c = 0.5 * (a + b)
        m = solve_map(prob, 10 ** c, x0=m)
        if prob.misfit(m) > target:
            b = c
        else:
            a = c
    beta = 10 ** (0.5 * (a + b))
    return beta, solve_map(prob, beta, x0=m)


def solve_irls(prob, beta, m0, p_s=2.0, p_r=2.0, n_iter=20, eps_frac=0.1, target=None):
    """Iteratively reweighted least squares for lp smallness (p_s) and smoothness (p_r).

    Starts from the l2 model m0. The thresholds eps are set once from m0 (eps_frac times the
    95th percentile of each block's residual). beta is rescaled every iteration towards
    chi^2 = target, so all norms are compared at the same data misfit.
    """
    target = target or len(prob.d)
    m = m0.copy()
    eps_s = eps_frac * np.percentile(np.abs(prob.Rs0 @ m0), 95)
    eps_r = eps_frac * np.percentile(np.abs(prob.Rr0 @ m0), 95)
    history = []
    for _ in range(n_iter):
        ws = None if p_s == 2 else lp_weights(prob.Rs0 @ m, p_s, eps_s)
        wr = None if p_r == 2 else lp_weights(prob.Rr0 @ m, p_r, eps_r)
        prob.set_weights(ws, wr)
        m = solve_map(prob, beta, x0=m)
        phi = prob.misfit(m)
        history.append((beta, phi))
        beta *= float(np.clip(target / phi, 0.5, 2.0))
    # final solve at a beta that honours the target misfit with the frozen weights
    beta, m = choose_beta(prob, target, lo=beta / 30, hi=beta * 30, n_iter=8)
    return beta, m, history


def sample_posterior(prob, beta, m_map, n_samples=40, seed=0):
    rng = np.random.default_rng(seed)
    K, rhs = prob.operator(beta)
    return np.array([solve_map(prob, beta, rhs_noise=rng.standard_normal(len(rhs)), x0=m_map)
                     for _ in range(n_samples)])
