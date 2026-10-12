"""Joint gravity-magnetic inversion with cross-gradient coupling (Gallardo & Meju, 2003).

Objective (Eq. 3 with both properties):
    Phi = ||Wg (Gg mr - dg)||^2 + ||Wm (Gm mk - dm)||^2
        + beta_g ||R_g mr||^2 + beta_m ||R_m mk||^2 + gamma ||tau(mr, mk)||^2
    tau = grad(mr / s_r) x grad(mk / s_k)
The cross-gradient is linearised about the current models (Gauss-Newton),
    tau(m) ~ tau(m0) + J_r (mr - mr0) + J_k (mk - mk0),
and each iteration solves the stacked least-squares system with LSQR. The trade-off
parameters are rescaled after every iteration towards chi^2 = N for each data set, so the
coupling changes the structure of the models but not their data fit.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, lsqr


def gradient_ops_interior(mesh):
    """Forward-difference gradient (per metre) on the cells that have +x, +y and +z
    neighbours (z downward index). Returns Dx, Dy, Dz of shape (n_int, n)."""
    nx, ny, nz = mesh.shape
    ix, iy, iz = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij")
    ix, iy, iz = ix.ravel(), iy.ravel(), iz.ravel()
    c = mesh.index(ix, iy, iz)
    rows = np.arange(len(c))
    ones = np.ones(len(c))

    def op(nb, h):
        return (sp.csr_matrix((ones, (rows, nb)), shape=(len(c), mesh.n))
                - sp.csr_matrix((ones, (rows, c)), shape=(len(c), mesh.n))) / h
    # +z index is downward: physical dz/dz_up = -(m[iz+1] - m[iz]) / dz
    return (op(mesh.index(ix + 1, iy, iz), mesh.dx), op(mesh.index(ix, iy + 1, iz), mesh.dy),
            -op(mesh.index(ix, iy, iz + 1), mesh.dz))


def cross_gradient(D, a, b):
    """tau = grad a x grad b (3 * n_int) and its Jacobians wrt a and b."""
    Dx, Dy, Dz = D
    ax, ay, az = Dx @ a, Dy @ a, Dz @ a
    bx, by, bz = Dx @ b, Dy @ b, Dz @ b
    tau = np.concatenate([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx])
    d = sp.diags
    Ja = sp.vstack([d(bz) @ Dy - d(by) @ Dz, d(bx) @ Dz - d(bz) @ Dx, d(by) @ Dx - d(bx) @ Dy]).tocsr()
    Jb = sp.vstack([d(ay) @ Dz - d(az) @ Dy, d(az) @ Dx - d(ax) @ Dz, d(ax) @ Dy - d(ay) @ Dx]).tocsr()
    return tau, Ja, Jb


def joint_invert(prob_g, prob_m, beta_g, beta_m, m_g, m_k, mesh, n_iter=8, gamma_frac=1.0,
                 iters=1500, tol=1e-7, verbose=True):
    """Gauss-Newton cross-gradient joint inversion starting from the separate models.

    prob_g, prob_m: drcinv.Problem instances (their R blocks give the regularisation).
    gamma is set so that gamma * ||tau0||^2 = gamma_frac * (N_g + N_m) at the start.
    """
    D = gradient_ops_interior(mesh)
    s_g, s_k = np.std(m_g), np.std(m_k)          # model scales for a unit-free tau
    n = mesh.n
    Ng, Nm = len(prob_g.d), len(prob_m.d)
    Gg = prob_g.G / prob_g.sd[:, None]
    Gm = prob_m.G / prob_m.sd[:, None]
    tau0, _, _ = cross_gradient(D, m_g / s_g, m_k / s_k)
    gamma = gamma_frac * (Ng + Nm) / max(float(tau0 @ tau0), 1e-30)
    history = [dict(it=0, tau=float(np.sqrt(tau0 @ tau0)), chi2_g=prob_g.misfit(m_g) / Ng,
                    chi2_m=prob_m.misfit(m_k) / Nm, beta_g=beta_g, beta_m=beta_m)]
    for it in range(1, n_iter + 1):
        tau, Ja, Jb = cross_gradient(D, m_g / s_g, m_k / s_k)
        Ja, Jb = Ja / s_g, Jb / s_k
        Rg, Rm = prob_g.R, prob_m.R
        sg, sm, sq = np.sqrt(beta_g), np.sqrt(beta_m), np.sqrt(gamma)
        rows = [Ng, Nm, Rg.shape[0], Rm.shape[0], Ja.shape[0]]
        cuts = np.cumsum([0] + rows)

        def mv(x):
            a, b = x[:n], x[n:]
            return np.concatenate([Gg @ a, Gm @ b, sg * (Rg @ a), sm * (Rm @ b), sq * (Ja @ a + Jb @ b)])

        def rmv(r):
            r1, r2, r3, r4, r5 = (r[cuts[i]:cuts[i + 1]] for i in range(5))
            return np.concatenate([Gg.T @ r1 + sg * (Rg.T @ r3) + sq * (Ja.T @ r5),
                                   Gm.T @ r2 + sm * (Rm.T @ r4) + sq * (Jb.T @ r5)])

        K = LinearOperator((cuts[-1], 2 * n), matvec=mv, rmatvec=rmv, dtype=float)
        rhs = np.zeros(cuts[-1])
        rhs[cuts[0]:cuts[1]] = prob_g.d / prob_g.sd
        rhs[cuts[1]:cuts[2]] = prob_m.d / prob_m.sd
        rhs[cuts[4]:cuts[5]] = sq * (Ja @ m_g + Jb @ m_k - tau)
        x = lsqr(K, rhs, atol=tol, btol=tol, iter_lim=iters, x0=np.concatenate([m_g, m_k]))[0]
        m_g, m_k = x[:n], x[n:]
        phi_g, phi_m = prob_g.misfit(m_g), prob_m.misfit(m_k)
        tau_new, _, _ = cross_gradient(D, m_g / s_g, m_k / s_k)
        history.append(dict(it=it, tau=float(np.sqrt(tau_new @ tau_new)), chi2_g=phi_g / Ng,
                            chi2_m=phi_m / Nm, beta_g=beta_g, beta_m=beta_m))
        if verbose:
            print(f"    GN {it}: |tau| {history[-1]['tau']:.3g} (start {history[0]['tau']:.3g}), "
                  f"chi2/N grav {phi_g / Ng:.2f} mag {phi_m / Nm:.2f}", flush=True)
        beta_g *= float(np.clip(Ng / phi_g, 0.5, 2.0))
        beta_m *= float(np.clip(Nm / phi_m, 0.5, 2.0))
    return m_g, m_k, history
