"""Cell graph whose edge weights follow a geological fabric (strike/dip plane).

The graph Laplacian L = B^T W B smooths strongly between cells that lie in the
fabric plane and weakly across it, so structures elongate along strike and dip.
"""
import itertools

import numpy as np
import scipy.sparse as sp


def fabric_normal(strike_deg, dip_deg):
    """Unit normal of a plane (x east, y north, z up). Dip direction = strike + 90 deg."""
    s, d = np.radians(strike_deg), np.radians(dip_deg)
    strike = np.array([np.sin(s), np.cos(s), 0.0])
    dipdir = np.array([np.sin(s + np.pi / 2), np.cos(s + np.pi / 2), 0.0])
    downdip = np.cos(d) * dipdir + np.array([0.0, 0.0, -np.sin(d)])
    n = np.cross(strike, downdip)
    return n / np.linalg.norm(n)


def fabric_graph(mesh, normal, eps=0.02, sigma_deg=10.0, normal_field=None):
    """Weighted incidence matrix B (n_edges x n_cells) and edge weights w.

    26-connectivity. theta = angle between the edge and the fabric plane;
    w_e = (eps + (1 - eps) * exp(-theta^2 / (2 sigma^2))) / |e|, with |e| in cell units.
    sigma is the angular tolerance of the fabric. normal_field: optional (n_cells, 3)
    array of local normals (overrides `normal`).
    """
    nx, ny, nz = mesh.shape
    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    rows, cols, vals, weights = [], [], [], []
    offsets = [o for o in itertools.product((-1, 0, 1), repeat=3) if o > (0, 0, 0)]
    e = 0
    for ox, oy, oz in offsets:
        jx, jy, jz = ix + ox, iy + oy, iz + oz
        ok = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny) & (jz >= 0) & (jz < nz)
        a = mesh.index(ix[ok], iy[ok], iz[ok])
        b = mesh.index(jx[ok], jy[ok], jz[ok])
        # physical edge vector (iz increases downward)
        vec = np.array([ox * mesh.dx, oy * mesh.dy, -oz * mesh.dz])
        length = np.linalg.norm(vec)
        ehat = vec / length
        nrm = normal_field[a] if normal_field is not None else np.broadcast_to(normal, (len(a), 3))
        theta = np.arcsin(np.clip(np.abs(nrm @ ehat), 0, 1))
        kernel = np.exp(-0.5 * (theta / np.radians(sigma_deg)) ** 2)
        w = (eps + (1 - eps) * kernel) / (length / min(mesh.dx, mesh.dy, mesh.dz))
        m = len(a)
        ids = np.arange(e, e + m)
        rows += [ids, ids]
        cols += [a, b]
        vals += [np.ones(m), -np.ones(m)]
        weights.append(w)
        e += m
    B = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(e, mesh.n))
    return B, np.concatenate(weights)
