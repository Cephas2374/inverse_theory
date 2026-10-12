"""Minimal regular 3D tensor mesh (x east, y north, z up; z <= 0 is below surface)."""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class TensorMesh:
    x0: float
    y0: float
    z_top: float
    dx: float
    dy: float
    dz: float
    nx: int
    ny: int
    nz: int

    @property
    def n(self):
        return self.nx * self.ny * self.nz

    @property
    def shape(self):
        # array index order (ix, iy, iz); iz = 0 is the top layer
        return (self.nx, self.ny, self.nz)

    @property
    def cell_volume(self):
        return self.dx * self.dy * self.dz

    def centers(self):
        xc = self.x0 + self.dx * (np.arange(self.nx) + 0.5)
        yc = self.y0 + self.dy * (np.arange(self.ny) + 0.5)
        zc = self.z_top - self.dz * (np.arange(self.nz) + 0.5)
        X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")
        return X.ravel(), Y.ravel(), Z.ravel()

    def bounds(self):
        """Cell bounds (x1, x2, y1, y2, ztop, zbot), each of length n."""
        X, Y, Z = self.centers()
        return (X - self.dx / 2, X + self.dx / 2, Y - self.dy / 2, Y + self.dy / 2,
                Z + self.dz / 2, Z - self.dz / 2)

    def to_grid(self, v):
        return np.asarray(v).reshape(self.shape)

    def index(self, ix, iy, iz):
        return np.ravel_multi_index((ix, iy, iz), self.shape)

    def cell_at(self, x, y, z):
        ix = np.clip(((np.asarray(x) - self.x0) // self.dx).astype(int), 0, self.nx - 1)
        iy = np.clip(((np.asarray(y) - self.y0) // self.dy).astype(int), 0, self.ny - 1)
        iz = np.clip(((self.z_top - np.asarray(z)) // self.dz).astype(int), 0, self.nz - 1)
        return self.index(ix, iy, iz)

    def gradient_ops(self):
        """First-difference operators Dx, Dy, Dz (sparse), scaled to unit cell length."""
        def d1(n):
            return sp.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n))
        Ix, Iy, Iz = (sp.identity(k) for k in self.shape)
        Dx = sp.kron(sp.kron(d1(self.nx), Iy), Iz)
        Dy = sp.kron(sp.kron(Ix, d1(self.ny)), Iz)
        Dz = sp.kron(sp.kron(Ix, Iy), d1(self.nz))
        return Dx.tocsr(), Dy.tocsr(), Dz.tocsr()
