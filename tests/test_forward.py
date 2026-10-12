import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drcinv import TensorMesh, gravity_kernel  # noqa: E402
from drcinv.forward import GAMMA, SI_TO_MGAL  # noqa: E402


def test_far_field_matches_point_mass():
    # one 100 m cube centred at 1 km depth, density contrast 1 g/cc
    mesh = TensorMesh(-50, -50, -950, 100, 100, 100, 1, 1, 1)
    obs = np.array([[0.0, 0.0, 0.0], [800.0, 300.0, 0.0], [3000.0, 0.0, 0.0]])
    g = gravity_kernel(mesh, obs)[:, 0]
    mass = 1000.0 * 100.0 ** 3
    dz = 1000.0
    r = np.sqrt(obs[:, 0] ** 2 + obs[:, 1] ** 2 + dz ** 2)
    g_point = GAMMA * mass * dz / r ** 3 * SI_TO_MGAL
    assert np.all(g > 0), g
    np.testing.assert_allclose(g, g_point, rtol=2e-3)


def test_infinite_slab_limit():
    # wide 100 m slab: Bouguer g = 2 pi G rho t = 0.0419 mGal per (g/cc * m) * 100 m
    mesh = TensorMesh(-100000, -100000, -500, 2000, 2000, 100, 100, 100, 1)
    g = gravity_kernel(mesh, np.array([[0.0, 0.0, 0.0]])).sum()
    np.testing.assert_allclose(g, 2 * np.pi * GAMMA * 1000 * 100 * SI_TO_MGAL, rtol=1e-2)


if __name__ == "__main__":
    test_far_field_matches_point_mass()
    test_infinite_slab_limit()
    print("forward tests passed")
