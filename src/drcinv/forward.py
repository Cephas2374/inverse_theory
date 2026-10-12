"""Sensitivity kernels: analytic right-rectangular prisms for gravity (Nagy et al., 2000;
Blakely, 1995) and induced magnetisation, and spherical tesseroids for gravity on
continental-scale windows (Uieda et al., 2016, via Harmonica)."""
import numpy as np

GAMMA = 6.674e-11   # m^3 kg^-1 s^-2
SI_TO_MGAL = 1e5
MU0 = 4e-7 * np.pi
R_EARTH = 6_371_000.0


def gravity_kernel(mesh, obs, chunk=256):
    """Sensitivity G (n_obs x n_cells): g_z in mGal per g/cc density contrast.

    obs: (n_obs, 3) array of x, y, z (z up). g_z is positive for excess mass below.
    """
    x1, x2, y1, y2, ztop, zbot = mesh.bounds()
    obs = np.asarray(obs, float)
    G = np.empty((len(obs), mesh.n))
    for s in range(0, len(obs), chunk):
        o = obs[s:s + chunk]
        xs = (x1[None] - o[:, :1], x2[None] - o[:, :1])
        ys = (y1[None] - o[:, 1:2], y2[None] - o[:, 1:2])
        # depth below the observation point, positive down
        zs = (o[:, 2:3] - ztop[None], o[:, 2:3] - zbot[None])
        acc = 0.0
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                for k, z in enumerate(zs):
                    r = np.sqrt(x * x + y * y + z * z)
                    term = (x * np.log(y + r) + y * np.log(x + r)
                            - z * np.arctan2(x * y, z * r))
                    acc = acc + (-1) ** (i + j + k) * term
        # g/cc -> kg/m^3, m/s^2 -> mGal; sign so that mass below gives g_z > 0
        G[s:s + chunk] = GAMMA * 1000.0 * SI_TO_MGAL * acc
    return G


def magnetic_kernel(mesh, obs, field_nT, inc_deg, dec_deg):
    """Total-field anomaly sensitivity (n_obs x n_cells): nT per SI susceptibility.

    Induced magnetisation only, M = kappa * F / mu0 along the main field; the anomaly is
    projected on the main-field direction. Inclination positive downward, declination
    clockwise from north; x east, y north, z up.
    """
    import harmonica as hm
    inc, dec = np.radians(inc_deg), np.radians(dec_deg)
    t = np.array([np.cos(inc) * np.sin(dec), np.cos(inc) * np.cos(dec), -np.sin(inc)])
    mag = field_nT * 1e-9 / MU0 * t                       # A/m for kappa = 1
    x1, x2, y1, y2, ztop, zbot = mesh.bounds()
    coords = tuple(np.asarray(obs, float).T)
    G = np.empty((len(obs), mesh.n))
    for j in range(mesh.n):
        be, bn, bu = hm.prism_magnetic(coords, [x1[j], x2[j], y1[j], y2[j], zbot[j], ztop[j]],
                                       (mag[0], mag[1], mag[2]), field="b", parallel=False)
        G[:, j] = be * t[0] + bn * t[1] + bu * t[2]
    return G


def tesseroid_kernel(lon_edges, lat_edges, depth_edges, obs_lonlat, h_obs, z_top=0.0):
    """Spherical gravity sensitivity (n_obs x n_cells) in mGal per g/cc for a lon/lat/depth
    grid of tesseroids, cell order (ilon, ilat, iz) as TensorMesh. depth_edges in metres
    below z_top (positive down); heights are relative to a sphere of radius R_EARTH."""
    import harmonica as hm
    coords = (obs_lonlat[:, 0], obs_lonlat[:, 1], np.full(len(obs_lonlat), R_EARTH + h_obs))
    nlon, nlat, nz = len(lon_edges) - 1, len(lat_edges) - 1, len(depth_edges) - 1
    G = np.empty((len(obs_lonlat), nlon * nlat * nz), dtype=np.float32)
    j = 0
    for i in range(nlon):
        for k in range(nlat):
            for m in range(nz):
                tess = [lon_edges[i], lon_edges[i + 1], lat_edges[k], lat_edges[k + 1],
                        R_EARTH + z_top - depth_edges[m + 1], R_EARTH + z_top - depth_edges[m]]
                G[:, j] = hm.tesseroid_gravity(coords, tess, density=1000.0, field="g_z",
                                               parallel=True)
                j += 1
    return G
