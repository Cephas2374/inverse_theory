"""Field inversions with open data (paper Section 3.5).

(1) Katanga Copperbelt: Cu-Co endowment beneath the occurrence district.
(2) Cuvette Centrale: sediment thickness (depth to basement) of the Congo Basin.

Gravity: XGM2019e to degree 2159 (~9 km), synthesised on the ellipsoid
(src/analysis/synth_gravity.py), upward-continued to a flat observation surface above the
highest terrain, and terrain-corrected with 3D prisms built from the 3-arc-minute global
relief (density 2670 kg/m3). The regional field is a Gaussian low-pass of the complete
Bouguer anomaly; the residual is inverted on a local flat-earth tensor mesh.

Methods (paper Section 2): Tikhonov (A), the l2 fabric-graph model (B), and in Katanga the
sparse-graph theory (G) for comparison. The archive term is absent because no digitised
drill records exist yet for these areas. beta follows the discrepancy principle and 40
randomise-then-optimise samples give the posterior.

Run:  python src/experiments/field_inversions.py [--quick] [--only katanga|cuvette]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import harmonica as hm
import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor and the raster backend)
import xarray as xr
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))
from drcinv import (TensorMesh, Problem, choose_beta, fabric_graph,  # noqa: E402
                    fabric_normal, gravity_kernel, sample_posterior)
from drcinv.inversion import sensitivity_weights, solve_irls  # noqa: E402
from regional_analysis import radial_spectrum, two_segment_fit  # noqa: E402

RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT = ROOT / "results" / "field"
KM_PER_DEG = 111.195
RHO_TOPO = 2670.0      # kg/m3, terrain correction
SD_MGAL = 2.0          # XGM2019e data error over sparsely surveyed land (assumed)

CASES = {
    "katanga": dict(
        window=(24.6, 28.4, -12.4, -9.8),     # inversion window (lon, lon, lat, lat)
        pad=3.0, step=0.1, dec=1,             # gravity grid step (deg), station decimation
        dx=5000.0, dz=1000.0, nz=8,           # mesh
        regional_km=150.0,
        configs={"A": dict(name="Tikhonov", graph=False, p_s=2, p_r=2, alpha_s=1e-3),
                 "B": dict(name="l2 fabric graph", graph=True, p_s=2, p_r=2, alpha_s=1e-3),
                 "G": dict(name="Sparse graph", graph=True, p_s=0, p_r=1, alpha_s=1.0)},
        strike=120.0, dip=30.0,               # Lufilian grain, thrusts dipping SSW
        slices=[(0, 1000), (1000, 2000), (2000, 3000), (3000, 5000), (5000, 8000)],
    ),
    "cuvette": dict(
        window=(16.5, 25.5, -5.0, 3.0),
        pad=3.0, step=0.1, dec=2,
        dx=20000.0, dz=1500.0, nz=10,
        regional_km=150.0,
        configs={"A": dict(name="Tikhonov", graph=False, p_s=2, p_r=2, alpha_s=1e-3),
                 "B": dict(name="l2 fabric graph", graph=True, p_s=2, p_r=2, alpha_s=1e-3)},
        strike=0.0, dip=0.0,                  # horizontal sedimentary layering
        slices=[(0, 3000), (3000, 6000), (6000, 9000), (9000, 15000)],
        regional_sensitivity=[100.0, 300.0],
    ),
}


# ------------------------------------------------------------------------------ data
def relief(w, e, s, n):
    """3-arc-minute global relief (m). The JPEG2000 tiles store elevation / 2."""
    tiles = []
    for name in ("S90E000", "N00E000"):
        r = rioxarray.open_rasterio(RAW / "relief" / f"{name}.earth_relief_03m_g.jp2").squeeze(drop=True)
        r = r.sortby("y").sel(x=slice(w, e), y=slice(s, n))
        if r.size:
            tiles.append(r.astype(float) * 0.5)
    r = xr.concat(tiles, "y").sortby("y")
    r = r.isel(y=np.unique(r.y.values, return_index=True)[1])
    return r.rename(x="longitude", y="latitude")


def project(lon, lat, lon0, lat0):
    k = KM_PER_DEG * 1000
    return (np.asarray(lon) - lon0) * k * np.cos(np.radians(lat0)), (np.asarray(lat) - lat0) * k


def bouguer_residual(case, model="xgm2019e", h_obs=None):
    """Complete Bouguer anomaly (mGal) on a padded grid and its residual after removing a
    Gaussian regional. model: 'xgm2019e' or 'eigen6c4'; h_obs (m) defaults to 300 m above
    the highest terrain of the window. Returns the grids and the observation height."""
    w, e, s, n = case["window"]
    pad, step = case["pad"], case["step"]
    W, E, S, N = w - pad, e + pad, max(s - pad, -13.95), min(n + pad, 5.95)
    lon0, lat0 = 0.5 * (w + e), 0.5 * (s + n)
    xgm = xr.load_dataset(PROC / f"{model}_drc.nc").gravity_disturbance
    lon = np.arange(W, E + 1e-9, step)
    lat = np.arange(S, N + 1e-9, step)
    g0 = xgm.interp(longitude=lon, latitude=lat)
    topo = relief(W - 1, E + 1, S - 1, N + 1)
    t_win = topo.sel(longitude=slice(w, e), latitude=slice(s, n))
    if h_obs is None:
        h_obs = float(np.ceil((t_win.max() + 300) / 100) * 100)
    # upward continuation from the ellipsoid (h = 0) to h_obs on a local Cartesian grid
    ex, ny_ = project(lon, lat0, lon0, lat0)[0], project(lon0, lat, lon0, lat0)[1]
    grid = xr.DataArray(g0.values, coords={"northing": ny_, "easting": ex}, dims=("northing", "easting"))
    padded = grid.pad(northing=20, easting=20, mode="reflect")
    padded = padded.assign_coords(
        easting=ex[0] + (np.arange(padded.easting.size) - 20) * (ex[1] - ex[0]),
        northing=ny_[0] + (np.arange(padded.northing.size) - 20) * (ny_[1] - ny_[0]))
    up = hm.upward_continuation(padded, height_displacement=h_obs)
    g_up = up.isel(easting=slice(20, -20), northing=slice(20, -20)).values
    # terrain: prisms from the ellipsoid to the relief surface (sea/lakes ignored: inland)
    tl = topo.sel(longitude=slice(W - 1, E + 1), latitude=slice(S - 1, N + 1))
    px, py = project(tl.longitude.values, tl.latitude.values[:, None], lon0, lat0)
    px = np.broadcast_to(px, tl.shape)
    py = np.broadcast_to(py, tl.shape)
    dxp = (tl.longitude.values[1] - tl.longitude.values[0]) * KM_PER_DEG * 1000 * np.cos(np.radians(lat0))
    dyp = (tl.latitude.values[1] - tl.latitude.values[0]) * KM_PER_DEG * 1000
    h = np.clip(tl.values, 0, None)
    prisms = np.column_stack([(px - dxp / 2).ravel(), (px + dxp / 2).ravel(),
                              (py - dyp / 2).ravel(), (py + dyp / 2).ravel(),
                              np.zeros(h.size), h.ravel()])
    keep = prisms[:, 5] > 0
    LON, LAT = np.meshgrid(lon, lat)
    sx, sy = project(LON, LAT, lon0, lat0)
    coords = (sx.ravel(), sy.ravel(), np.full(sx.size, h_obs))
    g_topo = hm.prism_gravity(coords, prisms[keep], np.full(keep.sum(), RHO_TOPO),
                              field="g_z", parallel=True).reshape(sx.shape)
    bouguer = g_up - g_topo
    out = xr.Dataset({"bouguer": (("latitude", "longitude"), bouguer),
                      "disturbance_up": (("latitude", "longitude"), g_up),
                      "terrain": (("latitude", "longitude"), g_topo),
                      "topography": (("latitude", "longitude"),
                                     topo.interp(longitude=lon, latitude=lat).values)},
                     coords={"longitude": lon, "latitude": lat})
    out.attrs.update(h_obs=h_obs, lon0=lon0, lat0=lat0, model=model)
    return out


def nettleton(ds, case, sigma_km, rhos=np.arange(1800, 2851, 50)):
    """Reduction density that removes the correlation between the high-passed Bouguer
    anomaly and the high-passed terrain inside the window (Nettleton, 1939)."""
    w, e, s, n = case["window"]
    sig = sigma_km / (float(ds.longitude[1] - ds.longitude[0]) * KM_PER_DEG)
    hp = lambda a: a - gaussian_filter(a, sig, mode="reflect")   # noqa: E731
    m = (((ds.longitude >= w) & (ds.longitude <= e)).values[None, :]
         & ((ds.latitude >= s) & (ds.latitude <= n)).values[:, None])
    th = hp(ds.topography.values)[m]
    corr = [float(np.corrcoef(hp(ds.disturbance_up.values - ds.terrain.values * r / RHO_TOPO)[m], th)[0, 1])
            for r in rhos]
    return float(rhos[int(np.argmin(np.abs(corr)))]), dict(zip(map(int, rhos), corr))


def residual(ds, sigma_km, rho=RHO_TOPO):
    b = ds.disturbance_up.values - ds.terrain.values * rho / RHO_TOPO
    sig = sigma_km / (float(ds.longitude[1] - ds.longitude[0]) * KM_PER_DEG)
    return b - gaussian_filter(b, sig, mode="reflect")


# ------------------------------------------------------------------------------ inversion
def build(case, ds, res):
    w, e, s, n = case["window"]
    lon0, lat0, h_obs = ds.attrs["lon0"], ds.attrs["lat0"], ds.attrs["h_obs"]
    sel = ((ds.longitude >= w) & (ds.longitude <= e)).values
    sel_y = ((ds.latitude >= s) & (ds.latitude <= n)).values
    lon = ds.longitude.values[sel][::case["dec"]]
    lat = ds.latitude.values[sel_y][::case["dec"]]
    r = res[np.ix_(sel_y, sel)][::case["dec"], ::case["dec"]]
    LON, LAT = np.meshgrid(lon, lat)
    ox, oy = project(LON, LAT, lon0, lat0)
    obs = np.column_stack([ox.ravel(), oy.ravel(), np.full(ox.size, h_obs)])
    x0, y0 = project(w, s, lon0, lat0)
    x1, y1 = project(e, n, lon0, lat0)
    nx, ny = int(round((x1 - x0) / case["dx"])), int(round((y1 - y0) / case["dx"]))
    t = ds.topography.sel(longitude=slice(w, e), latitude=slice(s, n))
    z_top = float(t.mean())
    mesh = TensorMesh(x0, y0, z_top, case["dx"], case["dx"], case["dz"], nx, ny, case["nz"])
    return mesh, obs, r.ravel(), (lon, lat)


def invert(case, mesh, obs, d, n_samples, keys, seed=0):
    t0 = time.time()
    G = gravity_kernel(mesh, obs)
    sd = np.full(len(d), SD_MGAL)
    Wz = sensitivity_weights(G, sd, 0.5)
    D = list(mesh.gradient_ops())
    B, wg = fabric_graph(mesh, fabric_normal(case["strike"], case["dip"]))
    print(f"  G {G.shape}, kernel+graph {time.time() - t0:.0f} s", flush=True)
    rng = np.random.default_rng(seed)
    out = {}
    for k in keys:
        c = case["configs"][k]
        t1 = time.time()
        prob = Problem(G=G, d=d, sd=sd, Wz=Wz, D=D, alpha_s=c["alpha_s"], B=B, w_graph=wg,
                       graph_ratio=4.0 if c["graph"] else 0.0)
        beta, m = choose_beta(prob)
        if c["p_s"] != 2 or c["p_r"] != 2:
            beta, m, _ = solve_irls(prob, beta, m, p_s=c["p_s"], p_r=c["p_r"])
        samples = sample_posterior(prob, beta, m, n_samples=n_samples, seed=int(rng.integers(1e9)))
        out[k] = dict(beta=beta, map=m, samples=samples, chi2=prob.misfit(m) / len(d),
                      pred=G @ m, seconds=time.time() - t1)
        print(f"  {k} {c['name']:16s} beta={beta:.3g} chi2/N={out[k]['chi2']:.2f} "
              f"({out[k]['seconds']:.0f} s)", flush=True)
    return out


def cell_lonlat(mesh, lon0, lat0):
    X, Y, Z = mesh.centers()
    k = KM_PER_DEG * 1000
    return X / (k * np.cos(np.radians(lat0))) + lon0, Y / k + lat0, Z


# ------------------------------------------------------------------------------ Katanga
def katanga(args):
    case = CASES["katanga"]
    print("Katanga: gravity processing", flush=True)
    ds = bouguer_residual(case)
    rho_n, corr_n = nettleton(ds, case, case["regional_km"])
    print(f"  Nettleton reduction density {rho_n:.0f} kg/m3", flush=True)
    runs, data = {}, {}
    # main runs at the Nettleton density; sensitivity: l2 fabric graph at 2670 kg/m3
    for rho, keys in [(rho_n, "AB" if args.quick else "ABG"), (RHO_TOPO, "B")]:
        res = residual(ds, case["regional_km"], rho)
        mesh, obs, d, (lon, lat) = build(case, ds, res)
        print(f"  rho {rho:.0f}: mesh {mesh.shape} ({mesh.n} cells), {len(d)} stations, "
              f"h_obs {ds.attrs['h_obs']:.0f} m, residual sd {d.std():.1f} mGal", flush=True)
        for k, r in invert(case, mesh, obs, d, 5 if args.quick else 40, keys).items():
            runs[f"{k}_{int(rho)}"] = dict(r, config=k, rho=rho)
        data[int(rho)] = d
    d = data[int(rho_n)]

    # sectors: cells within 10 km of a Cu-Co occurrence vs the rest of Haut-Katanga/Lualaba
    from scipy.spatial import cKDTree
    from shapely import contains_xy
    dep = gpd.read_file(PROC / "deposits.gpkg")
    cu = dep[dep.group == "Cu-Co"]
    lon0, lat0 = ds.attrs["lon0"], ds.attrs["lat0"]
    clon, clat, Z = cell_lonlat(mesh, lon0, lat0)
    cx, cy = project(cu.lon.values, cu.lat.values, lon0, lat0)
    X, Y, _ = mesh.centers()
    dist, _ = cKDTree(np.column_stack([cx, cy])).query(np.column_stack([X, Y]))
    adm1 = gpd.read_file(RAW / "boundaries" / "COD_ADM1.geojson").to_crs(4326)
    prov = adm1[adm1.shapeName.isin(["Haut-Katanga", "Lualaba"])].union_all()
    in_prov = contains_xy(prov, clon, clat)
    district = in_prov & (dist <= 10_000)
    backgr = in_prov & (dist > 30_000)
    depth = mesh.z_top - Z
    col_area = mesh.dx * mesh.dy / 1e6   # km2
    summary = {"h_obs_m": ds.attrs["h_obs"], "z_top_m": mesh.z_top, "n_stations": len(d),
               "mesh": list(mesh.shape), "cell_m": [mesh.dx, mesh.dy, mesh.dz],
               "residual_sd_mgal": float(d.std()), "residual_range_mgal": [float(d.min()), float(d.max())],
               "n_cu_sites": int(len(cu)), "district_area_km2": float(district.sum() / mesh.nz * col_area),
               "background_area_km2": float(backgr.sum() / mesh.nz * col_area),
               "nettleton_density": rho_n, "nettleton_corr": corr_n,
               "residual_sd_mgal_2670": float(data[int(RHO_TOPO)].std()), "runs": {}}
    # topography leakage check: residual Bouguer vs elevation at the stations
    LON, LAT = np.meshgrid(lon, lat)
    elev = ds.topography.interp(longitude=lon, latitude=lat).values.ravel()
    summary["corr_residual_elevation"] = float(np.corrcoef(d, elev)[0, 1])
    for k, r in runs.items():
        S = r["samples"]
        rows = {}
        for z1, z2 in case["slices"]:
            sl = (depth >= z1) & (depth < z2)
            m_d = S[:, district & sl].sum(1) * mesh.cell_volume * 1000 / 1e12          # Gt
            m_b = S[:, backgr & sl].sum(1) * mesh.cell_volume * 1000 / 1e12
            a_d = district.sum() / mesh.nz * col_area
            a_b = backgr.sum() / mesh.nz * col_area
            rows[f"{z1}-{z2}"] = {
                "district_Gt": np.percentile(m_d, [10, 50, 90]).tolist(),
                "district_Mt_per_km2": np.percentile(m_d / a_d * 1e3, [10, 50, 90]).tolist(),
                "background_Mt_per_km2": np.percentile(m_b / a_b * 1e3, [10, 50, 90]).tolist()}
        tot_d = S[:, district & (depth < 3000)].sum(1) * mesh.cell_volume * 1000 / 1e12
        tot_b = S[:, backgr & (depth < 3000)].sum(1) * mesh.cell_volume * 1000 / 1e12
        a_d = district.sum() / mesh.nz * col_area
        a_b = backgr.sum() / mesh.nz * col_area
        excess = tot_d - tot_b / a_b * a_d     # district excess over the background density, Gt
        drho, rho0 = 0.35, 2.67                 # ore-equivalent conversion as in Eq. (tonnage)
        summary["runs"][k] = {
            "name": case["configs"][r["config"]]["name"], "config": r["config"], "rho_reduction": r["rho"],
            "beta": r["beta"], "chi2_per_datum": r["chi2"],
            "seconds": r["seconds"], "slices": rows,
            "district_0_3km_Gt": np.percentile(tot_d, [10, 50, 90]).tolist(),
            "excess_over_background_0_3km_Gt": np.percentile(excess, [10, 50, 90]).tolist(),
            "ore_equivalent_upper_bound_Gt": np.percentile(excess * (rho0 + drho) / drho, [10, 50, 90]).tolist(),
            "peak_drho": float(np.abs(r["map"]).max()),
            "corr_shallow_mass_elevation": float(np.corrcoef(
                mesh.to_grid(r["map"])[:, :, :2].sum(2).T.ravel(),
                ds.topography.interp(longitude=np.unique(np.round(clon, 6)),
                                     latitude=np.unique(np.round(clat, 6))).values.ravel())[0, 1]),
        }
    np.savez_compressed(OUT / "katanga.npz", d=d, d_2670=data[int(RHO_TOPO)], rho_n=rho_n,
                        lon=lon, lat=lat, obs=obs, district=district,
                        backgr=backgr, mesh=np.array([mesh.x0, mesh.y0, mesh.z_top, mesh.dx, mesh.dy,
                                                      mesh.dz, mesh.nx, mesh.ny, mesh.nz]),
                        lon0=lon0, lat0=lat0,
                        **{f"map_{k}": r["map"] for k, r in runs.items()},
                        **{f"sd_{k}": r["samples"].std(0) for k, r in runs.items()},
                        **{f"pred_{k}": r["pred"] for k, r in runs.items()})
    ds.to_netcdf(OUT / "katanga_gravity.nc")
    return summary


# ------------------------------------------------------------------------------ Cuvette
def magnetic_depths(window, size=3.0, stride=1.0):
    """Local Spector-Grant depths (two-segment fit) in sliding windows of EMAG2v3."""
    mag = xr.load_dataset(PROC / "emag2_drc.nc").magnetic_anomaly.rename(lon="longitude", lat="latitude")
    topo = relief(*[window[0] - 3, window[1] + 3, window[2] - 3, window[3] + 3])
    w, e, s, n = window
    step = float(mag.longitude[1] - mag.longitude[0])
    rows = []
    for cy in np.arange(s + size / 2, n - size / 2 + 1e-9, stride):
        for cx in np.arange(w + size / 2, e - size / 2 + 1e-9, stride):
            win = mag.sel(longitude=slice(cx - size / 2, cx + size / 2), latitude=slice(cy - size / 2, cy + size / 2))
            cov = float(np.isfinite(win.values).mean())
            if cov < 0.9:
                continue
            k, lnp = radial_spectrum(win.values, step * KM_PER_DEG)
            try:
                f = two_segment_fit(k, lnp, kmin=0.03)
            except (ValueError, TypeError):
                continue
            elev = float(topo.sel(longitude=slice(cx - size / 2, cx + size / 2),
                                  latitude=slice(cy - size / 2, cy + size / 2)).mean()) / 1000
            clear = 4.0 - elev
            rows.append({"lon": float(cx), "lat": float(cy), "coverage": cov,
                         "shallow_km": f["shallow_depth_km"] - clear, "shallow_se_km": f["shallow_se_km"],
                         "deep_km": f["deep_depth_km"] - clear, "deep_se_km": f["deep_se_km"],
                         "k_break": f["k_break"]})
    return rows


def cuvette(args):
    case = CASES["cuvette"]
    print("Cuvette: gravity processing", flush=True)
    ds = bouguer_residual(case)
    lon0, lat0 = ds.attrs["lon0"], ds.attrs["lat0"]
    rho_n, corr_n = nettleton(ds, case, case["regional_km"])
    summary = {"h_obs_m": ds.attrs["h_obs"], "nettleton_density": rho_n, "nettleton_corr": corr_n,
               "runs": {}, "regional": {}}
    n_s = 5 if args.quick else 40
    rng = np.random.default_rng(11)
    store = {}
    for sigma in [case["regional_km"]] + ([] if args.quick else case["regional_sensitivity"]):
        res = residual(ds, sigma)
        mesh, obs, d, (lon, lat) = build(case, ds, res)
        print(f"  regional sigma {sigma:.0f} km: mesh {mesh.shape}, {len(d)} stations, "
              f"residual sd {d.std():.1f} mGal", flush=True)
        keys = "AB" if sigma == case["regional_km"] else "B"
        runs = invert(case, mesh, obs, d, n_s, keys)
        dz = mesh.dz
        # equivalent sediment thickness variation: column deficit / sediment contrast,
        # with the contrast drawn from N(-0.20, 0.04) g/cc for each posterior sample
        drho = np.clip(rng.normal(-0.20, 0.04, n_s), -0.35, -0.10)
        for k, r in runs.items():
            col = r["samples"].reshape(n_s, mesh.nx, mesh.ny, mesh.nz).sum(3) * dz / 1000   # g/cc * km
            thick = col / drho[:, None, None]                                              # km
            col_map = mesh.to_grid(r["map"]).sum(2) * dz / 1000 / -0.20
            tag = f"{k}_{int(sigma)}"
            store[tag] = dict(map=r["map"], thick_map=col_map, thick_p10=np.percentile(thick, 10, 0),
                              thick_p50=np.percentile(thick, 50, 0), thick_p90=np.percentile(thick, 90, 0),
                              pred=r["pred"], d=d)
            summary["runs"][tag] = {"name": case["configs"][k]["name"], "regional_km": sigma,
                                    "beta": r["beta"], "chi2_per_datum": r["chi2"], "seconds": r["seconds"],
                                    "thick_var_range_km": [float(np.percentile(col_map, 2)),
                                                           float(np.percentile(col_map, 98))],
                                    "thick_halfwidth_p10_p90_km_median": float(np.median(
                                        (np.percentile(thick, 90, 0) - np.percentile(thick, 10, 0)) / 2)),
                                    "peak_drho": [float(r["map"].min()), float(r["map"].max())]}
        summary["regional"][int(sigma)] = {"residual_sd_mgal": float(d.std()),
                                           "residual_range_mgal": [float(d.min()), float(d.max())]}
        if sigma == case["regional_km"]:
            base_mesh, base_lonlat = mesh, (lon, lat)
    # magnetic depths and comparison with the gravity thickness variations
    mags = magnetic_depths(case["window"])
    clon, clat, _ = cell_lonlat(base_mesh, lon0, lat0)
    glon = np.unique(np.round(clon, 6))
    glat = np.unique(np.round(clat, 6))
    tm = store[f"B_{int(case['regional_km'])}"]["thick_map"]       # (nx, ny)
    for r in mags:
        sel_x = np.abs(glon - r["lon"]) <= 1.5
        sel_y = np.abs(glat - r["lat"]) <= 1.5
        r["grav_thick_var_km"] = float(tm[np.ix_(sel_x, sel_y)].mean())
    good = [r for r in mags if 0 < r["shallow_km"] < 20 and r["shallow_se_km"] < 2.0]
    if len(good) > 3:
        a = np.array([r["grav_thick_var_km"] for r in good])
        b = np.array([r["shallow_km"] for r in good])
        summary["mag_vs_grav"] = {"n_windows": len(good), "n_all": len(mags),
                                  "pearson_r": float(np.corrcoef(a, b)[0, 1]),
                                  "spearman_r": float(__import__("scipy.stats").stats.spearmanr(a, b)[0]),
                                  "mag_depth_median_km": float(np.median(b)),
                                  "mag_depth_p10_p90_km": np.percentile(b, [10, 90]).tolist()}
    summary["mag_windows"] = mags
    np.savez_compressed(OUT / "cuvette.npz", lon0=lon0, lat0=lat0, glon=glon, glat=glat,
                        lon=base_lonlat[0], lat=base_lonlat[1],
                        mesh=np.array([base_mesh.x0, base_mesh.y0, base_mesh.z_top, base_mesh.dx,
                                       base_mesh.dy, base_mesh.dz, base_mesh.nx, base_mesh.ny, base_mesh.nz]),
                        **{f"{k}__{tag}": v for tag, s in store.items() for k, v in s.items()})
    ds.to_netcdf(OUT / "cuvette_gravity.nc")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["katanga", "cuvette"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    if args.only in (None, "katanga"):
        summary["katanga"] = katanga(args)
        path.write_text(json.dumps(summary, indent=1))
    if args.only in (None, "cuvette"):
        summary["cuvette"] = cuvette(args)
        path.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1)[:6000])


if __name__ == "__main__":
    main()
