"""Field inversions, version 2 (paper Section 3.5): robustness, validation and coupling.

Katanga (Cu-Co endowment)
  * main runs: XGM2019e at 300 m above the terrain with the Nettleton density (A, B, G);
    cross-checks with EIGEN-6C4, with continuation to 6 km, and with 2670 kg/m3 (B);
  * choice ensemble: 24 draws over gravity model, observation height, reduction density,
    regional filter, theory and fabric, 10 RTO samples each;
  * every run stores per-sample column masses by depth slice, so that sub-districts and
    backgrounds can be evaluated afterwards (src/experiments/katanga_validation.py).
Cuvette Centrale (depth to basement)
  * spherical gravity kernel from tesseroids (Earth curvature over a 1000 km window);
  * main runs (A, B), EIGEN-6C4 cross-check, prism-kernel comparison, choice ensemble;
  * cross-gradient joint inversion of gravity and EMAG2v3 magnetics (drcinv.joint).

Run:  python src/experiments/field_v2.py katanga|cuvette [--quick]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "experiments"))
from drcinv import (TensorMesh, Problem, choose_beta, fabric_graph,  # noqa: E402
                    fabric_normal, gravity_kernel, sample_posterior)
from drcinv.forward import magnetic_kernel, tesseroid_kernel  # noqa: E402
from drcinv.inversion import sensitivity_weights, solve_irls  # noqa: E402
from drcinv.joint import joint_invert, gradient_ops_interior, cross_gradient  # noqa: E402
from field_inversions import (CASES, KM_PER_DEG, PROC, RHO_TOPO, SD_MGAL,  # noqa: E402
                              bouguer_residual, build, magnetic_depths, nettleton, project, residual)

OUT = ROOT / "results" / "field_v2"
CACHE = OUT / "cache"
MODELS = ["xgm2019e", "eigen6c4"]
CONFIGS = {"A": dict(graph=False, p_s=2, p_r=2, alpha_s=1e-3),
           "B": dict(graph=True, p_s=2, p_r=2, alpha_s=1e-3),
           "G": dict(graph=True, p_s=0, p_r=1, alpha_s=1.0)}
IGRF_CUVETTE = dict(field_nT=33186.0, inc_deg=-29.3, dec_deg=-0.3)   # IGRF-13, 2010, 1S 21E, 4 km
SD_NT = 5.0


def gravity_data(name, model, h_obs=None):
    tag = f"{name}_{model}_{'auto' if h_obs is None else int(h_obs)}"
    path = CACHE / f"{tag}.nc"
    if path.exists():
        return xr.load_dataset(path)
    ds = bouguer_residual(CASES[name], model, h_obs)
    ds.to_netcdf(path)
    return ds


def invert_one(G, d, mesh, cfg, strike, dip, n_s, seed, graph_cache, sd=SD_MGAL):
    sdv = np.full(len(d), sd)
    Wz = sensitivity_weights(G, sdv, 0.5)
    key = (round(strike, 1), round(dip, 1))
    if cfg["graph"] and key not in graph_cache:
        graph_cache[key] = fabric_graph(mesh, fabric_normal(strike, dip))
    B, wg = graph_cache.get(key, (None, None))
    prob = Problem(G=G, d=d, sd=sdv, Wz=Wz, D=list(mesh.gradient_ops()), alpha_s=cfg["alpha_s"],
                   B=B, w_graph=wg, graph_ratio=4.0 if cfg["graph"] else 0.0)
    beta, m = choose_beta(prob)
    if cfg["p_s"] != 2 or cfg["p_r"] != 2:
        beta, m, _ = solve_irls(prob, beta, m, p_s=cfg["p_s"], p_r=cfg["p_r"])
    S = sample_posterior(prob, beta, m, n_samples=n_s, seed=seed)
    return prob, beta, m, S


# ================================================================================ Katanga
KAT_SLICES = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 5000), (5000, 8000)]


def slice_masses(S, mesh):
    """Per-sample column mass (Mt per cell column) in each depth slice: (n_s, n_slices, nx, ny)."""
    g = S.reshape(len(S), mesh.nx, mesh.ny, mesh.nz)
    depth = (np.arange(mesh.nz) + 0.5) * mesh.dz
    out = [g[..., (depth >= z1) & (depth < z2)].sum(-1) for z1, z2 in KAT_SLICES]
    return (np.stack(out, 1) * mesh.cell_volume * 1000 / 1e9).astype(np.float32)


def katanga(quick=False, name="katanga"):
    """Copperbelt endowment runs for name = katanga or zambia (same design)."""
    case = CASES[name]
    rd = OUT / f"{name}_runs"
    rd.mkdir(parents=True, exist_ok=True)
    G_cache, graph_cache, rho_n = {}, {}, {}
    n_main, n_ens_s, n_ens = (5, 3, 3) if quick else (40, 10, 24)

    def setup(model, h):
        ds = gravity_data(name, model, h)
        key = (model, h)
        if key not in rho_n:
            rho_n[key] = nettleton(ds, case, case["regional_km"])[0]
            print(f"  {model} h={ds.attrs['h_obs']:.0f} m: Nettleton {rho_n[key]:.0f} kg/m3", flush=True)
        return ds

    def run(tag, model, h, rho, sigma, cfg, strike, dip, n_s, seed):
        path = rd / f"{tag}.npz"
        if path.exists():
            return json.loads(str(np.load(path)["meta"]))
        t0 = time.time()
        ds = setup(model, h)
        rho = rho_n[(model, h)] if rho == "nettleton" else rho
        res = residual(ds, sigma, rho)
        mesh, obs, d, (lon, lat) = build(case, ds, res)
        hk = float(ds.attrs["h_obs"])
        if hk not in G_cache:
            G_cache[hk] = gravity_kernel(mesh, obs)
        prob, beta, m, S = invert_one(G_cache[hk], d, mesh, CONFIGS[cfg], strike, dip, n_s, seed, graph_cache)
        meta = dict(tag=tag, model=model, h_obs=hk, rho=float(rho), sigma_km=float(sigma), config=cfg,
                    strike=float(strike), dip=float(dip), n_samples=n_s, beta=float(beta),
                    chi2=prob.misfit(m) / len(d), resid_sd=float(d.std()), seconds=time.time() - t0)
        np.savez_compressed(path, meta=json.dumps(meta), slices=slice_masses(S, mesh),
                            map=m.astype(np.float32), d=d.astype(np.float32), lon=lon, lat=lat,
                            mesh=np.array([mesh.x0, mesh.y0, mesh.z_top, mesh.dx, mesh.dy, mesh.dz,
                                           mesh.nx, mesh.ny, mesh.nz]),
                            lon0=ds.attrs["lon0"], lat0=ds.attrs["lat0"])
        print(f"  {tag:22s} {model} h={hk:.0f} rho={rho:.0f} sig={sigma:.0f} {cfg} "
              f"strike={strike:.0f} dip={dip:.0f}  chi2/N={meta['chi2']:.2f} ({meta['seconds']:.0f} s)",
              flush=True)
        return meta

    metas = []
    st, dp = case["strike"], case["dip"]
    print(f"{name} main runs", flush=True)
    for cfg in "ABG":
        metas.append(run(f"main_{cfg}", "xgm2019e", None, "nettleton", 150.0, cfg, st, dp, n_main, 1))
    metas.append(run("eigen_B", "eigen6c4", None, "nettleton", 150.0, "B", st, dp, n_main, 2))
    metas.append(run("h6_B", "xgm2019e", 6000.0, "nettleton", 150.0, "B", st, dp, n_main, 3))
    metas.append(run("h6_B_2670", "xgm2019e", 6000.0, RHO_TOPO, 150.0, "B", st, dp, n_main, 4))
    metas.append(run("rho2670_B", "xgm2019e", None, RHO_TOPO, 150.0, "B", st, dp, n_main, 5))
    print(f"{name} choice ensemble", flush=True)
    rng = np.random.default_rng(2026)
    for i in range(n_ens):
        model = MODELS[rng.integers(2)]
        h = [None, 4000.0, 6000.0][rng.integers(3)]
        setup(model, h)
        lo, hi = sorted([rho_n[(model, h)], RHO_TOPO])
        rho = float(rng.uniform(lo, hi))
        sigma = float(rng.uniform(100, 300))
        cfg = "AB"[rng.integers(2)]
        strike, dip = float(rng.normal(st, 15)), float(rng.uniform(15, 60))
        metas.append(run(f"ens_{i:02d}", model, h, rho, sigma, cfg, strike, dip, n_ens_s, 100 + i))
    nett = {f"{m}_{'auto' if h is None else int(h)}": v for (m, h), v in rho_n.items()}
    (OUT / f"{name}_runs.json").write_text(json.dumps({"runs": metas, "nettleton": nett}, indent=1))


# ================================================================================ Cuvette
def cuvette_mesh(ds, case):
    """Lon/lat cells of 0.18 x 8/44 degrees (~20 km) and 1.5 km thickness down to 15 km.
    Returns the edges (for tesseroids) and an equivalent metric TensorMesh (graph, weights)."""
    w, e, s, n = case["window"]
    lon_e = np.linspace(w, e, 51)
    lat_e = np.linspace(s, n, 45)
    dep_e = np.arange(0, 15001, 1500.0)
    lon0, lat0 = ds.attrs["lon0"], ds.attrs["lat0"]
    x0, y0 = project(w, s, lon0, lat0)
    dx = (lon_e[1] - lon_e[0]) * KM_PER_DEG * 1000 * np.cos(np.radians(lat0))
    dy = (lat_e[1] - lat_e[0]) * KM_PER_DEG * 1000
    z_top = float(ds.topography.sel(longitude=slice(w, e), latitude=slice(s, n)).mean())
    mesh = TensorMesh(x0, y0, z_top, dx, dy, 1500.0, 50, 44, 10)
    return mesh, lon_e, lat_e, dep_e


def thickness(S, mesh, drho):
    col = S.reshape(len(S), mesh.nx, mesh.ny, mesh.nz).sum(3) * mesh.dz / 1000
    return (col / drho[:, None, None]).astype(np.float32)


def cuvette(quick=False):
    case = CASES["cuvette"]
    rd = OUT / "cuvette_runs"
    rd.mkdir(parents=True, exist_ok=True)
    n_main, n_ens_s, n_ens = (5, 3, 3) if quick else (40, 10, 20)
    graph_cache = {}
    ds0 = gravity_data("cuvette", "xgm2019e")
    mesh, lon_e, lat_e, dep_e = cuvette_mesh(ds0, case)
    _, obs_p, d0, (lon, lat) = build(case, ds0, residual(ds0, 150.0))
    LON, LAT = np.meshgrid(lon, lat)
    obs_ll = np.column_stack([LON.ravel(), LAT.ravel()])
    obs_m = np.column_stack([*project(LON.ravel(), LAT.ravel(), ds0.attrs["lon0"], ds0.attrs["lat0"]),
                             np.full(LON.size, ds0.attrs["h_obs"])])
    Gp = gravity_kernel(mesh, obs_m)
    gpath = CACHE / "cuvette_Gtess.npy"
    if quick:
        Gt = Gp                                   # smoke test: skip the 2-hour tesseroid kernel
    elif gpath.exists():
        Gt = np.load(gpath).astype(np.float64)
    else:
        t0 = time.time()
        print(f"Cuvette: tesseroid kernel {len(obs_ll)} x {mesh.n}", flush=True)
        Gt = tesseroid_kernel(lon_e, lat_e, dep_e, obs_ll, ds0.attrs["h_obs"], z_top=mesh.z_top)
        np.save(gpath, Gt)
        Gt = Gt.astype(np.float64)
        print(f"  done in {(time.time() - t0) / 60:.0f} min", flush=True)

    def run(tag, model, sigma, cfg, n_s, seed, kernel="tess"):
        path = rd / f"{tag}.npz"
        if path.exists():
            return json.loads(str(np.load(path)["meta"]))
        t0 = time.time()
        ds = gravity_data("cuvette", model)
        _, _, d, _ = build(case, ds, residual(ds, sigma))
        G = Gt if kernel == "tess" else Gp
        prob, beta, m, S = invert_one(G, d, mesh, CONFIGS[cfg], 0.0, 0.0, n_s, seed, graph_cache)
        drho = np.clip(np.random.default_rng(seed + 7).normal(-0.20, 0.04, n_s), -0.35, -0.10)
        meta = dict(tag=tag, model=model, sigma_km=float(sigma), config=cfg, kernel=kernel, n_samples=n_s,
                    beta=float(beta), chi2=prob.misfit(m) / len(d), resid_sd=float(d.std()),
                    seconds=time.time() - t0)
        np.savez_compressed(path, meta=json.dumps(meta), thick=thickness(S, mesh, drho), drho=drho,
                            map=m.astype(np.float32), d=d.astype(np.float32), lon=lon, lat=lat,
                            lon_e=lon_e, lat_e=lat_e,
                            mesh=np.array([mesh.x0, mesh.y0, mesh.z_top, mesh.dx, mesh.dy, mesh.dz,
                                           mesh.nx, mesh.ny, mesh.nz]))
        print(f"  {tag:14s} {model} sig={sigma:.0f} {cfg} {kernel} chi2/N={meta['chi2']:.2f} "
              f"({meta['seconds']:.0f} s)", flush=True)
        return meta

    metas = []
    print("Cuvette main runs", flush=True)
    metas.append(run("main_B", "xgm2019e", 150.0, "B", n_main, 1))
    metas.append(run("main_A", "xgm2019e", 150.0, "A", n_main, 2))
    metas.append(run("eigen_B", "eigen6c4", 150.0, "B", n_main, 3))
    metas.append(run("prism_B", "xgm2019e", 150.0, "B", n_main, 4, kernel="prism"))
    print("Cuvette choice ensemble", flush=True)
    rng = np.random.default_rng(2027)
    for i in range(n_ens):
        metas.append(run(f"ens_{i:02d}", MODELS[rng.integers(2)], float(rng.uniform(100, 300)),
                         "AB"[rng.integers(2)], n_ens_s, 100 + i))
    (OUT / "cuvette_runs.json").write_text(json.dumps({"runs": metas}, indent=1))

    # ---------------------------------------------------------------- joint gravity-magnetics
    jpath = OUT / "cuvette_joint.npz"
    if jpath.exists():
        return
    print("Cuvette joint gravity-magnetic inversion", flush=True)
    mag = xr.load_dataset(PROC / "emag2_drc.nc").magnetic_anomaly.rename(lon="longitude", lat="latitude")
    w, e, s, n = case["window"]
    win = mag.sel(longitude=slice(w - 3, e + 3), latitude=slice(s - 3, n + 3))
    step = float(win.longitude[1] - win.longitude[0])
    vals = win.values
    filled = np.where(np.isfinite(vals), vals, np.nanmean(vals))
    reg = gaussian_filter(filled, 150.0 / (step * KM_PER_DEG), mode="reflect")
    mres = xr.DataArray(vals - reg, coords=win.coords).interp(longitude=lon, latitude=lat).values.ravel()
    ok = np.isfinite(mres)
    obs_mag = np.column_stack([obs_m[ok, 0], obs_m[ok, 1], np.full(ok.sum(), 4000.0)])
    mpath = CACHE / "cuvette_Gmag.npy"
    if mpath.exists():
        Gm = np.load(mpath)
    else:
        t0 = time.time()
        Gm = magnetic_kernel(mesh, obs_mag, **IGRF_CUVETTE)
        np.save(mpath, Gm)
        print(f"  magnetic kernel {Gm.shape} in {(time.time() - t0) / 60:.1f} min", flush=True)
    dm = mres[ok]
    ds = gravity_data("cuvette", "xgm2019e")
    _, _, dg, _ = build(case, ds, residual(ds, 150.0))
    pg, bg, mg, _ = invert_one(Gt, dg, mesh, CONFIGS["B"], 0.0, 0.0, 1, 0, graph_cache)
    pm, bm, mk, _ = invert_one(Gm, dm, mesh, CONFIGS["B"], 0.0, 0.0, 1, 0, graph_cache, sd=SD_NT)
    tau0 = cross_gradient(gradient_ops_interior(mesh), mg / mg.std(), mk / mk.std())[0]
    print(f"  separate: chi2 grav {pg.misfit(mg) / len(dg):.2f}, mag {pm.misfit(mk) / len(dm):.2f}", flush=True)
    jg, jk, hist = joint_invert(pg, pm, bg, bm, mg.copy(), mk.copy(), mesh, n_iter=3 if quick else 8)
    np.savez_compressed(jpath, sep_g=mg, sep_k=mk, joint_g=jg, joint_k=jk, dm=dm, dg=dg,
                        hist=json.dumps(hist), tau0=float(np.sqrt(tau0 @ tau0)),
                        pred_m=pm.G @ jk, pred_g=pg.G @ jg, obs_ok=ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("area", choices=["katanga", "zambia", "cuvette"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    global OUT, CACHE
    if args.quick:
        OUT = OUT.parent / "field_v2_quick"
        CACHE = OUT.parent / "field_v2" / "cache"
    for p in (OUT, CACHE):
        p.mkdir(parents=True, exist_ok=True)
    if args.area == "cuvette":
        cuvette(args.quick)
    else:
        katanga(args.quick, args.area)
    print("done", args.area, flush=True)


if __name__ == "__main__":
    main()
