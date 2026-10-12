"""Synthetic Copperbelt benchmark (paper Section 3.3).

Truth: a dipping mineralised package (Cu-Co analogue), a granite cupola and a shallow
sedimentary basin. Gravity data are simulated at several station spacings and inverted
with seven configurations (see CONFIGS):
    A  smooth        sensitivity-weighted l2 smoothing (Tikhonov)
    B  +graph        + fabric-graph Laplacian (strike/dip from lineaments/archives)
    C  +archive      + historical drill-hole constraints (shallow, noisy, mislocated)
    D  full          graph + archive
    E  compact       minimum-support (p=0) smallness, l2 smoothing, archive
    F  TV            l1 (total-variation-like) smoothing, archive
    G  sparse graph  p=0 smallness + l1 on fabric-graph edges, archive
Outputs: results/synthetic/*.json|npz, paper/figures/syn_*.pdf, paper/sections/generated/*.tex

Run:  python src/experiments/synthetic_copperbelt.py [--quick]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from drcinv import (TensorMesh, Problem, choose_beta, fabric_graph,  # noqa: E402
                    fabric_normal, gravity_kernel, sample_posterior)
from drcinv.inversion import depth_weights, sensitivity_weights, solve_irls  # noqa: E402

OUT = ROOT / "results" / "synthetic"
FIG = ROOT / "paper" / "figures"
GEN = ROOT / "paper" / "sections" / "generated"

RHO_BG = 2.67          # background density, g/cc
ORE_DRHO = 0.35        # density contrast of the mineralised package, g/cc
STRIKE, DIP = 0.0, 20.0
OBS_Z = 50.0           # survey height above ground, m
DEPTH_SLICES = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 3000)]
#   key: name, fabric graph, archive, lp exponents (smallness, smoothness), alpha_s
CONFIGS = {
    "A": dict(name="Smooth", graph=False, archive=False, p_s=2, p_r=2, alpha_s=1e-3),
    "B": dict(name="+ Graph", graph=True, archive=False, p_s=2, p_r=2, alpha_s=1e-3),
    "C": dict(name="+ Archive", graph=False, archive=True, p_s=2, p_r=2, alpha_s=1e-3),
    "D": dict(name="Full (graph + archive)", graph=True, archive=True, p_s=2, p_r=2, alpha_s=1e-3),
    # alternative regularisation theories, all with the archive prior
    "E": dict(name="Compact (minimum support)", graph=False, archive=True, p_s=0, p_r=2, alpha_s=1.0),
    "F": dict(name="Total variation", graph=False, archive=True, p_s=2, p_r=1, alpha_s=1e-3),
    "G": dict(name="Sparse graph", graph=True, archive=True, p_s=0, p_r=1, alpha_s=1.0),
}
THR = ORE_DRHO / 2     # classification threshold for threshold-based tonnage


def build_truth(mesh):
    X, Y, Z = mesh.centers()
    depth = -Z
    m = np.zeros(mesh.n)
    # dipping package: top at 500 m depth at x = 3 km, deepening east at DIP degrees
    top = 500 + (X - 3000) * np.tan(np.radians(DIP))
    ore = (X >= 3000) & (depth >= top) & (depth <= top + 400) & (top <= 1500) \
        & (Y >= 2000) & (Y <= 10000)
    m[ore] = ORE_DRHO
    cupola = (X - 9000) ** 2 + (Y - 8000) ** 2 + (Z + 2300) ** 2 <= 1500 ** 2
    m[cupola] = -0.12
    basin = (X >= 7000) & (Y <= 3600) & (depth <= 600)
    m[basin] = -0.20
    return m, ore


def stations(spacing, extent=12000):
    c = np.arange(spacing / 2, extent, spacing)
    gx, gy = np.meshgrid(c, c, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, OBS_Z)])


def archive_records(mesh, truth, rng):
    """Historical vertical drill holes: max 1200 m deep, logged ore intervals with
    +-50 m depth error, and collar positions mislocated by sigma = 150 m."""
    collars = [(2600, 8500), (3400, 4000), (4200, 7000), (5000, 3000), (3800, 9500)]
    idx, val, sd = [], [], []
    X, Y, Z = mesh.centers()
    for cx, cy in collars:
        zs = -(np.arange(0, 1200, mesh.dz) + mesh.dz / 2)
        true_cells = mesh.cell_at(np.full(zs.size, cx), np.full(zs.size, cy), zs)
        logged = truth[true_cells] > 0.3
        # depth error: shift the logged intervals by up to one cell
        shift = int(np.round(rng.normal(0, 50) / mesh.dz))
        logged = np.roll(logged, shift)
        px, py = cx + rng.normal(0, 150), cy + rng.normal(0, 150)
        cells = mesh.cell_at(np.full(zs.size, px), np.full(zs.size, py), zs)
        idx.append(cells)
        val.append(np.where(logged, ORE_DRHO, 0.0))
        sd.append(np.full(zs.size, 0.08))
    return np.concatenate(idx), np.concatenate(val), np.concatenate(sd), collars


def prospect_sector(mesh):
    """Cells of the Cu-Co prospect (west of the basin and cupola)."""
    return mesh.centers()[0] < 6500


def tonnage(models, mesh, mask_depth):
    """Mass-balance tonnage (Mt) of mineralised rock per sample and depth slice.

    The signed excess mass in the prospect sector is fixed by the gravity data (Gauss's
    theorem); dividing by the ore density contrast and multiplying by the ore density
    converts it to rock tonnage. The inversion decides how it is split over depth.
    """
    sector = prospect_sector(mesh)
    out = []
    for z1, z2 in DEPTH_SLICES:
        sel = sector & (mask_depth >= z1) & (mask_depth < z2)
        excess_t = models[:, sel].sum(1) * mesh.cell_volume
        out.append(excess_t / ORE_DRHO * (RHO_BG + ORE_DRHO) / 1e6)
    return np.array(out).T   # (n_samples, n_slices)


def metrics(m, truth, ore, mesh, prob):
    _, _, Z = mesh.centers()
    sector = prospect_sector(mesh)
    pos = np.clip(m, 0, None) * sector
    zc_rec = float((pos * -Z).sum() / max(pos.sum(), 1e-12))
    zc_true = float((-Z[ore]).mean())
    # shape recovery: cells above half the sector maximum vs the true package
    pred = sector & (m > 0.5 * pos.max())
    inter, union = (pred & ore).sum(), (pred | ore).sum()
    return {
        "chi2_per_datum": prob.misfit(m) / len(prob.d),
        "rmse": float(np.sqrt(np.mean((m - truth) ** 2))),
        "ore_mean_drho": float(m[ore].mean()),
        "centroid_depth_true_m": zc_true,
        "centroid_depth_rec_m": zc_rec,
        "centroid_depth_error_m": zc_rec - zc_true,
        "ore_iou": float(inter / union) if union else 0.0,
    }


def run(spacing, configs, mesh, truth, ore, graph, archive, n_samples, rng, noise_seed=1,
        weighting="sensitivity", wexp=0.5):
    obs = stations(spacing)
    G = gravity_kernel(mesh, obs)
    d_true = G @ truth
    sd = 0.05 + 0.02 * np.abs(d_true)
    d = d_true + np.random.default_rng(noise_seed).normal(0, sd)
    Wz = (sensitivity_weights(G, sd, wexp) if weighting == "sensitivity"
          else depth_weights(mesh, OBS_Z, exponent=wexp))
    D = list(mesh.gradient_ops())
    B, w = graph
    a_idx, a_val, a_sd, _ = archive
    res = {}
    for key in configs:
        t0 = time.time()
        c = CONFIGS[key]
        prob = Problem(G=G, d=d, sd=sd, Wz=Wz, D=D, alpha_s=c["alpha_s"],
                       B=B, w_graph=w, graph_ratio=4.0 if c["graph"] else 0.0,
                       arch_idx=a_idx if c["archive"] else None,
                       arch_val=a_val if c["archive"] else None,
                       arch_sd=a_sd if c["archive"] else None)
        beta, m = choose_beta(prob)
        if c["p_s"] != 2 or c["p_r"] != 2:
            beta, m, _ = solve_irls(prob, beta, m, p_s=c["p_s"], p_r=c["p_r"])
        samples = sample_posterior(prob, beta, m, n_samples=n_samples, seed=int(rng.integers(1e9)))
        res[key] = {"beta": beta, "map": m, "samples": samples, "d": d,
                    "metrics": metrics(m, truth, ore, mesh, prob),
                    "seconds": time.time() - t0}
        print(f"  spacing {spacing:5.0f} m  {key}  beta={beta:.3g}  "
              f"{res[key]['metrics']}  ({res[key]['seconds']:.0f} s)", flush=True)
    return obs, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="few samples, one spacing (smoke test)")
    ap.add_argument("--configs", default="ABCDEFG")
    ap.add_argument("--sweep-configs", default="ADG")
    ap.add_argument("--weighting", default="sensitivity", choices=["sensitivity", "depth"])
    ap.add_argument("--wexp", type=float, default=0.5)
    args = ap.parse_args()
    for p in (OUT, FIG, GEN):
        p.mkdir(parents=True, exist_ok=True)

    mesh = TensorMesh(0, 0, 0, 400, 400, 200, 30, 30, 15)
    truth, ore = build_truth(mesh)
    _, _, Z = mesh.centers()
    depth = -Z
    rng = np.random.default_rng(42)
    graph = fabric_graph(mesh, fabric_normal(STRIKE, DIP))
    archive = archive_records(mesh, truth, rng)
    n_samples = 5 if args.quick else 40

    t_true = tonnage(truth[None], mesh, depth)[0]
    summary = {"true_tonnage_Mt": t_true.tolist(), "depth_slices_m": DEPTH_SLICES,
               "n_ore_cells": int(ore.sum()), "runs": {}}

    spacings = [500] if args.quick else [500, 1000, 2000, 4000]
    main_obs, main_res = None, None
    for s in spacings:
        configs = args.configs if s == 500 else args.sweep_configs
        obs, res = run(s, configs, mesh, truth, ore, graph, archive, n_samples, rng,
                       weighting=args.weighting, wexp=args.wexp)
        if s == 500:
            main_obs, main_res = obs, res
        for k, r in res.items():
            t = tonnage(r["samples"], mesh, depth)
            sector = prospect_sector(mesh)
            ore_s = (r["samples"] > THR) & sector
            t_thr = (ore_s * (RHO_BG + r["samples"])).sum(1) * mesh.cell_volume / 1e6
            summary["runs"][f"{s}_{k}"] = {
                "spacing_m": s, "config": k, "beta": r["beta"], **r["metrics"],
                "threshold_tonnage_p10_p50_p90": np.percentile(t_thr, [10, 50, 90]).tolist(),
                "peak_drho": float(r["map"][sector].max()),
                "tonnage_p10": np.percentile(t, 10, axis=0).tolist(),
                "tonnage_p50": np.percentile(t, 50, axis=0).tolist(),
                "tonnage_p90": np.percentile(t, 90, axis=0).tolist(),
                "total_tonnage_p10_p50_p90": np.percentile(t.sum(1), [10, 50, 90]).tolist(),
                "seconds": r["seconds"]}

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(OUT / "main_500m.npz", truth=truth, ore=ore, obs=main_obs,
                        d=next(iter(main_res.values()))["d"], collars=np.array(archive[3]),
                        **{f"map_{k}": r["map"] for k, r in main_res.items()},
                        **{f"samples_{k}": r["samples"] for k, r in main_res.items()})
    print("saved", OUT / "summary.json")


if __name__ == "__main__":
    main()
