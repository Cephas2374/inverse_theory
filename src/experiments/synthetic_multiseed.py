"""Robustness of the synthetic benchmark (paper Section 3.3): several noise realisations and
three ore-body geometries.

Geometries (all with the granite cupola and the shallow basin of the main benchmark):
    baseline  400 m thick package, top at 0.5 km, dip 20 degrees (as in the main benchmark)
    steep     400 m thick package, top at 0.5 km, dip 45 degrees
    thin-deep 200 m thick package, top at 1.0 km, dip 20 degrees
Each (geometry, seed) pair draws new data noise and new archive errors. All seven theories
are run at 0.5 km spacing and A, D, G at 1, 2 and 4 km; metrics use the MAP models and the
mass-balance tonnage of the MAP model (posterior sampling is not needed for the ranking).
Outputs: results/synthetic_multiseed/<geometry>_<seed>.json, summary.json and a figure.

Run:  python src/experiments/synthetic_multiseed.py --geometry baseline --seeds 0-7
      python src/experiments/synthetic_multiseed.py --summarise
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "experiments"))
from drcinv import TensorMesh, fabric_graph, fabric_normal  # noqa: E402
import synthetic_copperbelt as S  # noqa: E402

OUT = ROOT / "results" / "synthetic_multiseed"
GEOMETRIES = {"baseline": dict(dip=20.0, thick=400.0, top=500.0, top_max=1500.0),
              "steep": dict(dip=45.0, thick=400.0, top=500.0, top_max=2200.0),
              "thin-deep": dict(dip=20.0, thick=200.0, top=1000.0, top_max=2000.0)}
SPACINGS = {500: "ABCDEFG", 1000: "ADG", 2000: "ADG", 4000: "ADG"}


def build_truth(mesh, g):
    X, Y, Z = mesh.centers()
    depth = -Z
    m = np.zeros(mesh.n)
    top = g["top"] + (X - 3000) * np.tan(np.radians(g["dip"]))
    ore = ((X >= 3000) & (depth >= top) & (depth <= top + g["thick"]) & (top <= g["top_max"])
           & (Y >= 2000) & (Y <= 10000))
    m[ore] = S.ORE_DRHO
    m[(X - 9000) ** 2 + (Y - 8000) ** 2 + (Z + 2300) ** 2 <= 1500 ** 2] = -0.12
    m[(X >= 7000) & (Y <= 3600) & (depth <= 600)] = -0.20
    return m, ore


def one(geom, seed):
    path = OUT / f"{geom}_{seed}.json"
    if path.exists():
        return
    g = GEOMETRIES[geom]
    mesh = TensorMesh(0, 0, 0, 400, 400, 200, 30, 30, 15)
    truth, ore = build_truth(mesh, g)
    depth = -mesh.centers()[2]
    rng = np.random.default_rng(1000 + seed)
    graph = fabric_graph(mesh, fabric_normal(S.STRIKE, g["dip"]))
    archive = S.archive_records(mesh, truth, rng)
    t_true = float(S.tonnage(truth[None], mesh, depth)[0].sum())
    rec = {"geometry": geom, "seed": seed, "true_tonnage_Mt": t_true, "runs": {}}
    t0 = time.time()
    for spacing, configs in SPACINGS.items():
        _, res = S.run(spacing, configs, mesh, truth, ore, graph, archive, 0, rng, noise_seed=10_000 + seed)
        for k, r in res.items():
            t = float(S.tonnage(r["map"][None], mesh, depth)[0].sum())
            rec["runs"][f"{spacing}_{k}"] = dict(spacing_m=spacing, config=k, **r["metrics"],
                                                 tonnage_ratio=t / t_true, seconds=r["seconds"])
    rec["seconds"] = time.time() - t0
    path.write_text(json.dumps(rec, indent=1))
    print(f"done {geom} seed {seed} in {rec['seconds'] / 60:.1f} min", flush=True)


def summarise():
    recs = [json.loads(p.read_text()) for p in sorted(OUT.glob("*_*.json")) if p.stem != "summary"]
    out = {}
    for geom in GEOMETRIES:
        rs = [r for r in recs if r["geometry"] == geom]
        if not rs:
            continue
        keys = sorted({k for r in rs for k in r["runs"]}, key=lambda k: (int(k.split("_")[0]), k))
        tab = {}
        for k in keys:
            v = [r["runs"][k] for r in rs if k in r["runs"]]
            tab[k] = {m: [float(np.mean([x[m] for x in v])), float(np.std([x[m] for x in v]))]
                      for m in ("centroid_depth_error_m", "ore_iou", "rmse", "tonnage_ratio", "chi2_per_datum")}
            tab[k]["n"] = len(v)
        # rank stability at 0.5 km: how often each theory has the best IoU / smallest |centroid error|
        best_iou, best_dz = {}, {}
        for r in rs:
            sub = {k.split("_")[1]: x for k, x in r["runs"].items() if k.startswith("500_")}
            bi = max(sub, key=lambda c: sub[c]["ore_iou"])
            bz = min(sub, key=lambda c: abs(sub[c]["centroid_depth_error_m"]))
            best_iou[bi] = best_iou.get(bi, 0) + 1
            best_dz[bz] = best_dz.get(bz, 0) + 1
        out[geom] = dict(n_seeds=len(rs), metrics=tab, best_iou_count=best_iou, best_centroid_count=best_dz)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    return out


NAMES = {"A": "Smooth (Tikhonov)", "B": "+ Graph", "C": "+ Archive", "D": "Full $\\ell_2$ (graph + archive)",
         "E": "Compact", "F": "Total variation", "G": "Sparse graph"}
GLAB = {"baseline": "Baseline (20$^\\circ$, 400 m, top 0.5 km)", "steep": "Steep (45$^\\circ$, 400 m, top 0.5 km)",
        "thin-deep": "Thin and deep (20$^\\circ$, 200 m, top 1 km)"}


def outputs(s):
    """Robustness table (all theories at 0.5 km) and figure (tonnage vs spacing for A, D, G)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gen = ROOT / "paper" / "sections" / "generated"
    fig_dir = ROOT / "paper" / "figures"
    rows = []
    for geom, v in s.items():
        m = v["metrics"]
        best_i = max("ABCDEFG", key=lambda c: m[f"500_{c}"]["ore_iou"][0])
        best_z = min("ABCDEFG", key=lambda c: abs(m[f"500_{c}"]["centroid_depth_error_m"][0]))
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{GLAB[geom]}}}}} \\\\")
        for c in "ABCDEFG":
            r = m[f"500_{c}"]
            dz = r["centroid_depth_error_m"]
            iou = r["ore_iou"]
            t4 = m.get(f"4000_{c}", {}).get("tonnage_ratio")
            pm = r"\pm"
            dz_s = f"${dz[0]:+.0f}{pm}{dz[1]:.0f}$"
            iou_s = f"${iou[0]:.2f}{pm}{iou[1]:.2f}$"
            if c == best_z:
                dz_s = r"\textbf{" + dz_s + "}"
            if c == best_i:
                iou_s = r"\textbf{" + iou_s + "}"
            t4_s = f"${t4[0]:.2f}$" if t4 else "--"
            rows.append(f"\\quad {c} & {NAMES[c]} & {dz_s} & {iou_s} & ${r['tonnage_ratio'][0]:.2f}$ & {t4_s} \\\\")
    n = min(v["n_seeds"] for v in s.values())
    (gen / "tab_synthetic_robustness.tex").write_text(
        "\\begin{table*}[t]\n\\centering\\small\n\\caption{Robustness of the synthetic benchmark: mean $\\pm$ standard "
        f"deviation over {n} noise and archive realisations for three ore-body geometries, at \\SI{{0.5}}{{km}} spacing. "
        "$\\Delta z_c$: depth-centroid error; IoU: overlap with the true package; $T/T_0$: mass-balance tonnage of the "
        "MAP model over the truth, at 0.5 and \\SI{4}{km} spacing. Bold: best value in each geometry.}"
        "\\label{tab:synrobust}\n"
        "\\begin{tabular*}{\\tblwidth}{@{\\extracolsep{\\fill}}llrrrr@{}}\n\\toprule\n"
        " & Theory & $\\Delta z_c$ (m) & IoU & $T/T_0$ (0.5 km) & $T/T_0$ (4 km) \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table*}\n", encoding="utf-8")
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
    col = {"A": "#9a9893", "D": "#2a78d6", "G": "#eb6834"}
    fig, axs = plt.subplots(2, 3, figsize=(190 / 25.4, 105 / 25.4), sharex=True)
    sp = [500, 1000, 2000, 4000]
    for j, (geom, v) in enumerate(s.items()):
        m = v["metrics"]
        for c in "ADG":
            t = np.array([m[f"{x}_{c}"]["tonnage_ratio"] for x in sp])
            i = np.array([m[f"{x}_{c}"]["ore_iou"] for x in sp])
            axs[0, j].errorbar(np.array(sp) / 1000, t[:, 0], yerr=t[:, 1], marker="o", ms=3.5, color=col[c],
                               label=NAMES[c])
            axs[1, j].errorbar(np.array(sp) / 1000, i[:, 0], yerr=i[:, 1], marker="o", ms=3.5, color=col[c])
        axs[0, j].axhline(1, color="#52514e", lw=0.6, ls="--")
        axs[0, j].set_title(f"({'abc'[j]}) {GLAB[geom].split(' (')[0]}", loc="left")
        axs[1, j].set_title(f"({'def'[j]}) overlap", loc="left")
        axs[1, j].set_xscale("log")
        axs[1, j].set_xticks(np.array(sp) / 1000, ["0.5", "1", "2", "4"])
        axs[1, j].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axs[1, j].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        axs[1, j].set_xlabel("Station spacing (km)")
        for a in axs[:, j]:
            a.grid(color="#d9d8d4", lw=0.5)
            for s_ in ("top", "right"):
                a.spines[s_].set_visible(False)
    axs[0, 0].set_ylabel("Recovered / true tonnage")
    axs[1, 0].set_ylabel("IoU")
    axs[0, 0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(fig_dir / f"syn_robustness.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=list(GEOMETRIES))
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.summarise:
        s = summarise()
        outputs(s)
        print(json.dumps({g: {k: v[k] for k in ("n_seeds", "best_iou_count", "best_centroid_count")}
                          for g, v in s.items()}, indent=1))
        return
    lo, hi = map(int, a.seeds.split("-"))
    for seed in range(lo, hi + 1):
        one(a.geometry, seed)


if __name__ == "__main__":
    main()
