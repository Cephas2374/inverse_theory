"""Cuvette Centrale: choice ensemble, curvature effect and joint gravity-magnetic inversion
(paper Section 3.5.2). Inputs: results/field_v2/cuvette_runs/*.npz, cuvette_joint.npz and the
local magnetic spectral depths of results/field/summary.json.
Outputs: results/field_v2/cuvette_analysis.json, figure field_cuvette_v2, table.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from shapely import contains_xy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from drcinv import TensorMesh  # noqa: E402
from drcinv.joint import cross_gradient, gradient_ops_interior  # noqa: E402

QUICK = "--quick" in sys.argv
RES = ROOT / "results" / ("field_v2_quick" if QUICK else "field_v2")
FIG = RES if QUICK else ROOT / "paper" / "figures"
GEN = RES if QUICK else ROOT / "paper" / "sections" / "generated"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42})
MM = 1 / 25.4
SECTION_LAT = -0.5


def mag_windows():
    S = json.loads((ROOT / "results" / "field" / "summary.json").read_text())["cuvette"]
    return [r for r in S["mag_windows"] if 0 < r["shallow_km"] < 20 and r["shallow_se_km"] < 2.0]


def trend(glon, glat, mags, sigma_deg, reach=1.2):
    LON, LAT = np.meshgrid(glon, glat, indexing="ij")
    wl = np.array([[r["lon"], r["lat"]] for r in mags])
    z = np.array([r["shallow_km"] for r in mags])
    d2 = (LON[..., None] - wl[:, 0]) ** 2 + (LAT[..., None] - wl[:, 1]) ** 2
    w = np.exp(-0.5 * d2 / sigma_deg ** 2)
    t = (w * z).sum(-1) / w.sum(-1)
    t[np.sqrt(d2.min(-1)) > reach] = np.nan
    return t


def main():
    for p in (FIG, GEN):
        p.mkdir(parents=True, exist_ok=True)
    runs = {}
    for p in sorted((RES / "cuvette_runs").glob("*.npz")):
        z = np.load(p)
        runs[p.stem] = dict(meta=json.loads(str(z["meta"])), thick=z["thick"], map=z["map"], d=z["d"],
                            mesh=z["mesh"], lon_e=z["lon_e"], lat_e=z["lat_e"])
    r0 = runs["main_B"]
    m = r0["mesh"]
    mesh = TensorMesh(*m[:6], *map(int, m[6:]))
    glon = 0.5 * (r0["lon_e"][1:] + r0["lon_e"][:-1])
    glat = 0.5 * (r0["lat_e"][1:] + r0["lat_e"][:-1])
    LON, LAT = np.meshgrid(glon, glat, indexing="ij")
    adm0 = gpd.read_file(ROOT / "data" / "raw" / "boundaries" / "COD_ADM0.geojson").to_crs(4326)
    inside = contains_xy(adm0.union_all(), LON, LAT)
    mags = mag_windows()
    t1 = trend(glon, glat, mags, 1.0)
    interior = inside & np.isfinite(t1) & (LAT > -3.0)

    def stats_of(Z):
        v = Z[interior]
        return dict(p2_p50_p98=np.percentile(v, [2, 50, 98]).tolist(), frac_ge8=float((v >= 8).mean()),
                    frac_gt4p56=float((v > 4.56).mean()))

    out = {"runs": {}, "n_mag_windows": len(mags), "interior_cells": int(interior.sum())}
    for tag, r in runs.items():
        th = r["thick"]
        out["runs"][tag] = dict(meta=r["meta"], var_p2_p98=np.percentile(np.median(th, 0), [2, 98]).tolist(),
                                halfwidth_median=float(np.median((np.percentile(th, 90, 0)
                                                                  - np.percentile(th, 10, 0))[interior] / 2)),
                                combined=stats_of(t1 + np.median(th, 0)))
    # curvature and gravity-model effects on the thickness variation
    tmap = {t: mesh.to_grid(runs[t]["map"]).sum(2) * mesh.dz / 1000 / -0.20 for t in ("main_B", "prism_B", "eigen_B")}
    dP = tmap["prism_B"] - tmap["main_B"]
    dE = tmap["eigen_B"] - tmap["main_B"]
    out["curvature_effect_km"] = dict(rms=float(np.sqrt(np.mean(dP ** 2))), max=float(np.abs(dP).max()),
                                      corr=float(np.corrcoef(tmap["prism_B"].ravel(), tmap["main_B"].ravel())[0, 1]))
    out["model_effect_km"] = dict(rms=float(np.sqrt(np.mean(dE ** 2))), max=float(np.abs(dE).max()))
    # choice ensemble: thickness samples + magnetic level with a random kernel width
    rng = np.random.default_rng(5)
    ens = [r for t, r in runs.items() if t.startswith("ens_")]
    Z = []
    for r in ens:
        for s in r["thick"]:
            Z.append(trend(glon, glat, mags, rng.uniform(0.7, 1.5)) + s)
    if Z:
        Z = np.array(Z)
        P = np.nanpercentile(Z, [10, 50, 90], 0)
        out["ensemble"] = dict(n_runs=len(ens), n_samples=len(Z), combined_p50=stats_of(P[1]),
                               halfwidth_median=float(np.nanmedian(((P[2] - P[0]) / 2)[interior])),
                               interior_p10_p90_of_median=np.percentile(P[1][interior], [10, 90]).tolist())
    else:
        P = None
    # joint gravity-magnetic inversion
    jp = RES / "cuvette_joint.npz"
    if jp.exists():
        j = np.load(jp)
        hist = json.loads(str(j["hist"]))
        D = gradient_ops_interior(mesh)
        tau_sep = cross_gradient(D, j["sep_g"] / j["sep_g"].std(), j["sep_k"] / j["sep_k"].std())[0]
        tau_j = cross_gradient(D, j["joint_g"] / j["sep_g"].std(), j["joint_k"] / j["sep_k"].std())[0]
        th_sep = mesh.to_grid(j["sep_g"]).sum(2) * mesh.dz / 1000 / -0.20
        th_j = mesh.to_grid(j["joint_g"]).sum(2) * mesh.dz / 1000 / -0.20
        # magnetic source depth per column: susceptibility-weighted mean depth of the positive part
        depth = (np.arange(mesh.nz) + 0.5) * mesh.dz / 1000
        kg = np.clip(mesh.to_grid(j["joint_k"]), 0, None)
        zc = (kg * depth).sum(2) / np.maximum(kg.sum(2), 1e-12)
        mw = [(r["lon"], r["lat"], r["shallow_km"]) for r in mags]
        zc_w = [np.nanmean(np.where((np.abs(LON - lo) <= 1.5) & (np.abs(LAT - la) <= 1.5), zc, np.nan))
                for lo, la, _ in mw]
        out["joint"] = dict(
            iterations=len(hist) - 1, tau_reduction=float(1 - np.linalg.norm(tau_j) / np.linalg.norm(tau_sep)),
            chi2_final=[hist[-1]["chi2_g"], hist[-1]["chi2_m"]],
            corr_rho_kappa_sep=float(np.corrcoef(j["sep_g"], j["sep_k"])[0, 1]),
            corr_rho_kappa_joint=float(np.corrcoef(j["joint_g"], j["joint_k"])[0, 1]),
            thickness_change_rms_km=float(np.sqrt(np.mean((th_j - th_sep) ** 2))),
            thickness_corr_sep_joint=float(np.corrcoef(th_j.ravel(), th_sep.ravel())[0, 1]),
            mag_centroid_vs_spectral_spearman=float(stats.spearmanr(zc_w, [x[2] for x in mw])[0]),
            mag_centroid_median_km=float(np.median(zc[inside])),
            history=hist)
    (RES / "cuvette_analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1)[:3500])
    figure(out, runs, mesh, glon, glat, P, t1, jp)
    table(out)


def extent(glon, glat):
    dl, dt = glon[1] - glon[0], glat[1] - glat[0]
    return (glon[0] - dl / 2, glon[-1] + dl / 2, glat[0] - dt / 2, glat[-1] + dt / 2)


def base(ax, adm0, adm1):
    adm1.boundary.plot(ax=ax, color=INK2, lw=0.3)
    adm0.boundary.plot(ax=ax, color=INK, lw=0.7)
    ax.set_xlim(16.5, 25.5)
    ax.set_ylim(-5, 3)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(17, 26, 2))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}°E")
    ax.yaxis.set_major_formatter(lambda v, _: f"{abs(v):g}°{'S' if v < 0 else 'N' if v > 0 else ''}")


def figure(out, runs, mesh, glon, glat, P, t1, jp):
    adm0 = gpd.read_file(ROOT / "data" / "raw" / "boundaries" / "COD_ADM0.geojson").to_crs(4326)
    adm1 = gpd.read_file(ROOT / "data" / "raw" / "boundaries" / "COD_ADM1.geojson").to_crs(4326)
    fig = plt.figure(figsize=(190 * MM, 160 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.85], hspace=0.38, wspace=0.3)
    hcb = dict(shrink=0.85, pad=0.13, orientation="horizontal", aspect=25)
    ext = extent(glon, glat)
    mags = mag_windows()
    Z50 = P[1] if P is not None else t1 + np.median(runs["main_B"]["thick"], 0)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(Z50.T, origin="lower", extent=ext, cmap="viridis_r", vmin=0, vmax=11, interpolation="nearest")
    base(ax, adm0, adm1)
    ax.scatter([r["lon"] for r in mags], [r["lat"] for r in mags], c=[r["shallow_km"] for r in mags], s=18,
               cmap="viridis_r", vmin=0, vmax=11, edgecolors=INK, linewidths=0.5)
    ax.axhline(SECTION_LAT, color="white", lw=0.7, ls="--")
    ax.set_title("(a) Basement depth, P50", loc="left")
    fig.colorbar(im, ax=ax, label="km below ground", **hcb)
    ax = fig.add_subplot(gs[0, 1])
    hw = ((P[2] - P[0]) / 2) if P is not None else np.zeros_like(Z50)
    im = ax.imshow(hw.T, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=np.nanpercentile(hw, 98),
                   interpolation="nearest")
    base(ax, adm0, adm1)
    ax.set_title("(b) Half-width P10–P90", loc="left")
    fig.colorbar(im, ax=ax, label="km", **hcb)
    ax = fig.add_subplot(gs[0, 2])
    tm = {t: mesh.to_grid(runs[t]["map"]).sum(2) * mesh.dz / 1000 / -0.20 for t in ("main_B", "prism_B")}
    dP = tm["prism_B"] - tm["main_B"]
    lim = np.percentile(np.abs(dP), 99)
    im = ax.imshow(dP.T, origin="lower", extent=ext, cmap="PuOr_r", vmin=-lim, vmax=lim, interpolation="nearest")
    base(ax, adm0, adm1)
    ax.set_title("(c) Flat earth minus sphere", loc="left")
    fig.colorbar(im, ax=ax, label="thickness difference (km)", **hcb)
    if jp.exists():
        j = np.load(jp)
        jy = int(np.argmin(np.abs(glat - SECTION_LAT)))
        depth_max = mesh.nz * mesh.dz / 1000
        for col, key, title in [(slice(0, 2), "joint", "(d) Joint inversion, E–W section at 0.5°S: "
                                 "$\\Delta\\rho$ (colour), $\\kappa$ (contours)")]:
            ax = fig.add_subplot(gs[1, col])
            g = mesh.to_grid(j["joint_g"])[:, jy, :].T
            k = mesh.to_grid(j["joint_k"])[:, jy, :].T
            lim = np.percentile(np.abs(mesh.to_grid(j["joint_g"])), 99)
            im = ax.imshow(g, origin="upper", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                           extent=(glon[0], glon[-1], depth_max, 0), interpolation="nearest")
            kl = np.percentile(np.abs(k), 90)
            dz = (np.arange(mesh.nz) + 0.5) * mesh.dz / 1000
            cs = ax.contour(glon, dz, k, levels=[-kl, -kl / 2, kl / 2, kl], colors=INK, linewidths=0.6)
            ax.clabel(cs, fmt=lambda v: f"{v:.3f}", fontsize=5.5)
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Depth (km)")
            ax.set_title(title, loc="left")
            fig.colorbar(im, ax=ax, shrink=0.9, pad=0.015).set_label("g/cm$^3$", labelpad=1)
        ax = fig.add_subplot(gs[1, 2])
        h = json.loads(str(j["hist"]))
        it = [x["it"] for x in h]
        ax.plot(it, np.array([x["tau"] for x in h]) / h[0]["tau"], "o-", color="#2a78d6", ms=3,
                label="|cross-gradient|")
        ax.plot(it, [x["chi2_g"] for x in h], "s-", color="#9a9893", ms=3, label="$\\chi^2/N$ gravity")
        ax.plot(it, [x["chi2_m"] for x in h], "^-", color="#eb6834", ms=3, label="$\\chi^2/N$ magnetics")
        ax.set_xlabel("Gauss–Newton iteration")
        ax.set_title("(e) Joint inversion convergence", loc="left")
        ax.legend(frameon=False)
        ax.grid(color=GRID, lw=0.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"field_cuvette_v2.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def table(out):
    lab = {"main_B": "$\\ell_2$ fabric graph, XGM2019e", "main_A": "Tikhonov, XGM2019e",
           "eigen_B": "$\\ell_2$ fabric graph, EIGEN-6C4", "prism_B": "$\\ell_2$ fabric graph, flat earth"}
    rows = []
    for t in ["main_B", "main_A", "eigen_B", "prism_B"]:
        r = out["runs"][t]
        c = r["combined"]
        rows.append(f"{lab[t]} & {r['meta']['chi2']:.2f} & ${r['var_p2_p98'][0]:.1f}$ to $+{r['var_p2_p98'][1]:.1f}$ & "
                    f"{r['halfwidth_median']:.1f} & {c['p2_p50_p98'][1]:.1f} ({c['p2_p50_p98'][0]:.1f}--{c['p2_p50_p98'][2]:.1f}) & "
                    f"{100 * c['frac_ge8']:.0f} \\\\")
    if "ensemble" in out:
        e = out["ensemble"]
        c = e["combined_p50"]
        rows.append(f"Choice ensemble ({e['n_runs']} runs) & -- & -- & {e['halfwidth_median']:.1f} & "
                    f"{c['p2_p50_p98'][1]:.1f} ({c['p2_p50_p98'][0]:.1f}--{c['p2_p50_p98'][2]:.1f}) & {100 * c['frac_ge8']:.0f} \\\\")
    (GEN / "tab_field_cuvette.tex").write_text(
        "\\begin{table*}[t]\n\\centering\\small\n\\caption{Cuvette Centrale basement inversion with the spherical "
        "(tesseroid) kernel unless stated. Misfit, 2nd--98th percentile range of the gravity thickness variation, "
        "median P10--P90 half-width, and the combined depth to basement in the basin interior north of 3$^\\circ$S "
        "(median with 2nd--98th percentile) with the share deeper than 8~km. The ensemble varies gravity model, "
        "regional filter (100--300~km), theory, sediment contrast and magnetic kernel width.}\\label{tab:fieldcuvette}\n"
        "\\begin{tabular*}{\\tblwidth}{@{\\extracolsep{\\fill}}lrlrlr@{}}\n\\toprule\n"
        "Run & $\\chi^2/N$ & Variation (km) & $\\pm$ (km) & Basement depth (km) & $\\geq$8~km (\\%) \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table*}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
