"""Maps, figures and LaTeX tables for the national data analysis (paper Section 3.4)."""
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results" / "regional"
FIG = ROOT / "paper" / "figures"
GEN = ROOT / "paper" / "sections" / "generated"

# validated categorical order (dataviz reference palette); identity follows the commodity
GCOL = {"Cu-Co": "#2a78d6", "Au": "#eb6834", "3T": "#1baf7a", "Diamond": "#c98500", "Other": "#9a9893"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42,
})
MM = 1 / 25.4
DOUBLE, SINGLE = 190 * MM, 90 * MM
EXTENT = (12.0, 31.4, -13.6, 5.4)

ds = xr.load_dataset(PROC / "drc_grids.nc")
adm0 = gpd.read_file(ROOT / "data" / "raw" / "boundaries" / "COD_ADM0.geojson").to_crs(4326)
adm1 = gpd.read_file(ROOT / "data" / "raw" / "boundaries" / "COD_ADM1.geojson").to_crs(4326)
dep = gpd.read_file(RES / "deposits_districts.gpkg")
spec = json.loads((RES / "spectral.json").read_text())
cv = pd.read_csv(RES / "prospectivity_cv.csv")
sig = pd.read_csv(RES / "signatures.csv")
dist = pd.read_csv(RES / "districts.csv")
gstats = json.loads((RES / "graph_stats.json").read_text())
pros = xr.load_dataset(RES / "prospectivity.nc")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def base(ax, extent=EXTENT, provinces=True):
    if provinces:
        adm1.boundary.plot(ax=ax, color=INK2, lw=0.3)
    adm0.boundary.plot(ax=ax, color=INK, lw=0.7)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.tick_params(length=2)
    ax.set_xticks(np.arange(np.ceil(extent[0] / 4) * 4, extent[1] + 0.1, 4))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}°E")
    ax.yaxis.set_major_formatter(lambda v, _: f"{abs(v):g}°{'S' if v < 0 else 'N' if v > 0 else ''}")


def masked(var):
    return ds[var].where(ds.inside)


def add_residual_bouguer():
    """Bouguer minus a Gaussian-filtered regional (sigma = 150 km). The continental Bouguer
    field is negative everywhere (isostatic roots), which hides crustal structure."""
    from scipy.ndimage import gaussian_filter
    b = ds["bouguer"].values
    filled = np.where(np.isfinite(b), b, np.nanmean(b))
    sigma = 150 / (0.05 * 111.195)
    ds["bouguer_res"] = (("latitude", "longitude"), b - gaussian_filter(filled, sigma, mode="nearest"))


def show(ax, da, cmap, vmin=None, vmax=None):
    return ax.imshow(da.values, origin="lower", extent=(ds.longitude.min(), ds.longitude.max(),
                                                       ds.latitude.min(), ds.latitude.max()),
                     cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")


# ----------------------------------------------------------------------- overview map
def fig_overview():
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.62, DOUBLE * 0.62))
    im = show(ax, masked("topography") / 1000, "Greys", 0, 3)
    base(ax)
    for g in ["Au", "3T", "Cu-Co", "Diamond"]:
        sub = dep[dep.group == g]
        ax.scatter(sub.lon, sub.lat, s=4 if g in ("Au", "3T") else 9, c=GCOL[g], lw=0,
                   alpha=0.8, label=f"{g} ({len(sub)})")
    for i, (name, f) in enumerate(spec["fits"].items()):
        w, e, s, n = f["window"]
        ax.add_patch(Rectangle((w, s), e - w, n - s, fill=False, ec=INK, lw=0.8, ls="--"))
        ax.text(w + 0.15, s + 0.2, f"W{i + 1}", fontsize=7.5, color=INK, weight="bold")
    for x, y, t in [(15.3, -4.4, "Kinshasa"), (27.48, -11.66, "Lubumbashi"), (25.47, -10.71, "Kolwezi"),
                    (28.86, -2.5, "Bukavu"), (29.22, -1.68, "Goma"), (25.19, 0.52, "Kisangani"),
                    (23.6, -6.14, "Mbuji-Mayi")]:
        ax.plot(x, y, "s", ms=3, mfc="white", mec=INK, mew=0.6)
        ax.text(x + 0.2, y + 0.15, t, fontsize=6, color=INK)
    leg = ax.legend(loc="lower left", frameon=True, markerscale=2.2, title="Occurrences",
                    title_fontsize=7)
    leg.get_frame().set_linewidth(0)
    cb = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02)
    cb.set_label("Elevation (km)")
    ax.set_title("Mineral occurrences (MRDS, IPIS, OSM) and analysis windows")
    save(fig, "drc_overview")


# ----------------------------------------------------------------------- potential fields
def fig_fields():
    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE, DOUBLE * 0.98))
    panels = [("bouguer_res", "(a) Residual Bouguer anomaly", "RdBu_r", "mGal", None),
              ("magnetic", "(b) Magnetic anomaly (EMAG2v3, 4 km)", "RdBu_r", "nT", None),
              ("bouguer_hgm", "(c) Horizontal gradient of Bouguer", "Blues", "mGal/km", 1e3),
              ("mag_tilt", "(d) Magnetic tilt derivative", "RdBu_r", "rad", None)]
    for ax, (v, title, cmap, unit, scale) in zip(axs.ravel(), panels):
        da = masked(v) * (scale or 1)
        lim = float(np.nanpercentile(np.abs(da), 98))
        vmin, vmax = (0, lim) if cmap == "Blues" else (-lim, lim)
        if v == "mag_tilt":
            vmin, vmax = -np.pi / 2, np.pi / 2
        gap = ds.inside & ~np.isfinite(ds[v])
        ax.imshow(np.where(gap, 1, np.nan), origin="lower", cmap="Greys", vmin=0, vmax=4,
                  extent=(ds.longitude.min(), ds.longitude.max(), ds.latitude.min(), ds.latitude.max()))
        im = show(ax, da, cmap, vmin, vmax)
        base(ax)
        cu = dep[dep.group == "Cu-Co"]
        ax.scatter(cu.lon, cu.lat, s=3, c=INK, lw=0, label="Cu-Co")
        ax.set_title(title, loc="left")
        cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        cb.set_label(unit)
    axs[0, 1].text(13.0, -12.8, "grey: no EMAG2v3 coverage", fontsize=6.5, color=INK2)
    fig.tight_layout()
    save(fig, "drc_fields")


# ----------------------------------------------------------------------- spectra
def fig_spectra():
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 55 * MM), sharey=False)
    for ax, (name, c) in zip(axs, spec["curves"].items()):
        k, lnp = np.array(c["k"]), np.array(c["lnp"])
        f = spec["fits"][name]
        ax.plot(k, lnp, "o", ms=2.2, color=INK2, label="Radial average")
        kb = f["k_break"]
        k1 = k[(k >= f["k_min"]) & (k <= kb)]
        k2 = k[k >= kb]
        ax.plot(k1, f["deep_fit"][0] + f["deep_fit"][1] * k1, color="#2a78d6", lw=2,
                label=f"Deep: {f['deep_depth_bg_km']:.1f} ± {f['deep_se_km']:.1f} km")
        sh = f["shallow_depth_bg_km"]
        sh_lab = f"Shallow: {sh:.1f} ± {f['shallow_se_km']:.1f} km" if sh > 0 else "Shallow: near surface (unresolved)"
        ax.plot(k2, f["shallow_fit"][0] + f["shallow_fit"][1] * k2, color="#eb6834", lw=2, label=sh_lab)
        idx = list(spec["curves"]).index(name) + 1
        ax.set_title(f"W{idx} {name.split(' (')[0]}, coverage {100 * f['coverage']:.0f}%", loc="left")
        ax.set_xlabel("Wavenumber k (rad/km)")
        ax.grid(color=GRID, lw=0.5)
        ax.legend(frameon=False, loc="lower left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_ylabel("ln P(k)")
    fig.tight_layout()
    save(fig, "drc_spectra")


# ----------------------------------------------------------------------- districts
def fig_districts():
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 92 * MM), gridspec_kw={"width_ratios": [1.05, 1]})
    ext = (25.5, 30.8, -6.0, 3.6)
    base(a, ext)
    other = dep[dep.district < 0]
    a.scatter(other.lon, other.lat, s=2, c="#c3c2b7", lw=0, label="Unassigned / small")
    top = dist.head(12)
    for _, r in top.iterrows():
        sub = dep[dep.district == r.district - 1]
        a.scatter(sub.lon, sub.lat, s=3, c=GCOL[r.dominant], lw=0)
        a.text(r.lon + 0.12, r.lat, str(int(r.district)), fontsize=7, color=INK, weight="bold")
    a.set_title(f"(a) Louvain districts (Q = {gstats['modularity']:.2f})", loc="left")
    a.set_xticks(np.arange(26, 31, 1))
    y = np.arange(len(top))[::-1]
    left = np.zeros(len(top))
    for g in ["Au", "3T", "Cu-Co", "Diamond", "Other"]:
        vals = np.array([json.loads(c.replace("'", '"')).get(g, 0) for c in top.composition])
        if vals.sum() == 0:
            continue
        b.barh(y, vals, left=left, color=GCOL[g], height=0.7, label=g, edgecolor="white", lw=0.6)
        left += vals
    b.set_yticks(y, [f"D{int(d)}" for d in top.district])
    b.set_xlabel("Number of occurrences")
    b.set_title("(b) Composition of the 12 largest", loc="left")
    b.legend(frameon=False, loc="lower right")
    b.grid(axis="x", color=GRID, lw=0.5)
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "drc_districts")


# ----------------------------------------------------------------------- prospectivity
def fig_prospectivity():
    fig = plt.figure(figsize=(DOUBLE, 128 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
    views = [("CuCo", "Cu-Co", (24.0, 28.6, -12.6, -9.0), "(a) Cu–Co, Haut-Katanga & Lualaba"),
             ("Au", "Au", (25.8, 30.6, -6.0, 3.6), "(b) Au, eastern DRC"),
             ("3T", "3T", (25.8, 30.6, -6.0, 3.6), "(c) 3T, eastern DRC")]
    for i, (var, g, ext, title) in enumerate(views):
        ax = fig.add_subplot(gs[0, i])
        im = show(ax, pros[var], "Blues", 0, 100)
        base(ax, ext)
        sub = dep[dep.group == g]
        ax.scatter(sub.lon, sub.lat, s=2.5, c=GCOL[g], lw=0, alpha=0.9)
        meth = pros[var].attrs.get("method", "").replace("Graph label spreading", "graph LS")
        ax.set_title(f"{title}\n{meth}", loc="left")
        ax.set_xticks(np.arange(np.ceil(ext[0]), ext[1], 2))
    cax = fig.add_axes([0.92, 0.55, 0.012, 0.3])
    fig.colorbar(im, cax=cax, label="Out-of-fold score (percentile)")
    ax = fig.add_subplot(gs[1, :])
    methods = list(dict.fromkeys(cv.method))
    groups = ["Cu-Co", "Au", "3T"]
    x = np.arange(len(methods))
    for j, g in enumerate(groups):
        sub = cv[cv.group == g].set_index("method").loc[methods]
        ax.errorbar(x + (j - 1) * 0.18, sub.auc_fold_mean, yerr=sub.auc_fold_sd, fmt="o", ms=4,
                    color=GCOL[g], elinewidth=1.2, label=g)
    ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
    ax.text(len(methods) - 0.55, 0.51, "random", fontsize=6.5, color=INK2)
    ax.set_xticks(x, [m.replace("Graph label spreading ", "Graph LS\n") for m in methods])
    ax.set_ylabel("AUC, spatial block CV")
    ax.set_title("(d) Prospectivity skill by method (mean ± s.d. over 1° blocks)", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(axis="y", color=GRID, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.06, right=0.9, hspace=0.35, wspace=0.25)
    save(fig, "drc_prospectivity")


# ----------------------------------------------------------------------- tables
FNAME = {"bouguer": "Bouguer anomaly (mGal)", "bouguer_hgm": "Bouguer HGM (mGal/km)",
         "magnetic": "Magnetic anomaly (nT)", "mag_asa": "Analytic signal (nT/km)",
         "mag_tilt": "Tilt derivative (rad)", "topography": "Elevation (m)"}


def stars(p):
    return "$^{***}$" if p < 1e-3 else "$^{**}$" if p < 1e-2 else "$^{*}$" if p < 0.05 else ""


def tables():
    t = dep.groupby(["source", "group"]).size().unstack(fill_value=0)
    t = t[[c for c in ["Cu-Co", "Au", "3T", "Diamond", "Other"] if c in t]]
    t.loc["Total"] = t.sum()
    t["Total"] = t.sum(1)
    rows = "\n".join(f"{i} & " + " & ".join(str(v) for v in r) + " \\\\" for i, r in t.iterrows())
    (GEN / "tab_deposits.tex").write_text(
        "\\begin{table}[t]\n\\centering\\small\n\\caption{Occurrences inside the DRC by source and "
        "commodity group after harmonisation.}\\label{tab:deposits}\n"
        f"\\begin{{tabular*}}{{\\tblwidth}}{{@{{\\extracolsep{{\\fill}}}}l{'r' * len(t.columns)}@{{}}}}\n\\toprule\n"
        "Source & " + " & ".join(t.columns) + " \\\\\n\\midrule\n" + rows +
        "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n", encoding="utf-8")

    groups = [g for g in ["Cu-Co", "Au", "3T"] if g in set(sig.group)]
    lines = []
    for f in FNAME:
        cells = []
        for g in groups:
            r = sig[(sig.group == g) & (sig.feature == f)]
            if r.empty:
                cells.append("--")
                continue
            r = r.iloc[0]
            cells.append(f"{r.median_group:.3g} / {r.median_background:.3g} & {r.ks_D:.2f}{stars(r.mw_p)}")
        lines.append(f"{FNAME[f]} & " + " & ".join(cells) + " \\\\")
    ns = {g: int(sig[sig.group == g].n_cells.max()) for g in groups}
    (GEN / "tab_signatures.tex").write_text(
        "\\begin{table*}[t]\n\\centering\\small\n\\caption{Geophysical signatures of occurrence cells "
        "against the background of their own study region: median (occurrence / background) and "
        "Kolmogorov--Smirnov distance $D$; Mann--Whitney significance $^{*}p<0.05$, $^{**}p<0.01$, "
        "$^{***}p<0.001$. Number of occurrence cells: " +
        ", ".join(f"{g} {n}" for g, n in ns.items()) + ".}\\label{tab:signatures}\n"
        "\\begin{tabular*}{\\tblwidth}{@{\\extracolsep{\\fill}}l" + "rr" * len(groups) + "@{}}\n\\toprule\n"
        " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{g}}}" for g in groups) + " \\\\\n"
        "Feature & " + " & ".join("Median & $D$" for _ in groups) + " \\\\\n\\midrule\n" +
        "\n".join(lines) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table*}\n", encoding="utf-8")

    rows = []
    for name, f in spec["fits"].items():
        sh = f["shallow_depth_bg_km"]
        sh_txt = f"{sh:.1f} $\\pm$ {f['shallow_se_km']:.1f}" if sh > 0 else "unresolved"
        rows.append(f"{name} & {f['size_km'][0]:.0f}$\\times${f['size_km'][1]:.0f} & "
                    f"{100 * f['coverage']:.0f} & {f['mean_elevation_km']:.2f} & "
                    f"{f['deep_depth_bg_km']:.1f} $\\pm$ {f['deep_se_km']:.1f} & {sh_txt} \\\\")
    (GEN / "tab_spectral.tex").write_text(
        "\\begin{table*}[t]\n\\centering\\small\n\\caption{Depths to magnetic source ensembles from "
        "radially averaged EMAG2v3 power spectra (two-segment fit; depths below mean ground level, "
        "$\\pm$ one standard error of the slope).}\\label{tab:spectral}\n"
        "\\begin{tabular*}{\\tblwidth}{@{\\extracolsep{\\fill}}lrrrrr@{}}\n\\toprule\n"
        "Window & Size (km) & Coverage (\\%) & Elevation (km) & Deep sources (km) & Shallow sources (km) \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table*}\n",
        encoding="utf-8")

    top = dist.head(10).copy()
    pts = gpd.GeoDataFrame(top, geometry=gpd.points_from_xy(top.lon, top.lat), crs=4326)
    top["province"] = gpd.sjoin_nearest(pts, adm1[["shapeName", "geometry"]], how="left")["shapeName"].values
    top[["district", "province"]].to_csv(RES / "district_provinces.csv", index=False)
    rows = [f"D{int(r.district)} & {r.province} & {r.n_sites} & {r.lon:.2f}, {r.lat:.2f} & {r.dominant} "
            f"({100 * r.dominant_share:.0f}\\%) & {r.density:.2f} \\\\" for _, r in top.iterrows()]
    (GEN / "tab_districts.tex").write_text(
        "\\begin{table}[t]\n\\centering\\small\n\\caption{The ten largest mining districts found as "
        f"Louvain communities of the occurrence graph ({gstats['n_nodes']} nodes, {gstats['n_edges']} "
        f"edges within {gstats['link_km']:.0f}~km; modularity {gstats['modularity']:.2f}).}}"
        "\\label{tab:districts}\n"
        "\\begin{tabular*}{\\tblwidth}{@{\\extracolsep{\\fill}}llrlll@{}}\n\\toprule\n"
        "District & Province & Sites & Centre (°E, °N) & Dominant & Density \\\\\n\\midrule\n"
        .replace("°", "$^\\circ$") + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n",
        encoding="utf-8")

    methods = list(dict.fromkeys(cv.method))
    rows = []
    for m in methods:
        cells = []
        for g in ["Cu-Co", "Au", "3T"]:
            r = cv[(cv.group == g) & (cv.method == m)].iloc[0]
            cells.append(f"{r.auc_fold_mean:.2f} $\\pm$ {r.auc_fold_sd:.2f} & {100 * r.capture_top10:.0f}")
        rows.append(f"{m} & " + " & ".join(cells) + " \\\\")
    npos = {g: int(cv[cv.group == g].n_positive_cells.iloc[0]) for g in ["Cu-Co", "Au", "3T"]}
    (GEN / "tab_prospectivity.tex").write_text(
        "\\begin{table*}[t]\n\\centering\\small\n\\caption{Prospectivity skill under spatially blocked "
        "(1$^\\circ$) cross-validation: AUC (mean $\\pm$ s.d. over folds) and the percentage of "
        "occurrence cells captured in the top 10\\% of the ranked area. Positive cells: " +
        ", ".join(f"{g} {n}" for g, n in npos.items()) + ".}\\label{tab:prospectivity}\n"
        "\\begin{tabular*}{\\tblwidth}{@{\\extracolsep{\\fill}}lrrrrrr@{}}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Cu--Co} & \\multicolumn{2}{c}{Au} & \\multicolumn{2}{c}{3T} \\\\\n"
        "Method & AUC & Top 10\\% & AUC & Top 10\\% & AUC & Top 10\\% \\\\\n\\midrule\n" +
        "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table*}\n", encoding="utf-8")


if __name__ == "__main__":
    for p in (FIG, GEN):
        p.mkdir(parents=True, exist_ok=True)
    add_residual_bouguer()
    fig_overview()
    fig_fields()
    fig_spectra()
    fig_districts()
    fig_prospectivity()
    tables()
    print("regional figures + tables written")
