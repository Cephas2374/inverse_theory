"""Figures and LaTeX tables for the field inversions (paper Section 3.5)."""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from drcinv import TensorMesh  # noqa: E402

RES = ROOT / "results" / "field"
FIG = ROOT / "paper" / "figures"
GEN = ROOT / "paper" / "sections" / "generated"
KM_PER_DEG = 111.195
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
RUNCOL = {"A": "#9a9893", "B": "#2a78d6", "G": "#eb6834", "B2670": "#1baf7a"}
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42,
})
MM = 1 / 25.4
DOUBLE = 190 * MM

S = json.loads((RES / "summary.json").read_text())
adm0 = gpd.read_file(ROOT / "data" / "raw" / "boundaries" / "COD_ADM0.geojson").to_crs(4326)
adm1 = gpd.read_file(ROOT / "data" / "raw" / "boundaries" / "COD_ADM1.geojson").to_crs(4326)
dep = gpd.read_file(ROOT / "data" / "processed" / "deposits.gpkg")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def mesh_of(z):
    m = z["mesh"]
    return TensorMesh(*m[:6], *map(int, m[6:]))


def cell_axes(mesh, lon0, lat0):
    k = KM_PER_DEG * 1000
    xc = mesh.x0 + mesh.dx * (np.arange(mesh.nx) + 0.5)
    yc = mesh.y0 + mesh.dy * (np.arange(mesh.ny) + 0.5)
    return xc / (k * np.cos(np.radians(lat0))) + lon0, yc / k + lat0


def extent_of(lon, lat):
    dl, dt = lon[1] - lon[0], lat[1] - lat[0]
    return (lon[0] - dl / 2, lon[-1] + dl / 2, lat[0] - dt / 2, lat[-1] + dt / 2)


def base(ax, ext, step=1):
    adm1.boundary.plot(ax=ax, color=INK2, lw=0.3)
    adm0.boundary.plot(ax=ax, color=INK, lw=0.7)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.tick_params(length=2)
    ax.set_xticks(np.arange(np.ceil(ext[0]), ext[1] + 1e-9, step))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}°E")
    ax.yaxis.set_major_formatter(lambda v, _: f"{abs(v):g}°{'S' if v < 0 else 'N' if v > 0 else ''}")


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def pm(v, fmt="{:.1f}"):
    """P50 with P10--P90 range."""
    return f"{fmt.format(v[1])} ({fmt.format(v[0])}--{fmt.format(v[2])})"


# ----------------------------------------------------------------------------- Katanga
def fig_katanga():
    z = np.load(RES / "katanga.npz")
    K = S["katanga"]
    mesh = mesh_of(z)
    lon0, lat0 = float(z["lon0"]), float(z["lat0"])
    clon, clat = cell_axes(mesh, lon0, lat0)
    rho = int(K["nettleton_density"])
    cu = dep[dep.group == "Cu-Co"]
    ext = (24.6, 28.4, -12.4, -9.8)
    fig = plt.figure(figsize=(DOUBLE, 110 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.8], hspace=0.45, wspace=0.18)

    ax = fig.add_subplot(gs[0, 0])
    d = z["d"].reshape(len(z["lat"]), len(z["lon"]))
    lim = np.percentile(np.abs(d), 98)
    im = ax.imshow(d, origin="lower", extent=extent_of(z["lon"], z["lat"]), cmap="RdBu_r",
                   vmin=-lim, vmax=lim, interpolation="nearest")
    base(ax, ext)
    ax.scatter(cu.lon, cu.lat, s=3, c=INK, lw=0)
    ax.set_title(f"(a) Residual Bouguer, XGM2019e ($\\rho_B$ = {rho})", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="mGal")

    ax = fig.add_subplot(gs[0, 1])
    top = mesh.to_grid(z[f"map_B_{rho}"])[:, :, :2].mean(2).T
    lim = np.percentile(np.abs(top), 99)
    im = ax.imshow(top, origin="lower", extent=extent_of(clon, clat), cmap="RdBu_r",
                   vmin=-lim, vmax=lim, interpolation="nearest")
    dist = z["district"].reshape(mesh.shape)[:, :, 0].T
    ax.contour(clon, clat, dist.astype(float), levels=[0.5], colors=INK, linewidths=0.6)
    base(ax, ext)
    ax.scatter(cu.lon, cu.lat, s=3, c=INK, lw=0)
    ax.axvline(SECTION_LON, color=INK2, lw=0.7, ls="--")
    ax.set_title("(b) $\\ell_2$ fabric graph, mean $\\Delta\\rho$ at 0–2 km", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="g/cm$^3$")

    ix = int(np.argmin(np.abs(clon - SECTION_LON)))
    depth = (np.arange(mesh.nz) + 0.5) * mesh.dz / 1000
    sec_axes = [fig.add_subplot(gs[1, j]) for j in range(2)]
    for ax, key, lab in zip(sec_axes, [f"A_{rho}", f"B_{rho}"], ["(c) Tikhonov", "(d) $\\ell_2$ fabric graph"]):
        sec = mesh.to_grid(z[f"map_{key}"])[ix].T
        lim = np.percentile(np.abs(mesh.to_grid(z[f"map_B_{rho}"])), 99)
        im = ax.imshow(sec, origin="upper", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                       extent=(clat[0], clat[-1], depth[-1] + 0.5, 0), interpolation="nearest")
        dsec = z["district"].reshape(mesh.shape)[ix, :, 0]
        for j in np.where(dsec)[0]:
            ax.plot([clat[j] - 0.022, clat[j] + 0.022], [0.12, 0.12], color=INK, lw=2.5, solid_capstyle="butt")
        ax.set_xlabel("Latitude")
        ax.xaxis.set_major_formatter(lambda v, _: f"{abs(v):g}°S")
        ax.set_ylabel("Depth (km)")
        ax.set_title(f"{lab}, N–S section at {SECTION_LON:g}°E", loc="left")
    fig.colorbar(im, ax=sec_axes, shrink=0.9, pad=0.02, label="g/cm$^3$")

    save(fig, "field_katanga")


SECTION_LON = 26.2
NAME = {"A": "Tikhonov", "B": "$\\ell_2$ fabric graph", "G": "Sparse graph"}


def tab_katanga():
    K = S["katanga"]
    rho = int(K["nettleton_density"])
    keys = [k for k in [f"A_{rho}", f"B_{rho}", f"G_{rho}", "B_2670"] if k in K["runs"]]
    rows = []
    for k in keys:
        r = K["runs"][k]
        sl = r["slices"]
        cells = [pm(sl[s]["district_Mt_per_km2"], "{:.0f}") for s in ["0-1000", "1000-2000", "2000-3000"]]
        bg = sum(sl[s]["background_Mt_per_km2"][1] for s in ["0-1000", "1000-2000", "2000-3000"])
        rows.append(f"{NAME[r['config']]} & {int(r['rho_reduction'])} & {r['chi2_per_datum']:.2f} & " + " & ".join(cells)
                    + f" & ${bg:.0f}$ & {pm(r['excess_over_background_0_3km_Gt'], '{:.0f}')} \\\\")
    (GEN / "tab_field_katanga.tex").write_text(
        "\\begin{table*}[t]\n\\centering\\small\n\\caption{Katanga endowment inversion. Anomalous mass per "
        "unit area beneath the Cu--Co district (cells within 10~km of an occurrence, "
        f"{K['district_area_km2']:.0f}~km$^2$) by depth slice (P50 with P10--P90 from 40 RTO samples), the "
        f"background of Haut-Katanga and Lualaba farther than 30~km from any occurrence ({K['background_area_km2']:.0f}~km$^2$, "
        "P50 summed over 0--3~km), and the district excess over that background. $\\rho_B$: terrain "
        "reduction density.}\\label{tab:fieldkatanga}\n"
        "\\begin{tabular*}{\\tblwidth}{@{\\extracolsep{\\fill}}lrrllllr@{}}\n\\toprule\n"
        " & & & \\multicolumn{3}{c}{District (Mt/km$^2$)} & Background & Excess \\\\\n"
        "Method & $\\rho_B$ (kg/m$^3$) & $\\chi^2/N$ & 0--1 km & 1--2 km & 2--3 km & 0--3 km (Mt/km$^2$) & 0--3 km (Gt) \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table*}\n", encoding="utf-8")


# ----------------------------------------------------------------------------- Cuvette
def mag_windows(C):
    return [r for r in C["mag_windows"] if 0 < r["shallow_km"] < 20 and r["shallow_se_km"] < 2.0]


def combined_depth(z, C, tag, sigma_deg=1.0, reach_deg=1.2):
    """Depth to basement = smooth magnetic trend (Gaussian-weighted window depths) + gravity
    thickness variation. The two are complementary in wavelength: the 3-degree spectral windows
    give the absolute level over ~330 km, the residual gravity (150 km regional) the sub-basin
    variation. Cells farther than reach_deg from every window centre are masked."""
    glon, glat = z["glon"], z["glat"]
    LON, LAT = np.meshgrid(glon, glat, indexing="ij")
    mags = mag_windows(C)
    wl = np.array([[r["lon"], r["lat"]] for r in mags])
    dep_ = np.array([r["shallow_km"] for r in mags])
    d2 = (LON[..., None] - wl[:, 0]) ** 2 + (LAT[..., None] - wl[:, 1]) ** 2
    wts = np.exp(-0.5 * d2 / sigma_deg ** 2)
    trend = (wts * dep_).sum(-1) / wts.sum(-1)
    trend[np.sqrt(d2.min(-1)) > reach_deg] = np.nan
    out = {q: trend + z[f"thick_{q}__{tag}"] for q in ("p10", "p50", "p90")}
    out["trend"] = trend
    return out


def fig_cuvette():
    z = np.load(RES / "cuvette.npz")
    C = S["cuvette"]
    mesh = mesh_of(z)
    glon, glat = z["glon"], z["glat"]
    ext = (16.5, 25.5, -5.0, 3.0)
    reg = int(min(int(t.split("_")[1]) for t in C["runs"] if t.startswith("A_")))
    tag = f"B_{reg}"
    mags = mag_windows(C)
    comb = combined_depth(z, C, tag)
    fig = plt.figure(figsize=(DOUBLE, 158 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.8], hspace=0.32, wspace=0.3)
    hcb = dict(shrink=0.85, pad=0.13, orientation="horizontal", aspect=25)

    ax = fig.add_subplot(gs[0, 0])
    d = z[f"d__{tag}"].reshape(len(z["lat"]), len(z["lon"]))
    lim = np.percentile(np.abs(d), 98)
    im = ax.imshow(d, origin="lower", extent=extent_of(z["lon"], z["lat"]), cmap="RdBu_r",
                   vmin=-lim, vmax=lim, interpolation="nearest")
    base(ax, ext, 2)
    ax.set_title(f"(a) Residual Bouguer ({reg} km)", loc="left")
    fig.colorbar(im, ax=ax, label="mGal", **hcb)

    ax = fig.add_subplot(gs[0, 1])
    t = z[f"thick_p50__{tag}"].T
    lim = np.percentile(np.abs(t), 98)
    im = ax.imshow(t, origin="lower", extent=extent_of(glon, glat), cmap="PuOr_r", vmin=-lim, vmax=lim,
                   interpolation="nearest")
    base(ax, ext, 2)
    ax.axhline(SECTION_LAT, color=INK2, lw=0.7, ls="--")
    ax.set_title("(b) Thickness variation (gravity)", loc="left")
    fig.colorbar(im, ax=ax, label="km (positive: thicker)", **hcb)

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(comb["p50"].T, origin="lower", extent=extent_of(glon, glat), cmap="viridis_r",
                   vmin=0, vmax=11, interpolation="nearest")
    base(ax, ext, 2)
    ax.scatter([r["lon"] for r in mags], [r["lat"] for r in mags], c=[r["shallow_km"] for r in mags],
               s=20, cmap="viridis_r", vmin=0, vmax=11, edgecolors=INK, linewidths=0.5)
    ax.set_title("(c) Depth to basement, combined", loc="left")
    fig.colorbar(im, ax=ax, label="km below ground (dots: magnetic windows)", **hcb)

    ax = fig.add_subplot(gs[1, :2])
    jy = int(np.argmin(np.abs(glat - SECTION_LAT)))
    sec = mesh.to_grid(z[f"map__{tag}"])[:, jy, :].T
    lim = np.percentile(np.abs(mesh.to_grid(z[f"map__{tag}"])), 99)
    depth_max = mesh.nz * mesh.dz / 1000
    im = ax.imshow(sec, origin="upper", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   extent=(glon[0], glon[-1], depth_max, 0), interpolation="nearest")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Depth (km)")
    ax.set_title(f"(d) $\\ell_2$ fabric-graph $\\Delta\\rho$, E–W section at {abs(SECTION_LAT):g}°"
                 f"{'S' if SECTION_LAT < 0 else 'N'}", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.9, pad=0.015).set_label("g/cm$^3$", labelpad=1)

    ax = fig.add_subplot(gs[1, 2])
    a = np.array([r["grav_thick_var_km"] for r in mags])
    b = np.array([r["shallow_km"] for r in mags])
    se = np.array([r["shallow_se_km"] for r in mags])
    ax.errorbar(a, b, yerr=se, fmt="o", ms=3, color="#2a78d6", elinewidth=0.6, alpha=0.8)
    mv = C.get("mag_vs_grav", {})
    ax.set_xlabel("Gravity thickness variation (km)")
    ax.set_ylabel("Magnetic depth (km)")
    ax.set_title(f"(e) Spearman {mv.get('spearman_r', np.nan):.2f}, n = {mv.get('n_windows', 0)}", loc="left")
    ax.grid(color=GRID, lw=0.5)
    clean(ax)
    save(fig, "field_cuvette")

    # statistics of the combined depth inside the DRC part of the basin interior
    from shapely import contains_xy
    LON, LAT = np.meshgrid(glon, glat, indexing="ij")
    inside = contains_xy(adm0.union_all(), LON, LAT) & np.isfinite(comb["p50"])
    interior = inside & (LAT > -3.0)
    stats = {"interior_p50_percentiles_2_50_98": np.percentile(comb["p50"][interior], [2, 50, 98]).tolist(),
             "interior_max_p50": float(np.nanmax(comb["p50"][interior])),
             "interior_frac_deeper_4p56km": float((comb["p50"][interior] > 4.56).mean()),
             "interior_frac_8_to_9plus_km": float((comb["p50"][interior] >= 8.0).mean()),
             "all_p50_percentiles_2_50_98": np.percentile(comb["p50"][inside], [2, 50, 98]).tolist(),
             "interior_halfwidth_median_km": float(np.median((comb["p90"] - comb["p10"])[interior] / 2)),
             "n_cells_interior": int(interior.sum())}
    (RES / "cuvette_combined.json").write_text(json.dumps(stats, indent=1))
    print("combined depth:", stats)


SECTION_LAT = -0.5


def tab_cuvette():
    C = S["cuvette"]
    order = sorted(C["runs"], key=lambda t: (int(t.split("_")[1]) != 150, t))
    rows = []
    for t in order:
        r = C["runs"][t]
        reg = C["regional"][str(int(r["regional_km"]))]
        rows.append(f"{NAME[t.split('_')[0]]} & {r['regional_km']:.0f} & {reg['residual_sd_mgal']:.1f} & "
                    f"{r['chi2_per_datum']:.2f} & ${r['thick_var_range_km'][0]:.1f}$ to $+{r['thick_var_range_km'][1]:.1f}$ & "
                    f"{r['thick_halfwidth_p10_p90_km_median']:.1f} \\\\")
    (GEN / "tab_field_cuvette.tex").write_text(
        "\\begin{table*}[t]\n\\centering\\small\n\\caption{Cuvette Centrale basement inversion. Residual "
        "standard deviation for each Gaussian regional, misfit, the 2nd--98th percentile range of the "
        "sediment-thickness variation and the median P10--P90 half-width (40 RTO samples, sediment "
        "contrast $-0.20\\pm0.04$~g/cm$^3$).}\\label{tab:fieldcuvette}\n"
        "\\begin{tabular*}{\\tblwidth}{@{\\extracolsep{\\fill}}lrrrlr@{}}\n\\toprule\n"
        "Method & Regional (km) & sd (mGal) & $\\chi^2/N$ & Variation (km) & $\\pm$ (km) \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table*}\n", encoding="utf-8")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    # tables are written by katanga_validation.py and cuvette_analysis.py (version 2 runs)
    if "katanga" in S:
        fig_katanga()
    if "cuvette" in S:
        fig_cuvette()
    print("field figures + tables written")
