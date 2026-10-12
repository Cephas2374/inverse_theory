"""Depth sections along profiles (paper Section 3.5).

Katanga: a swath (+-10 km) along the Copperbelt arc from Kalongwe and Kolwezi to
Lubumbashi. Anomalous mass in 0-3 km (main run P10-P90 and choice ensemble), the Roan and
province background levels, the USGS deposits projected onto the profile, and the density
section of the l2 fabric-graph model beneath it.
Cuvette Centrale: two sections (N-S along 21 E from the Kasai margin into the basin, and
E-W along 0.5 S). Residual Bouguer anomaly, depth to basement (choice-ensemble P10-P50-P90)
with the sediment fill shaded, and the local magnetic spectral depths of windows centred
within 0.5 degrees of the line. Well total depths are drawn as reference levels only: the
wells did not reach basement and their coordinates are not public.
Inputs: results/field_v2. Outputs: paper/figures/field_profiles_{katanga,cuvette}.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "experiments"))
from drcinv import TensorMesh  # noqa: E402
from cuvette_analysis import mag_windows, trend  # noqa: E402
from katanga_validation import USGS, cell_centres, load_runs  # noqa: E402

RES = ROOT / "results" / "field_v2"
FIG = ROOT / "paper" / "figures"
K = 111.195          # km per degree
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, GREY, ORANGE, SED, BASE = "#2a78d6", "#9a9893", "#eb6834", "#f1dfb0", "#8c8c8c"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42})
MM = 1 / 25.4
WELLS = [("Gilson-1", 4.563), ("Mbandaka-1", 4.350), ("Samba", 2.039), ("Dekese", 1.856)]


def polyline(nodes, step_km):
    """Points every step_km along a lon/lat polyline; returns lon, lat, distance (km)."""
    lon, lat, dist = [], [], []
    d0 = 0.0
    for (x1, y1), (x2, y2) in zip(nodes[:-1], nodes[1:]):
        L = np.hypot((x2 - x1) * K * np.cos(np.radians(0.5 * (y1 + y2))), (y2 - y1) * K)
        n = max(int(L // step_km), 1)
        t = np.arange(n) / n
        lon += list(x1 + t * (x2 - x1))
        lat += list(y1 + t * (y2 - y1))
        dist += list(d0 + t * L)
        d0 += L
    lon.append(nodes[-1][0])
    lat.append(nodes[-1][1])
    dist.append(d0)
    return np.array(lon), np.array(lat), np.array(dist)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# =============================================================================== Katanga
ARC = [(25.20, -10.88), (25.45, -10.70), (25.90, -10.73), (26.20, -10.60), (26.42, -10.77),
       (26.81, -10.97), (27.08, -11.23), (27.50, -11.62)]
TOWNS = [(25.47, -10.71, "Kolwezi"), (26.29, -10.60, "Fungurume"), (26.73, -10.98, "Likasi"),
         (27.48, -11.66, "Lubumbashi")]


def katanga():
    runs = load_runs()
    rmain = next(r for r in runs if r["meta"]["tag"] == "main_B")
    mesh, X, Y, CLON, CLAT = cell_centres(rmain["mesh"], rmain["lon0"], rmain["lat0"])
    area = mesh.dx * mesh.dy / 1e6
    v = json.loads((RES / "katanga_validation.json").read_text())
    lon, lat, dist = polyline(ARC, 2.5)
    cx = CLON.ravel() * K * np.cos(np.radians(-11))
    cy = CLAT.ravel() * K
    tree = cKDTree(np.column_stack([cx, cy]))
    px, py = lon * K * np.cos(np.radians(-11)), lat * K
    swath = tree.query_ball_point(np.column_stack([px, py]), r=10.0)

    def profile(slices):                      # (n_s, nslices, nx, ny) -> (n_s, n_points) Mt/km2
        M = slices[:, 0:3].sum(1).reshape(len(slices), -1) / area
        return np.stack([M[:, idx].mean(1) for idx in swath], 1)

    Pm = profile(rmain["slices"])
    ens = np.concatenate([profile(r["slices"]) for r in runs if r["meta"]["tag"].startswith("ens_")])
    bg_roan = v["runs"]["main_B"]["bg_geo"]
    bg_prov = v["runs"]["main_B"]["bg_prov"]
    # density section along the profile (MAP model, swath mean per layer)
    G = rmain["map"].reshape(mesh.nx * mesh.ny, mesh.nz)
    sec = np.stack([G[idx].mean(0) for idx in swath], 1)            # (nz, n_points)
    # USGS deposits within 15 km of the profile
    usgs = gpd.read_file(USGS / "CACB_deposits_prospects.shp")
    usgs = usgs[usgs.Country.str.contains("Congo") & (usgs.Tonnage_Mt > 0)].copy()
    usgs["cu"] = usgs.Tonnage_Mt * usgs.Cu_pct / 100
    ux, uy = usgs.Longitude.values * K * np.cos(np.radians(-11)), usgs.Latitude.values * K
    dd, ii = cKDTree(np.column_stack([px, py])).query(np.column_stack([ux, uy]))
    sel = dd <= 15
    ud, ucu, uname = dist[ii[sel]], usgs.cu.values[sel], usgs.Name.values[sel]
    tdist = [dist[np.argmin(np.hypot(px - x * K * np.cos(np.radians(-11)), py - y * K))] for x, y, _ in TOWNS]

    fig, axs = plt.subplots(3, 1, figsize=(190 * MM, 160 * MM), sharex=True,
                            gridspec_kw={"height_ratios": [0.6, 1, 0.9], "hspace": 0.32})
    fig.subplots_adjust(right=0.88)
    ax = axs[0]
    ax.vlines(ud, 0, ucu, color=ORANGE, lw=2)
    ax.scatter(ud, ucu, s=10, color=ORANGE, zorder=3)
    placed = []
    for k_ in np.argsort(-ucu):
        d_, c_, n_ = ud[k_], ucu[k_], uname[k_]
        if c_ >= 1.5 and all(abs(d_ - p_) > 14 for p_ in placed):
            ax.text(d_, c_ * 1.25, n_.split("-")[0].split(" (")[0], fontsize=6, ha="center", va="bottom", color=INK)
            placed.append(d_)
    ax.set_yscale("log")
    ax.set_ylim(0.03, 200)
    ax.set_ylabel("USGS Cu (Mt)")
    ax.set_title("(a) USGS deposits within 15 km of the profile (contained copper)", loc="left")
    clean(ax)
    ax = axs[1]
    e10, e50, e90 = np.percentile(ens, [10, 50, 90], 0)
    m10, m50, m90 = np.percentile(Pm, [10, 50, 90], 0)
    ax.fill_between(dist, e10, e90, color=GREY, alpha=0.35, lw=0, label="Choice ensemble P10–P90")
    ax.fill_between(dist, m10, m90, color=BLUE, alpha=0.3, lw=0, label="$\\ell_2$ fabric graph P10–P90")
    ax.plot(dist, m50, color=BLUE, lw=1.4, label="$\\ell_2$ fabric graph P50")
    ax.axhspan(bg_roan[0], bg_roan[2], color="#7a5c00", alpha=0.15, lw=0)
    ax.axhline(bg_roan[1], color="#7a5c00", lw=1, ls="--", label="Roan background (barren host)")
    ax.axhline(bg_prov[1], color=INK2, lw=0.8, ls=":", label="Province background")
    ax.set_ylabel("Mass 0–3 km (Mt/km$^2$)")
    ax.set_title("(b) Anomalous mass in the upper 3 km, ±10 km swath", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=6.5)
    ax.grid(axis="y", color=GRID, lw=0.5)
    clean(ax)
    ax = axs[2]
    depth = np.arange(mesh.nz + 1) * mesh.dz / 1000
    lim = np.percentile(np.abs(sec), 98)
    im = ax.pcolormesh(dist, depth, np.vstack([sec, sec[-1:]]), cmap="RdBu_r", vmin=-lim, vmax=lim,
                       shading="nearest")
    ax.invert_yaxis()
    ax.set_ylabel("Depth (km)")
    ax.set_xlabel("Distance along the arc (km)")
    ax.set_title("(c) $\\ell_2$ fabric-graph density contrast beneath the profile (depth is not resolved by the data)",
                 loc="left")
    pos = axs[2].get_position()
    cb = fig.colorbar(im, cax=fig.add_axes([0.9, pos.y0, 0.012, pos.height]))
    cb.set_label("g/cm$^3$")
    for a in axs:
        for td, (_, _, name) in zip(tdist, TOWNS):
            a.axvline(td, color=INK2, lw=0.5, ls=(0, (2, 2)))
    for td, (_, _, name) in zip(tdist, TOWNS):
        axs[1].text(td + 2, axs[1].get_ylim()[0] + 8, name, fontsize=6.5, ha="left", va="bottom", color=INK2)
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"field_profiles_katanga.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return dict(length_km=float(dist[-1]), n_deposits=int(sel.sum()),
                mass_p50_range=[float(m50.min()), float(m50.max())],
                frac_above_roan=float(np.mean(m50 > bg_roan[1])),
                frac_above_roan_p10=float(np.mean(m10 > bg_roan[2])))


# =============================================================================== Cuvette
LINES = {"N–S along 21°E": [(21.0, -4.6), (21.0, 1.6)], "E–W along 0.5°S": [(17.0, -0.5), (25.0, -0.5)]}


def cuvette():
    runs = {}
    for p in sorted((RES / "cuvette_runs").glob("*.npz")):
        z = np.load(p)
        runs[p.stem] = dict(meta=json.loads(str(z["meta"])), thick=z["thick"], d=z["d"], lon=z["lon"],
                            lat=z["lat"], lon_e=z["lon_e"], lat_e=z["lat_e"])
    r0 = runs["main_B"]
    glon = 0.5 * (r0["lon_e"][1:] + r0["lon_e"][:-1])
    glat = 0.5 * (r0["lat_e"][1:] + r0["lat_e"][:-1])
    mags = mag_windows()
    rng = np.random.default_rng(5)
    Z = np.array([trend(glon, glat, mags, rng.uniform(0.7, 1.5)) + s
                  for t, r in runs.items() if t.startswith("ens_") for s in r["thick"]])
    P = np.nanpercentile(Z, [10, 50, 90], 0)
    d = r0["d"].reshape(len(r0["lat"]), len(r0["lon"]))
    gi = RegularGridInterpolator((r0["lat"], r0["lon"]), d, bounds_error=False)
    out = {}
    fig = plt.figure(figsize=(190 * MM, 135 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.45, 1], hspace=0.22, wspace=0.18)
    for j, (name, nodes) in enumerate(LINES.items()):
        lon, lat, dist = polyline(nodes, 5.0)
        pts = np.column_stack([lon, lat])
        q = [np.clip(RegularGridInterpolator((glon, glat), P[k], bounds_error=False)(pts), 0, None) for k in range(3)]
        ax0 = fig.add_subplot(gs[0, j])
        ax0.plot(dist, gi(np.column_stack([lat, lon])), color=INK, lw=1)
        ax0.axhline(0, color=INK2, lw=0.5)
        ax0.set_ylabel("Residual (mGal)" if j == 0 else "")
        ax0.set_title(f"({'ab'[j]}) {name}: residual Bouguer anomaly", loc="left")
        ax0.tick_params(labelbottom=False)
        clean(ax0)
        ax = fig.add_subplot(gs[1, j], sharex=ax0)
        ok = np.isfinite(q[1])
        top = 0.0
        ax.fill_between(dist, top, np.where(ok, q[1], np.nan), color=SED, lw=0, label="Sediment fill (P50)")
        ax.fill_between(dist, np.where(ok, q[1], np.nan), 13, color=BASE, alpha=0.35, lw=0, label="Basement")
        ax.fill_between(dist, q[0], q[2], color=BLUE, alpha=0.3, lw=0, label="Basement depth P10–P90")
        ax.plot(dist, q[1], color=BLUE, lw=1.4)
        # magnetic windows centred within 0.5 degrees of the line
        for r in mags:
            dl = np.hypot((lon - r["lon"]) * np.cos(np.radians(lat)), lat - r["lat"])
            if dl.min() <= 0.5:
                ax.errorbar(dist[dl.argmin()], r["shallow_km"], yerr=r["shallow_se_km"], fmt="o", ms=4,
                            color=ORANGE, mec=INK, mew=0.4, elinewidth=1,
                            label="Magnetic spectral depth (3° window)")
        for w_, (wname, td) in enumerate(WELLS):
            ax.axhline(td, color=INK2, lw=0.6, ls=(0, (4, 2)))
            right = w_ % 2 == 0
            ax.text(dist[-1] if right else dist[0] + 0.02 * dist[-1], td - 0.08, f"{wname} TD {td:.2f} km",
                    fontsize=5.5, ha="right" if right else "left", va="bottom", color=INK2)
        at0 = ok & (q[1] <= 0.05)
        if at0.any():
            ax.annotate("basement at or\nnear the surface", (dist[at0].mean(), 0.1), xytext=(dist[at0].mean(), 3.0),
                        fontsize=6, ha="center", color=INK, arrowprops=dict(arrowstyle="->", lw=0.6, color=INK))
        ax.set_ylim(13, 0)
        ax.set_xlim(dist[0], dist[-1])
        ax.set_xlabel("Distance along the line (km)")
        ax.set_ylabel("Depth below ground (km)" if j == 0 else "")
        ax.set_title(f"({'cd'[j]}) Depth to basement", loc="left")
        h, lab = ax.get_legend_handles_labels()
        uniq = dict(zip(lab, h))
        if j == 0:
            ax.legend(uniq.values(), uniq.keys(), frameon=True, framealpha=0.9, edgecolor="none",
                      loc="lower left", fontsize=6)
        clean(ax)
        out[name] = dict(length_km=float(dist[-1]), depth_p50_range=[float(np.nanmin(q[1])), float(np.nanmax(q[1]))],
                         halfwidth_median=float(np.nanmedian((q[2] - q[0]) / 2)))
    fig.text(0.5, -0.02, "Dashed levels: total depths of the four deep wells, all of which ended in sediments; "
             "their coordinates are not public, so they are lower bounds, not located data.",
             ha="center", fontsize=6.5, color=INK2)
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"field_profiles_cuvette.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    s = {"katanga": katanga(), "cuvette": cuvette()}
    (RES / "profiles.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
