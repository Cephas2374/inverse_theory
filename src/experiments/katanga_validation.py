"""Katanga: sub-district validation against USGS tonnages, geology-matched backgrounds and
the choice ensemble (paper Section 3.5.1).

Sub-districts are Louvain communities (15 km links) of the Cu-Co occurrences with at least
7 sites. Each is compared with the contained copper of the USGS Central African Copperbelt
deposits (Taylor et al., 2013; Zientek et al., 2014) that lie within 10 km of its sites.
Backgrounds: (i) the province, cells farther than 30 km from any occurrence; (ii) the
geology-matched Roan Group, cells of the USGS permissive tracts farther than 10 km from
any occurrence or USGS site. Inputs: results/field_v2/katanga_runs/*.npz.
Outputs: results/field_v2/katanga_validation.json, figure and table.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from shapely import contains_xy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from drcinv import TensorMesh  # noqa: E402

QUICK = "--quick" in sys.argv
RES = ROOT / "results" / ("field_v2_quick" if QUICK else "field_v2")
FIG = RES if QUICK else ROOT / "paper" / "figures"
GEN = RES if QUICK else ROOT / "paper" / "sections" / "generated"
USGS = ROOT / "data" / "raw" / "usgs_copper" / "sir2010-5090T_GISdata" / "sir2010-5090-T_GIS"
K = 111195.0
SLICE_0_3 = slice(0, 3)          # first three slices: 0-1, 1-2, 2-3 km
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42})
MM = 1 / 25.4


def load_runs(name="katanga"):
    runs = []
    for p in sorted((RES / f"{name}_runs").glob("*.npz")):
        z = np.load(p)
        runs.append(dict(meta=json.loads(str(z["meta"])), slices=z["slices"], map=z["map"],
                         mesh=z["mesh"], lon0=float(z["lon0"]), lat0=float(z["lat0"])))
    return runs


def cell_centres(mesh_arr, lon0, lat0):
    m = TensorMesh(*mesh_arr[:6], *map(int, mesh_arr[6:]))
    xc = m.x0 + m.dx * (np.arange(m.nx) + 0.5)
    yc = m.y0 + m.dy * (np.arange(m.ny) + 0.5)
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    return m, X, Y, X / (K * np.cos(np.radians(lat0))) + lon0, Y / K + lat0


def sub_districts(cu, link=15.0, min_sites=7):
    xy = np.column_stack([cu.lon * K / 1000 * np.cos(np.radians(cu.lat)), cu.lat * K / 1000])
    pairs = cKDTree(xy).query_pairs(link, output_type="ndarray")
    G = nx.Graph()
    G.add_nodes_from(range(len(cu)))
    d = np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)
    G.add_weighted_edges_from((int(a), int(b), float(1 - di / link)) for (a, b), di in zip(pairs, d))
    comms = sorted(nx.community.louvain_communities(G, weight="weight", seed=7), key=len, reverse=True)
    label = np.full(len(cu), -1)
    for i, c in enumerate(c for c in comms if len(c) >= min_sites):
        label[list(c)] = i
    return label, float(nx.community.modularity(G, comms, weight="weight"))


def main():
    for p in (FIG, GEN):
        p.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    r0 = runs[0]
    mesh, X, Y, CLON, CLAT = cell_centres(r0["mesh"], r0["lon0"], r0["lat0"])
    lon0, lat0 = r0["lon0"], r0["lat0"]
    area = mesh.dx * mesh.dy / 1e6
    dep = gpd.read_file(ROOT / "data" / "processed" / "deposits.gpkg")
    w, e, s, n = 24.6, 28.4, -12.4, -9.8
    cu = dep[(dep.group == "Cu-Co") & dep.lon.between(w, e) & dep.lat.between(s, n)].reset_index(drop=True)
    label, Q = sub_districts(cu)
    cx = (cu.lon.values - lon0) * K * np.cos(np.radians(lat0))
    cy = (cu.lat.values - lat0) * K
    tree = cKDTree(np.column_stack([cx, cy]))
    dist, near = tree.query(np.column_stack([X.ravel(), Y.ravel()]))
    dist, near = dist.reshape(X.shape), near.reshape(X.shape)
    sub = np.where((dist <= 10_000) & (label[near] >= 0), label[near], -1)
    nsub = label.max() + 1

    usgs = gpd.read_file(USGS / "CACB_deposits_prospects.shp")
    usgs = usgs[usgs.Country.str.contains("Congo") & (usgs.Tonnage_Mt > 0)].copy()
    usgs["cu_mt"] = usgs.Tonnage_Mt * usgs.Cu_pct / 100
    usgs["co_mt"] = usgs.Tonnage_Mt * usgs.Co_pct.fillna(0) / 100
    ux = (usgs.Longitude.values - lon0) * K * np.cos(np.radians(lat0))
    uy = (usgs.Latitude.values - lat0) * K
    du, nu = tree.query(np.column_stack([ux, uy]))
    usgs["sub"] = np.where((du <= 10_000) & (label[nu] >= 0), label[nu], -1)
    usgs_all = gpd.read_file(USGS / "CACB_deposits_prospects.shp")
    ax_, ay_ = ((usgs_all.Longitude.values - lon0) * K * np.cos(np.radians(lat0)),
                (usgs_all.Latitude.values - lat0) * K)
    d_usgs = cKDTree(np.column_stack([ax_, ay_])).query(np.column_stack([X.ravel(), Y.ravel()]))[0].reshape(X.shape)

    adm1 = gpd.read_file(ROOT / "data" / "raw" / "boundaries" / "COD_ADM1.geojson").to_crs(4326)
    prov = contains_xy(adm1[adm1.shapeName.isin(["Haut-Katanga", "Lualaba"])].union_all(), CLON, CLAT)
    tracts = gpd.read_file(USGS / "CACB_permissive_tracts.shp").to_crs(4326)
    in_tract = contains_xy(tracts.union_all(), CLON, CLAT)
    district = prov & (dist <= 10_000)
    bg_prov = prov & (dist > 30_000)
    bg_geo = prov & in_tract & (dist > 10_000) & (d_usgs > 10_000)
    print(f"sub-districts {nsub} (Q={Q:.2f}); cells: district {district.sum()}, province bg {bg_prov.sum()}, "
          f"Roan bg {bg_geo.sum()}", flush=True)

    # names: the largest USGS deposit of each sub-district
    names, cu_sub, co_sub, nd_sub, area_sub, centre = [], [], [], [], [], []
    for k in range(nsub):
        u = usgs[usgs["sub"] == k]
        names.append(u.sort_values("cu_mt").Name.iloc[-1] if len(u) else f"SD{k + 1}")
        cu_sub.append(float(u.cu_mt.sum()))
        co_sub.append(float(u.co_mt.sum()))
        nd_sub.append(int(len(u)))
        area_sub.append(float((sub == k).sum() * area))
        centre.append((float(cu.lon[label == k].mean()), float(cu.lat[label == k].mean())))

    def per_run(r):
        M = r["slices"][:, SLICE_0_3].sum(1)                     # (n_s, nx, ny) Mt in 0-3 km
        bg_p = M[:, bg_prov].sum(1) / (bg_prov.sum() * area)       # Mt/km2
        bg_g = M[:, bg_geo].sum(1) / (bg_geo.sum() * area)
        tot = M[:, district].sum(1)
        A_d = district.sum() * area
        subm = np.stack([M[:, sub == k].sum(1) for k in range(nsub)], 1)          # (n_s, nsub)
        sub_ex = subm - bg_g[:, None] * np.array(area_sub)[None]
        return dict(district=tot, excess_prov=tot - bg_p * A_d, excess_geo=tot - bg_g * A_d,
                    bg_prov=bg_p, bg_geo=bg_g, sub=subm, sub_ex=sub_ex,
                    district_density=tot / A_d)

    out = {"n_subdistricts": int(nsub), "louvain_Q": Q, "names": names, "cu_mt": cu_sub, "co_mt": co_sub,
           "n_usgs": nd_sub, "area_km2": area_sub, "centre": centre,
           "cells": {"district": int(district.sum()), "bg_province": int(bg_prov.sum()),
                     "bg_roan": int(bg_geo.sum())}, "runs": {}}
    pooled = {k: [] for k in ("excess_prov", "excess_geo", "district")}
    rho_runs = []
    sub_pool = []
    for r in runs:
        m = r["meta"]
        q = per_run(r)
        rs = [stats.spearmanr(q["sub_ex"][i], cu_sub)[0] for i in range(len(q["sub_ex"]))]
        out["runs"][m["tag"]] = dict(meta=m, **{k: np.percentile(q[k], [10, 50, 90]).tolist()
                                                for k in ("district", "excess_prov", "excess_geo",
                                                          "bg_prov", "bg_geo", "district_density")},
                                     sub_ex_p50=np.median(q["sub_ex"], 0).tolist(),
                                     spearman_p50=float(np.median(rs)))
        if m["tag"].startswith("ens_"):
            # each run weighted equally: resample to 10 values per run
            idx = np.linspace(0, len(q["district"]) - 1, 10).astype(int)
            for k in pooled:
                pooled[k].append(q[k][idx])
            sub_pool.append(q["sub_ex"][idx])
            rho_runs.append(np.median(rs))
    if pooled["district"]:
        out["ensemble"] = {k: np.percentile(np.concatenate(v), [10, 50, 90]).tolist() for k, v in pooled.items()}
        S = np.concatenate(sub_pool)
        out["ensemble"]["sub_ex_p10_p50_p90"] = np.percentile(S, [10, 50, 90], 0).tolist()
        out["ensemble"]["spearman_runs"] = [float(x) for x in rho_runs]
        out["ensemble"]["frac_runs_positive"] = float(np.mean(np.array(rho_runs) > 0))
    main = per_run(next(r for r in runs if r["meta"]["tag"] == "main_B"))
    p50 = np.median(main["sub_ex"], 0)
    rho, p = stats.spearmanr(p50, cu_sub)
    # exact permutation p-value of Spearman's rho for small n
    rng = np.random.default_rng(0)
    perm = np.array([stats.spearmanr(p50, rng.permutation(cu_sub))[0] for _ in range(20000)])
    out["main_B"] = dict(spearman=float(rho), p_asymptotic=float(p),
                         p_permutation=float(np.mean(perm >= rho)),
                         pearson_log=float(np.corrcoef(p50, np.log10(np.array(cu_sub) + 0.01))[0, 1]),
                         sub_ex_p10_p50_p90=np.percentile(main["sub_ex"], [10, 50, 90], 0).tolist(),
                         per_area_spearman=float(stats.spearmanr(p50 / np.array(area_sub),
                                                                 np.array(cu_sub) / np.array(area_sub))[0]))
    (RES / "katanga_validation.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1)[:4000])
    figure(out, runs, mesh, CLON, CLAT, sub, usgs, cu, label, bg_geo)
    table(out)
    run_table(out, runs)


def num(x, nd=2):
    return f"${x:.{nd}f}$".replace("-", "-")


def f(v, scale=1e6, nd=2):
    """P50 (P10--P90) in math mode, values divided by scale."""
    return f"{num(v[1] / scale, nd)} ({num(v[0] / scale, nd)} to {num(v[2] / scale, nd)})"


LOCALITY = {"Kansalawile-Mambilima": "Tenke-Fungurume", "Kamoto-KOV-Musonie-Mupine": "Kolwezi",
            "Kambove Principal-Kambove West": "Kambove-Kakanda", "Mutanda-Ya-Mukonkota": "Mutanda",
            "Kalongwe": "Kalongwe (W Kolwezi)", "Shituru": "Likasi", "Luishia": "Luishia",
            "Ruashi": "Lubumbashi"}


def run_table(out, runs):
    """Katanga run summary (replaces the version-1 table)."""
    lab = {"main_A": "Tikhonov", "main_B": "$\\ell_2$ fabric graph", "main_G": "Sparse graph",
           "eigen_B": "$\\ell_2$ fabric graph", "h6_B": "$\\ell_2$ fabric graph", "h6_B_2670": "$\\ell_2$ fabric graph",
           "rho2670_B": "$\\ell_2$ fabric graph"}
    rows = []
    for t in ["main_A", "main_B", "main_G", "eigen_B", "h6_B", "h6_B_2670", "rho2670_B"]:
        r = out["runs"][t]
        m = r["meta"]
        rows.append(f"{lab[t]} & {'XGM2019e' if m['model'] == 'xgm2019e' else 'EIGEN-6C4'} & {m['h_obs'] / 1000:.1f} & "
                    f"{m['rho']:.0f} & {m['chi2']:.2f} & {f(r['excess_prov'])} & {f(r['excess_geo'])} & "
                    f"{num(r['spearman_p50'])} \\\\")
    e = out["ensemble"]
    rows.append(f"Choice ensemble (24 runs) & both & 2.1--6 & var. & 1.00 & {f(e['excess_prov'])} & "
                f"{f(e['excess_geo'])} & $\\leq{num(max(e['spearman_runs'])).strip('$')}$ \\\\")
    c = out["cells"]
    (GEN / "tab_field_katanga.tex").write_text(
        "\\begin{table*}[t]\n\\centering\\small\n\\caption{Katanga endowment inversion. District excess mass in "
        f"0--3~km (Tt, P50 with P10--P90) over the province background ({c['bg_province'] * 25} km$^2$ farther than "
        f"30~km from any occurrence) and over the geology-matched Roan Group background ({c['bg_roan'] * 25}~km$^2$ of "
        "the USGS permissive tracts away from known sites), and the Spearman correlation between sub-district excess "
        "and USGS contained copper. $h$: observation height; $\\rho_B$: terrain reduction density.}\\label{tab:fieldkatanga}\n"
        "\\begin{tabular*}{\\tblwidth}{@{\\extracolsep{\\fill}}llrrrllr@{}}\n\\toprule\n"
        "Theory & Gravity model & $h$ (km) & $\\rho_B$ & $\\chi^2/N$ & Excess, province (Tt) & Excess, Roan (Tt) & "
        "$\\rho_{\\text{Cu}}$ \\\\\n\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular*}\n\\end{table*}\n", encoding="utf-8")


def figure(out, runs, mesh, CLON, CLAT, sub, usgs, cu, label, bg_geo):
    adm0 = gpd.read_file(ROOT / "data" / "raw" / "boundaries" / "COD_ADM0.geojson").to_crs(4326)
    adm1 = gpd.read_file(ROOT / "data" / "raw" / "boundaries" / "COD_ADM1.geojson").to_crs(4326)
    tracts = gpd.read_file(USGS / "CACB_permissive_tracts.shp").to_crs(4326)
    fig = plt.figure(figsize=(190 * MM, 165 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 0.85], hspace=0.42, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    rB = next(r for r in runs if r["meta"]["tag"] == "main_B")
    M = np.median(rB["slices"][:, SLICE_0_3].sum(1), 0) / (mesh.dx * mesh.dy / 1e6)
    lim = np.percentile(np.abs(M), 98)
    ext = (CLON[0, 0] - 0.0225, CLON[-1, 0] + 0.0225, CLAT[0, 0] - 0.0225, CLAT[0, -1] + 0.0225)
    im = ax.imshow(M.T, origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    tracts.boundary.plot(ax=ax, color="#7a5c00", lw=0.6)
    adm1.boundary.plot(ax=ax, color=INK2, lw=0.3)
    adm0.boundary.plot(ax=ax, color=INK, lw=0.7)
    ax.contour(CLON[:, 0], CLAT[0], (sub >= 0).T.astype(float), levels=[0.5], colors=INK, linewidths=0.6)
    sz = 4 + 10 * np.sqrt(usgs.cu_mt.values)
    ax.scatter(usgs.Longitude, usgs.Latitude, s=sz, facecolors="none", edgecolors=INK, linewidths=0.6)
    for k, (lo, la) in enumerate(out["centre"]):
        ax.text(lo, la + 0.13, str(k + 1), fontsize=7.5, weight="bold", ha="center", color=INK)
    ax.set_xlim(24.6, 28.4)
    ax.set_ylim(-12.4, -9.8)
    ax.set_xticks(np.arange(25, 29, 1))
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}°E")
    ax.yaxis.set_major_formatter(lambda v, _: f"{abs(v):g}°S")
    ax.set_title("(a) $\\ell_2$ fabric-graph mass, sub-districts,\nUSGS deposits (∝ √Cu), Roan tracts (brown)", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.1, aspect=30, orientation="horizontal", label="Mt/km$^2$ (0–3 km)")

    ax = fig.add_subplot(gs[0, 1])
    q = np.array(out["main_B"]["sub_ex_p10_p50_p90"]) / 1000          # Gt
    cuv = np.array(out["cu_mt"])
    ax.errorbar(cuv, q[1], yerr=[q[1] - q[0], q[2] - q[1]], fmt="o", ms=4, color="#2a78d6", elinewidth=1)
    if "ensemble" in out:
        e = np.array(out["ensemble"]["sub_ex_p10_p50_p90"]) / 1000
        ax.errorbar(cuv * 1.06, e[1], yerr=[e[1] - e[0], e[2] - e[1]], fmt="s", ms=3, color="#9a9893",
                    elinewidth=0.8, label="choice ensemble")
    for k, (x, y) in enumerate(zip(cuv, q[1])):
        ax.text(x * 1.12, y, str(k + 1), fontsize=7, color=INK, va="center")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("USGS contained Cu (Mt)")
    ax.set_ylabel("Excess mass over Roan bg (Gt)")
    mb = out["main_B"]
    ax.set_title(f"(b) Spearman {mb['spearman']:.2f} (permutation p = {mb['p_permutation']:.2f})", loc="left")
    ax.grid(color=GRID, lw=0.5)
    ax.legend(frameon=False, loc="upper left")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)

    ax = fig.add_subplot(gs[1, :])
    tags = [t for t in ["main_A", "main_B", "main_G", "eigen_B", "h6_B", "h6_B_2670", "rho2670_B"] if t in out["runs"]]
    lab = {"main_A": "Tikhonov", "main_B": "$\\ell_2$ graph", "main_G": "Sparse graph", "eigen_B": "EIGEN-6C4",
           "h6_B": "6 km, Nettleton", "h6_B_2670": "6 km, 2670", "rho2670_B": "2.1 km, 2670"}
    y = np.arange(len(tags))[::-1]
    for yy, t in zip(y, tags):
        for key, col, off in [("excess_prov", "#9a9893", 0.15), ("excess_geo", "#2a78d6", -0.15)]:
            v = np.array(out["runs"][t][key]) / 1000
            ax.errorbar(v[1], yy + off, xerr=[[v[1] - v[0]], [v[2] - v[1]]], fmt="o", ms=3, color=col, elinewidth=1)
    if "ensemble" in out:
        for key, col, off in [("excess_prov", "#9a9893", 0.15), ("excess_geo", "#2a78d6", -0.15)]:
            v = np.array(out["ensemble"][key]) / 1000
            ax.errorbar(v[1], -1 + off, xerr=[[v[1] - v[0]], [v[2] - v[1]]], fmt="D", ms=3.5, color=col,
                        elinewidth=1.4)
        y = np.append(y, -1)
        tags = tags + ["ensemble"]
        lab["ensemble"] = "Choice ensemble"
    ax.set_yticks(y, [lab[t] for t in tags])
    ax.set_xlabel("District excess mass, 0–3 km (Gt)")
    ax.set_title("(c) Excess over province (grey) and Roan (blue) background", loc="left")
    ax.grid(axis="x", color=GRID, lw=0.5)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for ext_ in ("pdf", "png"):
        fig.savefig(FIG / f"field_katanga_validation.{ext_}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def table(out):
    q = np.array(out["main_B"]["sub_ex_p10_p50_p90"]) / 1000
    order = np.argsort(out["cu_mt"])[::-1]
    rows = []
    for k in order:
        rows.append(f"{k + 1} & {LOCALITY.get(out['names'][k], out['names'][k])} & {out['n_usgs'][k]} & "
                    f"{out['area_km2'][k]:.0f} & {out['cu_mt'][k]:.1f} & {out['co_mt'][k]:.2f} & "
                    f"{num(q[1, k], 0)} ({num(q[0, k], 0)} to {num(q[2, k], 0)}) \\\\")
    mb = out["main_B"]
    (GEN / "tab_katanga_validation.tex").write_text(
        "\\begin{table*}[t]\n\\centering\\small\n\\caption{Katanga sub-districts (Louvain communities of the "
        f"Cu--Co occurrences, $Q={out['louvain_Q']:.2f}$), the copper and cobalt contained in the USGS deposits "
        "within 10~km of their sites \\citep{taylor2013cacb}, and the excess mass over the Roan Group background "
        "in 0--3~km from the $\\ell_2$ fabric-graph model (P50 with P10--P90, 40 RTO samples). Spearman rank "
        f"correlation between excess mass and contained Cu: {num(mb['spearman'])} (permutation $p={mb['p_permutation']:.2f}$)."
        "}\\label{tab:katangaval}\n"
        "\\begin{tabular*}{\\tblwidth}{@{\\extracolsep{\\fill}}rlrrrrl@{}}\n\\toprule\n"
        "\\# & Sub-district & USGS deposits & Area (km$^2$) & Cu (Mt) & Co (Mt) & Excess mass (Gt) \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table*}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
