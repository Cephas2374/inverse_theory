"""Zambian Copperbelt: repeat of the Katanga validation (paper Section 3.5.1).

Same inversion design (field_v2.py zambia) and the same tests:
  * sub-districts = Louvain communities (15 km links, >= 3 sites) of the USGS deposits and
    prospects (no MRDS/IPIS/OSM occurrence layer exists for Zambia in this study);
  * contained copper of the USGS deposits within 10 km of each sub-district;
  * backgrounds: Zambian cells farther than 30 km from any USGS site (province) and cells of
    the Roan permissive tracts farther than 10 km from any site (geology-matched).
A pooled test ranks the Katanga and Zambian sub-districts together, using the excess per
unit area over each side's own Roan background.
Inputs: results/field_v2/zambia_runs, katanga_validation.json. Outputs: zambia_validation.json,
figure field_zambia_validation, table tab_zambia_validation.
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
from scipy.spatial import cKDTree
from shapely import contains_xy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "experiments"))
from katanga_validation import (FIG, GEN, GRID, INK, INK2, K, MM, RES, SLICE_0_3, USGS,  # noqa: E402
                                cell_centres, load_runs, sub_districts)

LOCALITY = {"Konkola-Musoshi": "Konkola", "Chingola-Nchanga": "Chingola", "Mindola-Nkana N-S": "Kitwe (Nkana)",
            "Baluba-Muliashi-Luanshya": "Luanshya", "Mufulira": "Mufulira", "Chambishi Southeast": "Chambishi",
            "Ndola East": "Ndola", "Itawa": "Ndola", "Bwana Mkubwa": "Ndola", "Sebembere": "Sebembere"}


def main():
    runs = load_runs("zambia")
    r0 = next(r for r in runs if r["meta"]["tag"] == "main_B")
    mesh, X, Y, CLON, CLAT = cell_centres(r0["mesh"], r0["lon0"], r0["lat0"])
    lon0, lat0 = r0["lon0"], r0["lat0"]
    area = mesh.dx * mesh.dy / 1e6
    allu = gpd.read_file(USGS / "CACB_deposits_prospects.shp")
    z = allu[allu.Country.str.contains("Zambia")].reset_index(drop=True)
    z = z.assign(lon=z.Longitude, lat=z.Latitude, cu_mt=z.Tonnage_Mt * z.Cu_pct / 100)
    label, Q = sub_districts(z, min_sites=3)
    nsub = label.max() + 1
    zx = (z.lon.values - lon0) * K * np.cos(np.radians(lat0))
    zy = (z.lat.values - lat0) * K
    tree = cKDTree(np.column_stack([zx, zy]))
    dist, near = tree.query(np.column_stack([X.ravel(), Y.ravel()]))
    dist, near = dist.reshape(X.shape), near.reshape(X.shape)
    sub = np.where((dist <= 10_000) & (label[near] >= 0), label[near], -1)
    ax_ = (allu.Longitude.values - lon0) * K * np.cos(np.radians(lat0))
    ay_ = (allu.Latitude.values - lat0) * K
    d_all = cKDTree(np.column_stack([ax_, ay_])).query(np.column_stack([X.ravel(), Y.ravel()]))[0].reshape(X.shape)
    adm0 = gpd.read_file(ROOT / "data" / "raw" / "boundaries" / "COD_ADM0.geojson").to_crs(4326)
    zambia = ~contains_xy(adm0.union_all(), CLON, CLAT)
    tracts = gpd.read_file(USGS / "CACB_permissive_tracts.shp").to_crs(4326)
    in_tract = contains_xy(tracts.union_all(), CLON, CLAT)
    district = zambia & (dist <= 10_000)
    bg_prov = zambia & (d_all > 30_000)
    bg_geo = zambia & in_tract & (d_all > 10_000)
    names, cu_sub, area_sub, centre, n_dep = [], [], [], [], []
    for k in range(nsub):
        u = z[(label == k) & (z.Tonnage_Mt > 0)]
        names.append(u.sort_values("cu_mt").Name.iloc[-1] if len(u) else f"SD{k + 1}")
        cu_sub.append(float(u.cu_mt.sum()))
        n_dep.append(int(len(u)))
        area_sub.append(float((sub == k).sum() * area))
        centre.append((float(z.lon[label == k].mean()), float(z.lat[label == k].mean())))
    print(f"Zambia: {nsub} sub-districts (Q={Q:.2f}); cells district {district.sum()}, province {bg_prov.sum()}, "
          f"Roan {bg_geo.sum()}", flush=True)

    def per_run(r):
        M = r["slices"][:, SLICE_0_3].sum(1)
        bg_p = M[:, bg_prov].sum(1) / (bg_prov.sum() * area)
        bg_g = M[:, bg_geo].sum(1) / (bg_geo.sum() * area)
        tot = M[:, district].sum(1)
        A_d = district.sum() * area
        subm = np.stack([M[:, sub == k].sum(1) for k in range(nsub)], 1)
        return dict(district=tot, excess_prov=tot - bg_p * A_d, excess_geo=tot - bg_g * A_d, bg_prov=bg_p,
                    bg_geo=bg_g, district_density=tot / A_d, sub_ex=subm - bg_g[:, None] * np.array(area_sub)[None])

    out = {"n_subdistricts": int(nsub), "louvain_Q": Q, "names": names, "cu_mt": cu_sub, "n_usgs": n_dep,
           "area_km2": area_sub, "centre": centre,
           "cells": {"district": int(district.sum()), "bg_province": int(bg_prov.sum()), "bg_roan": int(bg_geo.sum())},
           "runs": {}}
    pooled = {k: [] for k in ("excess_prov", "excess_geo", "district")}
    rho_runs, sub_pool = [], []
    for r in runs:
        q = per_run(r)
        rs = [stats.spearmanr(q["sub_ex"][i], cu_sub)[0] for i in range(len(q["sub_ex"]))]
        out["runs"][r["meta"]["tag"]] = dict(meta=r["meta"], spearman_p50=float(np.median(rs)),
                                             sub_ex_p50=np.median(q["sub_ex"], 0).tolist(),
                                             **{k: np.percentile(q[k], [10, 50, 90]).tolist()
                                                for k in ("district", "excess_prov", "excess_geo", "bg_prov",
                                                          "bg_geo", "district_density")})
        if r["meta"]["tag"].startswith("ens_"):
            idx = np.linspace(0, len(q["district"]) - 1, 10).astype(int)
            for k in pooled:
                pooled[k].append(q[k][idx])
            sub_pool.append(q["sub_ex"][idx])
            rho_runs.append(np.median(rs))
    if pooled["district"]:
        out["ensemble"] = {k: np.percentile(np.concatenate(v), [10, 50, 90]).tolist() for k, v in pooled.items()}
        out["ensemble"]["sub_ex_p10_p50_p90"] = np.percentile(np.concatenate(sub_pool), [10, 50, 90], 0).tolist()
        out["ensemble"]["spearman_runs"] = [float(x) for x in rho_runs]
    qm = per_run(r0)
    p50 = np.median(qm["sub_ex"], 0)
    rho = stats.spearmanr(p50, cu_sub)[0]
    rng = np.random.default_rng(0)
    perm = np.array([stats.spearmanr(p50, rng.permutation(cu_sub))[0] for _ in range(20000)])
    out["main_B"] = dict(spearman=float(rho), p_permutation=float(np.mean(perm >= rho)),
                         sub_ex_p10_p50_p90=np.percentile(qm["sub_ex"], [10, 50, 90], 0).tolist())
    # pooled Katanga + Zambia test on per-area excess
    kv = json.loads((RES / "katanga_validation.json").read_text())
    k_ex = np.array(kv["main_B"]["sub_ex_p10_p50_p90"][1]) / np.array(kv["area_km2"])
    z_ex = p50 / np.array(area_sub)
    ex = np.concatenate([k_ex, z_ex])
    cu_all = np.concatenate([np.array(kv["cu_mt"]) / np.array(kv["area_km2"]), np.array(cu_sub) / np.array(area_sub)])
    cu_tot = np.concatenate([kv["cu_mt"], cu_sub])
    rho_a = stats.spearmanr(ex, cu_all)[0]
    rho_t = stats.spearmanr(np.concatenate([np.array(kv["main_B"]["sub_ex_p10_p50_p90"][1]), p50]), cu_tot)[0]
    perm_a = np.array([stats.spearmanr(ex, rng.permutation(cu_all))[0] for _ in range(20000)])
    out["pooled"] = dict(n=int(len(ex)), spearman_per_area=float(rho_a), p_per_area=float(np.mean(perm_a >= rho_a)),
                         spearman_totals=float(rho_t))
    (RES / "zambia_validation.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1)[:3000])
    figure(out, r0, mesh, CLON, CLAT, sub, z, tracts, kv, k_ex, z_ex)
    table(out)


def figure(out, r0, mesh, CLON, CLAT, sub, z, tracts, kv, k_ex, z_ex):
    adm0 = gpd.read_file(ROOT / "data" / "raw" / "boundaries" / "COD_ADM0.geojson").to_crs(4326)
    fig = plt.figure(figsize=(190 * MM, 82 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    M = np.median(r0["slices"][:, SLICE_0_3].sum(1), 0) / (mesh.dx * mesh.dy / 1e6)
    lim = np.percentile(np.abs(M), 98)
    d = 0.0225
    im = ax.imshow(M.T, origin="lower", extent=(CLON[0, 0] - d, CLON[-1, 0] + d, CLAT[0, 0] - d, CLAT[0, -1] + d),
                   cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    tracts.boundary.plot(ax=ax, color="#7a5c00", lw=0.6)
    adm0.boundary.plot(ax=ax, color=INK, lw=0.8)
    ax.contour(CLON[:, 0], CLAT[0], (sub >= 0).T.astype(float), levels=[0.5], colors=INK, linewidths=0.6)
    zt = z[z.Tonnage_Mt > 0]
    ax.scatter(zt.lon, zt.lat, s=4 + 10 * np.sqrt(zt.cu_mt), facecolors="none", edgecolors=INK, linewidths=0.6)
    for k, (lo, la) in enumerate(out["centre"]):
        ax.text(lo + 0.12, la, str(k + 1), fontsize=7.5, weight="bold", color=INK)
    ax.set_xlim(26.8, 29.8)
    ax.set_ylim(-14.6, -11.6)
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}°E")
    ax.yaxis.set_major_formatter(lambda v, _: f"{abs(v):g}°S")
    ax.set_title("(a) Zambian Copperbelt: mass 0–3 km, sub-districts,\nUSGS deposits (∝ √Cu), Roan tracts (brown)",
                 loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Mt/km$^2$")
    ax = fig.add_subplot(gs[1])
    kc = np.array(kv["cu_mt"]) / np.array(kv["area_km2"]) * 1000
    zc = np.array(out["cu_mt"]) / np.array(out["area_km2"]) * 1000
    ax.scatter(kc, k_ex, s=22, color="#2a78d6", label="Katanga sub-districts", zorder=3)
    ax.scatter(zc, z_ex, s=22, color="#eb6834", marker="s", label="Zambian sub-districts", zorder=3)
    ax.set_xscale("symlog", linthresh=1)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlabel("USGS contained Cu per area (kt/km$^2$)")
    ax.set_ylabel("Excess mass over Roan bg (Mt/km$^2$)")
    p = out["pooled"]
    ax.set_title(f"(b) Both belts: Spearman {p['spearman_per_area']:.2f} (n = {p['n']})", loc="left")
    ax.legend(frameon=False, loc="best")
    ax.grid(color=GRID, lw=0.5)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"field_zambia_validation.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def table(out):
    def num(x, nd=0):
        return f"${x:.{nd}f}$"
    q = np.array(out["main_B"]["sub_ex_p10_p50_p90"]) / 1000
    rows = []
    for k in np.argsort(out["cu_mt"])[::-1]:
        nm = out["names"][k]
        rows.append(f"{k + 1} & {LOCALITY.get(nm, nm)} & {out['n_usgs'][k]} & {out['area_km2'][k]:.0f} & "
                    f"{out['cu_mt'][k]:.1f} & {num(q[1, k])} ({num(q[0, k])} to {num(q[2, k])}) \\\\")
    r = out["runs"]["main_B"]
    e = out.get("ensemble", {})
    tt = lambda v: f"${v[1] / 1e6:.2f}$ (${v[0] / 1e6:.2f}$ to ${v[2] / 1e6:.2f}$)"  # noqa: E731
    cap = (f"Zambian Copperbelt validation. Sub-districts are Louvain communities of the USGS sites "
           f"($Q={out['louvain_Q']:.2f}$); copper from \\citet{{taylor2013cacb}}; excess mass over the Roan background "
           f"in 0--3~km from the $\\ell_2$ fabric-graph model (Gt, P50 with P10--P90). District excess over the "
           f"province: {tt(r['excess_prov'])}~Tt (main run), {tt(e['excess_prov']) if e else '--'}~Tt (ensemble); over "
           f"the Roan background: {tt(r['excess_geo'])}~Tt and {tt(e['excess_geo']) if e else '--'}~Tt. Spearman "
           f"correlation with contained Cu: ${out['main_B']['spearman']:.2f}$ (permutation $p={out['main_B']['p_permutation']:.2f}$).")
    (GEN / "tab_zambia_validation.tex").write_text(
        "\\begin{table*}[t]\n\\centering\\small\n\\caption{" + cap + "}\\label{tab:zambiaval}\n"
        "\\begin{tabular*}{\\tblwidth}{@{\\extracolsep{\\fill}}rlrrrl@{}}\n\\toprule\n"
        "\\# & Sub-district & Dep. & Area (km$^2$) & Cu (Mt) & Excess (Gt) \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table*}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
