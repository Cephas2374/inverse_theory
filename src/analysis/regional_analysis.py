"""National-scale data analysis of the DRC (paper Section 3.4).

1. Geophysical signatures of each commodity group vs. the national background
   (medians, Mann-Whitney U, Kolmogorov-Smirnov).
2. Radially averaged power spectra of EMAG2 in three domains -> depth to magnetic sources
   (Spector & Grant, 1970), including magnetic basement under the Cuvette Centrale.
3. Mining districts as communities of a spatial deposit graph (Louvain modularity).
4. Prospectivity for Cu-Co, Au and 3T: logistic regression, random forest, gradient
   boosting and graph label spreading, scored by spatially blocked cross-validation.
Outputs: results/regional/*.json|nc|csv
"""
import json
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "regional"
FEATURES = ["bouguer", "bouguer_hgm", "magnetic", "mag_asa", "mag_tilt", "topography"]
KM_PER_DEG = 111.195
SEED = 7

# Spectral windows (lon_min, lon_max, lat_min, lat_max)
WINDOWS = {
    "Katanga Copperbelt": (24.0, 28.5, -12.5, -8.5),
    "Kibaran belt (Kivu-Maniema)": (26.0, 29.6, -5.0, -0.5),
    "Cuvette Centrale": (18.0, 23.5, -3.5, 1.5),
}

# Commodity study regions, limited to where the labels were collected (presence-background)
REGIONS = {"Cu-Co": ["Haut-Katanga", "Lualaba"],
           "Diamond": ["Kasai", "Kasai-Oriental", "Central Kasai", "Lomami", "Sankuru"],
           "Au": None, "3T": None}   # None -> within 50 km of an IPIS site


def grid_frame(ds):
    LON, LAT = np.meshgrid(ds.longitude.values, ds.latitude.values)
    df = pd.DataFrame({"lon": LON.ravel(), "lat": LAT.ravel(),
                       "inside": ds.inside.values.ravel()})
    for f in FEATURES:
        df[f] = ds[f].values.ravel()
    df["bouguer_hgm"] *= 1e3          # mGal/m -> mGal/km
    df["mag_asa"] *= 1e3              # nT/m -> nT/km
    return df


def cell_index(ds, lon, lat):
    step = float(ds.longitude[1] - ds.longitude[0])
    i = np.round((np.asarray(lat) - float(ds.latitude[0])) / step).astype(int)
    j = np.round((np.asarray(lon) - float(ds.longitude[0])) / step).astype(int)
    return i * ds.longitude.size + j


# ----------------------------------------------------------------------------------- 1
def signatures(df, dep, ds, adm1):
    """Deposit cells vs. the background of the same study region (not the whole country),
    so that province-scale differences (e.g. the Katanga plateau) are not mistaken for
    deposit signatures. Each feature uses only the cells where it is defined."""
    rows = []
    for g in ["Cu-Co", "Au", "3T", "Diamond"]:
        region = region_mask(df, dep, adm1, g)
        idx = np.unique(cell_index(ds, dep.lon[dep.group == g], dep.lat[dep.group == g]))
        is_dep = df.index.isin(idx)
        for f in FEATURES:
            sub = df.loc[region & is_dep, f].dropna()
            bg = df.loc[region & ~is_dep, f].dropna()
            if len(sub) < 5:
                continue
            u = stats.mannwhitneyu(sub, bg)
            ks = stats.ks_2samp(sub, bg)
            rows.append({"group": g, "feature": f, "n_cells": len(sub), "n_background": len(bg),
                         "median_group": float(sub.median()),
                         "median_background": float(bg.median()),
                         "mw_p": float(u.pvalue), "ks_D": float(ks.statistic),
                         "ks_p": float(ks.pvalue)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------- 2
def radial_spectrum(z, dx_km):
    z = z - np.nanmean(z)
    z = np.nan_to_num(z)
    ny, nx = z.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    z = z - np.polyval(np.polyfit(xx.ravel(), z.ravel(), 1), xx)      # remove linear trend in x
    z = z - np.polyval(np.polyfit(yy.ravel(), z.ravel(), 1), yy)      # and in y
    taper = np.outer(np.hanning(ny), np.hanning(nx))
    P = np.abs(np.fft.fftshift(np.fft.fft2(z * taper))) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(ny, dx_km)) * 2 * np.pi
    kx = np.fft.fftshift(np.fft.fftfreq(nx, dx_km)) * 2 * np.pi
    K = np.hypot(*np.meshgrid(kx, ky))
    edges = np.linspace(0, min(kx.max(), ky.max()), min(nx, ny) // 2)
    which = np.digitize(K.ravel(), edges)
    k = np.array([K.ravel()[which == b].mean() for b in range(1, len(edges))])
    p = np.array([P.ravel()[which == b].mean() for b in range(1, len(edges))])
    ok = np.isfinite(k) & (p > 0)
    return k[ok], np.log(p[ok])


def two_segment_fit(k, lnp, kmin=0.02):
    """Best two-segment linear fit of ln P(k); depth h = -slope / 2 (k in rad/km)."""
    m = k >= kmin
    k, lnp = k[m], lnp[m]
    best = None
    for b in range(4, len(k) - 4):
        f1 = stats.linregress(k[:b], lnp[:b])
        f2 = stats.linregress(k[b:], lnp[b:])
        sse = (np.sum((lnp[:b] - (f1.intercept + f1.slope * k[:b])) ** 2)
               + np.sum((lnp[b:] - (f2.intercept + f2.slope * k[b:])) ** 2))
        if best is None or sse < best[0]:
            best = (sse, b, f1, f2)
    _, b, f1, f2 = best
    return {"k_break": float(k[b]),
            "deep_depth_km": float(-f1.slope / 2), "deep_se_km": float(f1.stderr / 2),
            "shallow_depth_km": float(-f2.slope / 2), "shallow_se_km": float(f2.stderr / 2),
            "deep_fit": [float(f1.intercept), float(f1.slope)],
            "shallow_fit": [float(f2.intercept), float(f2.slope)], "k_min": float(k[0])}


def spectral_depths(ds_mag, ds):
    res, curves = {}, {}
    step = float(ds_mag.longitude[1] - ds_mag.longitude[0])
    for name, (w, e, s, n) in WINDOWS.items():
        win = ds_mag.sel(longitude=slice(w, e), latitude=slice(s, n))
        coverage = float(np.isfinite(win.values).mean())
        k, lnp = radial_spectrum(win.values, step * KM_PER_DEG)
        fit = two_segment_fit(k, lnp)
        elev = float(ds.topography.sel(longitude=slice(w, e), latitude=slice(s, n)).mean()) / 1000
        # EMAG2 UpCont is 4 km above the geoid: depth below ground = depth below sensor - (4 - elev)
        clearance = 4.0 - elev
        fit.update(coverage=coverage, mean_elevation_km=elev, sensor_clearance_km=clearance,
                   deep_depth_bg_km=fit["deep_depth_km"] - clearance,
                   shallow_depth_bg_km=fit["shallow_depth_km"] - clearance,
                   window=[w, e, s, n], size_km=[(e - w) * KM_PER_DEG, (n - s) * KM_PER_DEG])
        res[name] = fit
        curves[name] = {"k": k.tolist(), "lnp": lnp.tolist()}
    return res, curves


# ----------------------------------------------------------------------------------- 3
def districts(dep, link_km=15.0):
    """Graph: deposits are nodes, edges join sites closer than link_km (equirectangular km).
    Communities by Louvain modularity maximisation; districts = communities with >= 10 sites."""
    xy = np.column_stack([dep.lon * KM_PER_DEG * np.cos(np.radians(dep.lat)), dep.lat * KM_PER_DEG])
    pairs = cKDTree(xy).query_pairs(link_km, output_type="ndarray")
    G = nx.Graph()
    G.add_nodes_from(range(len(dep)))
    d = np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)
    G.add_weighted_edges_from((int(a), int(b), float(1 - di / link_km)) for (a, b), di in zip(pairs, d))
    comms = nx.community.louvain_communities(G, weight="weight", resolution=1.0, seed=SEED)
    label = np.full(len(dep), -1)
    rows = []
    comms = sorted(comms, key=len, reverse=True)
    for c_id, c in enumerate(c for c in comms if len(c) >= 10):
        c = list(c)
        label[c] = c_id
        sub = dep.iloc[c]
        counts = sub.group.value_counts()
        H = G.subgraph(c)
        rows.append({"district": c_id + 1, "n_sites": len(c),
                     "lon": float(sub.lon.mean()), "lat": float(sub.lat.mean()),
                     "dominant": counts.index[0], "dominant_share": float(counts.iloc[0] / len(c)),
                     "composition": counts.to_dict(),
                     "density": float(nx.density(H)),
                     "mean_degree": float(np.mean([d for _, d in H.degree()]))})
    modularity = nx.community.modularity(G, comms, weight="weight")
    return label, pd.DataFrame(rows), {"n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
                                       "n_communities": len(comms), "modularity": float(modularity),
                                       "link_km": link_km,
                                       "n_components": nx.number_connected_components(G)}


# ----------------------------------------------------------------------------------- 4
def region_mask(df, dep, adm1, group):
    from shapely import contains_xy
    if REGIONS[group]:
        poly = adm1[adm1.shapeName.isin(REGIONS[group])].union_all()
        return df.inside & contains_xy(poly, df.lon.values, df.lat.values)
    ip = dep[dep.source == "IPIS"]
    tree = cKDTree(np.column_stack([ip.lon, ip.lat]))
    d, _ = tree.query(np.column_stack([df.lon, df.lat]))
    return df.inside & (d * KM_PER_DEG < 50)


def imputer():
    return SimpleImputer(strategy="median", add_indicator=True)


def models():
    """Magnetic features are missing where EMAG2 has no coverage: logistic regression and
    the forest get a median imputer plus missing-value indicators; boosting handles NaN."""
    return {
        "Logistic regression": make_pipeline(imputer(), StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=2000)),
        "Random forest": make_pipeline(imputer(), RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED)),
        "Gradient boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                            class_weight="balanced",
                                                            random_state=SEED),
    }


def label_spreading_scores(Xtr, ytr, Xte):
    """Graph-based semi-supervised learning on a kNN graph of feature + location vectors."""
    X = np.vstack([Xtr, Xte])
    y = np.concatenate([ytr, -np.ones(len(Xte), int)])
    m = LabelSpreading(kernel="knn", n_neighbors=15, alpha=0.5, max_iter=200)
    m.fit(X, y)
    return m.label_distributions_[len(Xtr):, 1]


def prospectivity(df, dep, adm1, ds):
    results, maps, importances = [], {}, {}
    for group in ["Cu-Co", "Au", "3T"]:
        mask = region_mask(df, dep, adm1, group) & df[["bouguer", "topography"]].notna().all(1)
        sub = df[mask].copy()
        pos_cells = np.unique(cell_index(ds, dep.lon[dep.group == group], dep.lat[dep.group == group]))
        sub["y"] = sub.index.isin(pos_cells).astype(int)
        # 1-degree spatial blocks -> GroupKFold (no spatial leakage between folds)
        sub["block"] = (np.floor(sub.lon) * 1000 + np.floor(sub.lat)).astype(int)
        X = sub[FEATURES].values
        Xf = StandardScaler().fit_transform(imputer().fit_transform(X))
        Xg = np.column_stack([Xf, StandardScaler().fit_transform(sub[["lon", "lat"]].values) * 0.5])
        y = sub.y.values
        n_folds = min(5, sub.block.nunique())
        cv = GroupKFold(n_splits=n_folds)
        graph_names = ["Graph label spreading (features)",
                       "Graph label spreading (features + location)"]
        oof = {name: np.zeros(len(sub)) for name in list(models()) + graph_names}
        for tr, te in cv.split(X, y, sub.block):
            if y[tr].sum() == 0:
                continue
            for name, mdl in models().items():
                oof[name][te] = mdl.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            oof[graph_names[0]][te] = label_spreading_scores(Xf[tr], y[tr], Xf[te])
            oof[graph_names[1]][te] = label_spreading_scores(Xg[tr], y[tr], Xg[te])
        for name, s in oof.items():
            auc_folds = []
            for _, te in cv.split(X, y, sub.block):
                if 0 < y[te].sum() < len(te):
                    auc_folds.append(roc_auc_score(y[te], s[te]))
            order = np.argsort(-s)
            top10 = order[: int(0.1 * len(s))]
            results.append({"group": group, "method": name, "n_cells": int(len(sub)),
                            "n_positive_cells": int(y.sum()),
                            "auc": float(roc_auc_score(y, s)),
                            "auc_fold_mean": float(np.mean(auc_folds)),
                            "auc_fold_sd": float(np.std(auc_folds)),
                            "capture_top10": float(y[top10].sum() / y.sum())})
        # map = out-of-fold scores of the best method (each cell scored by a model that never
        # saw its 1-degree block), expressed as percentile rank within the study region
        best = max(oof, key=lambda name: next(r["auc_fold_mean"] for r in results
                                              if r["group"] == group and r["method"] == name))
        full = np.full(len(df), np.nan)
        full[sub.index] = stats.rankdata(oof[best]) / len(oof[best]) * 100
        maps[group] = (full.reshape(ds.inside.shape), best)
        rf = models()["Random forest"].fit(X, y)
        importances[group] = dict(zip(FEATURES, rf[-1].feature_importances_[:len(FEATURES)].tolist()))
    return pd.DataFrame(results), maps, importances


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ds = xr.load_dataset(PROC / "drc_grids.nc")
    dep = gpd.read_file(PROC / "deposits.gpkg")
    adm1 = gpd.read_file(ROOT / "data" / "raw" / "boundaries" / "COD_ADM1.geojson").to_crs(4326)
    df = grid_frame(ds)

    sig = signatures(df, dep, ds, adm1)
    sig.to_csv(OUT / "signatures.csv", index=False)
    print(sig.pivot_table(index="feature", columns="group", values="ks_D").round(2))

    mag = xr.load_dataset(PROC / "emag2_drc.nc").magnetic_anomaly.rename(lon="longitude", lat="latitude")
    spec, curves = spectral_depths(mag, ds)
    (OUT / "spectral.json").write_text(json.dumps({"fits": spec, "curves": curves}, indent=1))
    for k, v in spec.items():
        print(f"{k:30s} deep {v['deep_depth_bg_km']:.1f}+-{v['deep_se_km']:.1f} km  "
              f"shallow {v['shallow_depth_bg_km']:.1f}+-{v['shallow_se_km']:.1f} km")

    label, dist, gstats = districts(dep)
    dep["district"] = label
    dep.to_file(OUT / "deposits_districts.gpkg", driver="GPKG")
    dist.to_csv(OUT / "districts.csv", index=False)
    (OUT / "graph_stats.json").write_text(json.dumps(gstats, indent=1))
    print(gstats)
    print(dist.head(12).to_string())

    res, maps, imp = prospectivity(df, dep, adm1, ds)
    res.to_csv(OUT / "prospectivity_cv.csv", index=False)
    (OUT / "importances.json").write_text(json.dumps(imp, indent=1))
    pds = xr.Dataset({g.replace("-", ""): (("latitude", "longitude"), m) for g, (m, _) in maps.items()},
                     coords={"latitude": ds.latitude, "longitude": ds.longitude})
    for g, (_, best) in maps.items():
        pds[g.replace("-", "")].attrs["method"] = best
    pds.to_netcdf(OUT / "prospectivity.nc")
    print(res.round(3).to_string())
    print(json.dumps(imp, indent=1))


if __name__ == "__main__":
    main()
