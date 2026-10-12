"""Build the national analysis grids and the unified deposit table for the DRC.

Outputs (data/processed):
    drc_grids.nc     common 0.05-degree grid: topography, gravity disturbance, Bouguer
                     anomaly, horizontal gradient of Bouguer, magnetic anomaly (EMAG2),
                     analytic-signal amplitude, tilt derivative, in/out-of-DRC mask
    deposits.gpkg    MRDS + IPIS + OSM occurrences with a commodity group
"""
import json
import re
from pathlib import Path

import boule as bl
import geopandas as gpd
import harmonica as hm
import numpy as np
import pandas as pd
import xarray as xr
from shapely import contains_xy

ROOT = Path(__file__).resolve().parents[2]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
STEP = 0.05                       # degrees (~5.5 km)
W, S, E, N = 12.0, -13.6, 31.4, 5.4
R_EARTH = 6_371_000.0

GROUPS = {   # regex on lower-case commodity text -> group (first match wins)
    "Cu-Co": r"copper|cobalt|cuivre|cobalt|digénite|digenite|malachite",
    "3T": r"tin|cassit|coltan|tantal|wolfram|tungsten|niob|columb",
    "Au": r"gold|\bor\b|^or$",
    "Diamond": r"diamond|diamant",
}


def commodity_group(text):
    t = str(text).lower()
    for g, pat in GROUPS.items():
        if re.search(pat, t):
            return g
    return "Other"


def load_boundaries():
    adm0 = gpd.read_file(RAW / "boundaries" / "COD_ADM0.geojson").to_crs(4326)
    adm1 = gpd.read_file(RAW / "boundaries" / "COD_ADM1.geojson").to_crs(4326)
    return adm0, adm1


def deposits(adm0):
    rows = []
    m = pd.read_csv(RAW / "mrds" / "mrds.csv", low_memory=False, encoding="latin-1")
    m = m[m["country"].astype(str).str.contains("Kinshasa")]
    for _, r in m.iterrows():
        txt = " ".join(str(r[c]) for c in ("commod1", "commod2", "commod3", "ore") if pd.notna(r[c]))
        rows.append(("MRDS", r["site_name"], r["longitude"], r["latitude"], txt, r["dev_stat"]))
    ip = gpd.read_file(RAW / "ipis" / "cod_mines_ipis.geojson")
    for _, r in ip.iterrows():
        txt = " ".join(str(r[c]) for c in ("mineral1", "mineral2", "mineral3") if pd.notna(r[c]))
        rows.append(("IPIS", r["name"], r["longitude"], r["latitude"], txt, "Artisanal"))
    osm = json.loads((RAW / "osm" / "drc_mines_osm.json").read_text(encoding="utf-8"))["elements"]
    for e in osm:
        t = e.get("tags", {})
        res = t.get("resource")
        if not res and t.get("industrial") != "mine":
            continue                       # untagged quarries are mostly aggregate
        lon = e.get("lon", e.get("center", {}).get("lon"))
        lat = e.get("lat", e.get("center", {}).get("lat"))
        rows.append(("OSM", t.get("name", ""), lon, lat, res or "mine", "Mine/quarry"))
    df = pd.DataFrame(rows, columns=["source", "name", "lon", "lat", "commodity", "status"])
    df = df.dropna(subset=["lon", "lat"])
    df["group"] = df["commodity"].map(commodity_group)
    # OSM 'mine' without a resource tag: Katanga industrial mines are Cu-Co
    katanga = (df["lat"] < -9.5) & (df["lon"] > 22.5) & (df["lon"] < 29)
    df.loc[(df["source"] == "OSM") & (df["commodity"] == "mine") & katanga, "group"] = "Cu-Co"
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326)
    inside = contains_xy(adm0.union_all(), g["lon"].values, g["lat"].values)
    return g[inside].reset_index(drop=True)


def derivatives(da):
    """Analytic signal and tilt derivative on an approximately metric grid.

    Near the equator one degree spans ~111 km in both directions (cos 13 deg = 0.97), so the
    grid is treated as Cartesian with spacing STEP * 111.2 km (distortion < 3 % in the DRC).
    """
    filled = da.fillna(float(da.mean()))
    k = np.pi / 180 * R_EARTH
    g = filled.assign_coords(easting=("longitude", da.longitude.values * k),
                             northing=("latitude", da.latitude.values * k))
    g = g.swap_dims({"longitude": "easting", "latitude": "northing"})
    pad = {"easting": 40, "northing": 40}
    dz = hm.derivative_upward(xr.DataArray(g.values, coords={"northing": g.northing, "easting": g.easting},
                                           dims=("northing", "easting")).pad(pad, mode="reflect")
                              .assign_coords(easting=np.arange(g.easting.size + 80) * STEP * k,
                                             northing=np.arange(g.northing.size + 80) * STEP * k))
    dz = dz.isel(easting=slice(40, -40), northing=slice(40, -40)).values
    dy, dx = np.gradient(g.values, STEP * k)
    hgm = np.hypot(dx, dy)
    asa = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    tilt = np.arctan2(dz, hgm)
    # no data in -> no derivative out (gaps were filled only to run the FFT);
    # one extra cell of margin removes the gradient stencil's contact with the fill
    gap = ~np.isfinite(da.values)
    gap = gap | np.roll(gap, 1, 0) | np.roll(gap, -1, 0) | np.roll(gap, 1, 1) | np.roll(gap, -1, 1)
    for a in (hgm, asa, tilt):
        a[gap] = np.nan
    return hgm, asa, tilt


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    adm0, adm1 = load_boundaries()

    lon = np.arange(W, E + 1e-9, STEP)
    lat = np.arange(S, N + 1e-9, STEP)
    grav = xr.load_dataset(RAW / "gravity" / "earth_gravity_drc.nc")
    topo = xr.load_dataset(RAW / "gravity" / "earth_topography_drc.nc")
    lon2, lat2 = xr.broadcast(grav.longitude, grav.latitude)
    normal = xr.DataArray(
        bl.WGS84.normal_gravity((lon2.T.values, lat2.T.values, grav.height.values)),
        coords=grav.gravity.coords, dims=grav.gravity.dims)
    disturbance = grav.gravity - normal
    bouguer = disturbance - hm.bouguer_correction(topo.topography)
    mag = xr.load_dataset(PROC / "emag2_drc.nc").magnetic_anomaly

    def on_grid(da):
        return da.interp(longitude=lon, latitude=lat) if "longitude" in da.dims else \
            da.rename(lon="longitude", lat="latitude").interp(longitude=lon, latitude=lat)

    out = xr.Dataset({
        "topography": on_grid(topo.topography),
        "gravity_disturbance": on_grid(disturbance),
        "bouguer": on_grid(bouguer),
        "magnetic": on_grid(mag),
    })
    out["bouguer_hgm"] = (("latitude", "longitude"), derivatives(out.bouguer)[0])
    _, asa, tilt = derivatives(out.magnetic)
    out["mag_asa"] = (("latitude", "longitude"), asa)
    out["mag_tilt"] = (("latitude", "longitude"), tilt)
    LON, LAT = np.meshgrid(lon, lat)
    out["inside"] = (("latitude", "longitude"), contains_xy(adm0.union_all(), LON, LAT))
    for v, u in [("topography", "m"), ("gravity_disturbance", "mGal"), ("bouguer", "mGal"),
                 ("magnetic", "nT"), ("bouguer_hgm", "mGal/m"), ("mag_asa", "nT/m"),
                 ("mag_tilt", "rad")]:
        out[v].attrs["units"] = u
    out.to_netcdf(PROC / "drc_grids.nc")

    dep = deposits(adm0)
    dep.to_file(PROC / "deposits.gpkg", driver="GPKG")
    print(out)
    print(dep.groupby(["source", "group"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
