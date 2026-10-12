"""Download open datasets into data/raw.

Usage (after `. .\\activate.ps1`):
    python src/download_data.py               # all automatic datasets
    python src/download_data.py boundaries mrds

Datasets that need registration (WGM2012, EnMAP, PRISMA) are listed in
data/README.md and must be placed in data/raw by hand.
"""
import json
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# lon_min, lat_min, lon_max, lat_max
DRC_BBOX = (12.0, -13.6, 31.5, 5.5)
AOIS = {
    "kolwezi": (25.2, -11.0, 26.0, -10.4),   # Cu-Co validation
    "manono": (27.1, -7.5, 27.6, -7.1),      # Li-Sn-Ta pegmatites
    "kivu": (27.8, -2.6, 29.2, -1.2),        # Sn-Ta-W
}


def fetch(url, dest, chunk=1 << 20):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  exists: {dest.relative_to(ROOT)}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  GET {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk):
                f.write(block)
        tmp.replace(dest)
    return dest


def boundaries():
    """DRC national + provincial boundaries (geoBoundaries, CC-BY)."""
    for level in ("ADM0", "ADM1", "ADM2"):
        meta = requests.get(
            f"https://www.geoboundaries.org/api/current/gbOpen/COD/{level}/", timeout=60
        ).json()
        fetch(meta["gjDownloadURL"], RAW / "boundaries" / f"COD_{level}.geojson")


def mrds():
    """USGS Mineral Resources Data System (global deposit/occurrence points)."""
    z = fetch("https://mrdata.usgs.gov/mrds/mrds-csv.zip", RAW / "mrds" / "mrds-csv.zip")
    with zipfile.ZipFile(z) as zf:
        zf.extractall(z.parent)


def gravity():
    """Global gravity disturbance + topography grids (10 arc-min) via ensaio."""
    import ensaio
    import xarray as xr

    out = RAW / "gravity"
    out.mkdir(parents=True, exist_ok=True)
    w, s, e, n = DRC_BBOX
    for name, fn in [("earth_gravity", ensaio.fetch_earth_gravity),
                     ("earth_topography", ensaio.fetch_earth_topography)]:
        grid = xr.load_dataset(fn(version=1))
        grid.sel(longitude=slice(w, e), latitude=slice(s, n)).to_netcdf(out / f"{name}_drc.nc")
        print(f"  wrote gravity/{name}_drc.nc")


def emag2():
    """EMAG2v3 global magnetic anomaly grid (NOAA NCEI)."""
    fetch("https://www.ngdc.noaa.gov/geomag/data/EMAG2/EMAG2_V3_20170530/EMAG2_V3_20170530.zip",
          RAW / "magnetics" / "EMAG2_V3_20170530.zip")


def _stac_download(collection, bbox, outdir, assets, query=None, max_items=None):
    import planetary_computer
    import pystac_client

    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = cat.search(collections=[collection], bbox=bbox, query=query, max_items=max_items,
                        sortby=[{"field": "eo:cloud_cover", "direction": "asc"}] if query else None)
    items = list(search.items())
    print(f"  {collection}: {len(items)} items")
    for it in items:
        for a in assets:
            if a in it.assets:
                ext = Path(it.assets[a].href.split("?")[0]).suffix or ".tif"
                fetch(it.assets[a].href, outdir / it.id / f"{a}{ext}")
        (outdir / it.id).mkdir(parents=True, exist_ok=True)
        (outdir / it.id / "item.json").write_text(json.dumps(it.to_dict()))


def dem():
    """Copernicus GLO-30 DEM tiles for the study AOIs."""
    for name, bbox in AOIS.items():
        _stac_download("cop-dem-glo-30", bbox, RAW / "dem" / name, ["data"])


def sentinel2():
    """Least-cloudy Sentinel-2 L2A scenes (all 10/20 m bands) for each AOI."""
    bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"]
    for name, bbox in AOIS.items():
        _stac_download("sentinel-2-l2a", bbox, RAW / "sentinel2" / name, bands,
                       query={"eo:cloud_cover": {"lt": 5}}, max_items=2)


HEADERS = {"User-Agent": "drc-deep-research/0.1 (academic; open geodata)"}


def osm():
    """Mines, quarries, mine shafts and resource-tagged features in the DRC (OpenStreetMap, ODbL)."""
    q = ('[out:json][timeout:300];area["ISO3166-1"="CD"][admin_level=2]->.a;('
         'node["landuse"="quarry"](area.a);way["landuse"="quarry"](area.a);'
         'node["man_made"="mineshaft"](area.a);node["industrial"="mine"](area.a);'
         'way["industrial"="mine"](area.a);node["resource"](area.a);way["resource"](area.a);'
         ');out center tags;')
    dest = RAW / "osm" / "drc_mines_osm.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in ("https://overpass-api.de/api/interpreter",
                "https://overpass.kumi.systems/api/interpreter"):
        r = requests.post(url, data={"data": q}, headers=HEADERS, timeout=400)
        print(f"  {url}: HTTP {r.status_code}")
        if r.ok:
            dest.write_bytes(r.content)
            print(f"  {len(r.json()['elements'])} features")
            return
    raise RuntimeError("all Overpass endpoints failed")


def ipis():
    """IPIS artisanal mining sites, eastern DRC (open data, WFS)."""
    url = ("http://geo.ipisresearch.be/geoserver/public/ows?service=WFS&version=1.0.0"
           "&request=GetFeature&typeName=public:cod_mines_curated_all_opendata_p_ipis"
           "&outputFormat=application/json")
    fetch(url, RAW / "ipis" / "cod_mines_ipis.geojson")


DATASETS = {f.__name__: f for f in (boundaries, mrds, gravity, emag2, dem, sentinel2, osm, ipis)}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(DATASETS)
    failed = []
    for name in wanted:
        print(f"[{name}]")
        try:
            DATASETS[name]()
        except Exception as exc:  # keep going; report at the end
            print(f"  FAILED: {exc}")
            failed.append(name)
    print("done" if not failed else f"failed: {', '.join(failed)}")
    sys.exit(1 if failed else 0)
