"""Check the example localities of Appendix Table A.1 against the open databases on disk:
USGS MRDS, IPIS artisanal sites, OpenStreetMap mines and the USGS Central African Copperbelt
deposits. For each (row, example) the script reports whether a site of that name exists,
which commodities it lists and in which province (geoBoundaries ADM1) it lies.

Run:  python src/tools_check_appendix.py -> results/appendix_check.json and a printed report.
"""
import json
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import contains_xy

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ROWS = {   # row: (commodity regex, listed provinces, examples)
    "Copper": (r"copper|cuivre|cu\b|malachite", ["Lualaba", "Haut-Katanga"], ["Kamoa", "Kakula", "Tenke", "Fungurume", "Kolwezi", "Kambove"]),
    "Cobalt": (r"cobalt", ["Lualaba", "Haut-Katanga"], ["Mutanda", "Tenke", "Fungurume", "Kamoto"]),
    "Zn-Pb-Ge-Ag": (r"zinc|lead|germanium|silver", ["Haut-Katanga"], ["Kipushi"]),
    "Lithium": (r"lithium|spodumene|tin|tantal", ["Tanganyika"], ["Manono", "Kitotolo"]),
    "Manganese": (r"manganese", ["Lualaba"], ["Kisenge"]),
    "Nickel": (r"nickel", ["Kasai", "Sankuru"], []),
    "Tin": (r"tin|cassit", ["Nord-Kivu", "Sud-Kivu", "Maniema", "Tanganyika"], ["Bisie", "Manono"]),
    "Tantalum-niobium": (r"tantal|coltan|niob|columb", ["Nord-Kivu", "Sud-Kivu", "Maniema", "Tanganyika"], ["Rubaya", "Kalehe"]),
    "Tungsten": (r"tungsten|wolfram", ["Nord-Kivu", "Sud-Kivu", "Maniema"], ["Nyabibwe"]),
    "Niobium-REE": (r"niob|pyrochlore|rare earth|carbonatite", ["Nord-Kivu"], ["Lueshe", "Bingo"]),
    "Gold": (r"gold|\bor\b", ["Haut-Uele", "Ituri", "Sud-Kivu", "Maniema", "Tshopo"], ["Kibali", "Kilo", "Moto", "Twangiza", "Namoya"]),
    "PGE": (r"platinum|palladium|pge", ["Kasai", "Haut-Katanga"], []),
    "Diamonds": (r"diamond|diamant", ["Kasai", "Kasai-Oriental", "Lomami"], ["Mbuji", "Tshikapa"]),
    "Uranium": (r"uranium", ["Haut-Katanga"], ["Shinkolobwe"]),
    "Coal": (r"coal|charbon", ["Tanganyika", "Haut-Katanga"], ["Luena", "Lukuga"]),
    "Iron ore": (r"iron|fer\b", ["Tshopo", "Haut-Uele", "Ituri"], ["Banalia"]),
    "Bauxite": (r"bauxite|alumin", ["Kongo-Central"], ["Mayumbe"]),
    "Phosphate": (r"phosphat", ["Kongo-Central"], ["Lukula"]),
    "Limestone etc.": (r"limestone|cement|gypsum|salt|dolomite", ["Kongo-Central", "Haut-Katanga", "Lualaba"], ["Lukala"]),
}


def load():
    adm1 = gpd.read_file(RAW / "boundaries" / "COD_ADM1.geojson").to_crs(4326)
    recs = []
    m = pd.read_csv(RAW / "mrds" / "mrds.csv", low_memory=False, encoding="latin-1")
    m = m[m["country"].astype(str).str.contains("Kinshasa|Congo", regex=True)]
    for _, r in m.iterrows():
        txt = " ".join(str(r[c]) for c in ("commod1", "commod2", "commod3", "ore") if pd.notna(r.get(c)))
        recs.append(("MRDS", str(r["site_name"]), r["longitude"], r["latitude"], txt))
    ip = gpd.read_file(RAW / "ipis" / "cod_mines_ipis.geojson")
    for _, r in ip.iterrows():
        txt = " ".join(str(r[c]) for c in ("mineral1", "mineral2", "mineral3") if pd.notna(r.get(c)))
        recs.append(("IPIS", str(r["name"]), r["longitude"], r["latitude"], txt))
    osm = json.loads((RAW / "osm" / "drc_mines_osm.json").read_text(encoding="utf-8"))["elements"]
    for e in osm:
        t = e.get("tags", {})
        lon = e.get("lon", e.get("center", {}).get("lon"))
        lat = e.get("lat", e.get("center", {}).get("lat"))
        recs.append(("OSM", t.get("name", ""), lon, lat, t.get("resource", "") + " " + t.get("product", "")))
    u = gpd.read_file(RAW / "usgs_copper" / "sir2010-5090T_GISdata" / "sir2010-5090-T_GIS" / "CACB_deposits_prospects.shp")
    for _, r in u.iterrows():
        txt = str(r.get("Comm_major", "")) + (" cobalt" if (r.get("Co_pct") or 0) > 0 else "")
        recs.append(("USGS", f"{r['Name']} {r.get('Name_other') or ''} {r.get('Includes') or ''}",
                     r["Longitude"], r["Latitude"], txt))
    df = pd.DataFrame(recs, columns=["source", "name", "lon", "lat", "commodity"]).dropna(subset=["lon", "lat"])
    df["province"] = None
    for _, p in adm1.iterrows():
        df.loc[contains_xy(p.geometry, df.lon.values, df.lat.values), "province"] = p.shapeName
    return df, adm1


def main():
    df, adm1 = load()
    names = df.name.str.lower()
    out = {}
    for row, (pat, provs, examples) in ROWS.items():
        com = df.commodity.str.lower().str.contains(pat, regex=True, na=False)
        prov_counts = df[com].province.value_counts().head(6).to_dict()
        ex = {}
        for e in examples:
            hit = df[names.str.contains(e.lower(), regex=False, na=False)]
            ex[e] = dict(n=int(len(hit)), with_commodity=int((hit.commodity.str.lower().str.contains(pat, regex=True, na=False)).sum()),
                         sources=sorted(hit.source.unique().tolist()),
                         provinces=sorted({p for p in hit.province.dropna()}),
                         commodities=sorted({c.strip()[:40] for c in hit.commodity.head(5)}))
        out[row] = dict(listed_provinces=provs, sites_with_commodity=int(com.sum()),
                        top_provinces_in_data=prov_counts, examples=ex)
    (ROOT / "results" / "appendix_check.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("province names in geoBoundaries:", sorted(adm1.shapeName.unique()))
    for row, r in out.items():
        print(f"\n## {row}: listed {r['listed_provinces']}; {r['sites_with_commodity']} sites; data provinces {r['top_provinces_in_data']}")
        for e, v in r["examples"].items():
            print(f"   {e:12s} n={v['n']:3d} with_commodity={v['with_commodity']:3d} {v['sources']} {v['provinces']} {v['commodities'][:3]}")


if __name__ == "__main__":
    main()
