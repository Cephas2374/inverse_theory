"""Stream the 4 GB EMAG2v3 CSV (inside its zip) and keep the DRC window as a NetCDF grid.

Columns: i, j, lon (0-360), lat, SeaLevel, UpCont (anomaly at 4 km above geoid, nT),
code (data source), error. Missing values are 99999. EMAG2v3 leaves areas without
near-surface surveys empty (no satellite gap filling), so coverage is reported explicitly.
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "raw" / "magnetics" / "EMAG2_V3_20170530.zip"
OUT = ROOT / "data" / "processed" / "emag2_drc.nc"
W, S, E, N = 11.8, -13.8, 31.8, 5.8

parts = []
with zipfile.ZipFile(SRC) as z, z.open("EMAG2_V3_20170530.csv") as f:
    for chunk in pd.read_csv(f, header=None, usecols=[2, 3, 4, 5, 6],
                             names=["lon", "lat", "sealevel", "upcont", "code"],
                             chunksize=2_000_000):
        chunk["lon"] = chunk["lon"].where(chunk["lon"] <= 180, chunk["lon"] - 360)
        keep = chunk["lon"].between(W, E) & chunk["lat"].between(S, N)
        if keep.any():
            parts.append(chunk[keep])
df = pd.concat(parts)
for c in ("sealevel", "upcont"):
    df.loc[df[c] > 9e4, c] = np.nan
df["magnetic_anomaly"] = df["upcont"].fillna(df["sealevel"])
ds = df.round({"lon": 6, "lat": 6}).set_index(["lat", "lon"]).to_xarray()
ds["magnetic_anomaly"].attrs.update(units="nT", long_name="EMAG2v3 anomaly (UpCont, SeaLevel fill)")
OUT.parent.mkdir(parents=True, exist_ok=True)
ds.to_netcdf(OUT)
for c in ("upcont", "sealevel", "magnetic_anomaly"):
    print(c, "coverage", float(np.isfinite(ds[c]).mean()))
print("codes", df["code"].value_counts().head(10).to_dict())
