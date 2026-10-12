"""Synthesise a full-degree gravity disturbance (XGM2019e to 2159, EIGEN-6C4 to 2190) for the DRC.

Resolution harmonisation, step 1: replace the 10-arc-minute (~18 km) Ensaio grid by a
full-degree synthesis. pyshtools expands the coefficients on a global Driscoll-Healy grid
on the WGS84 ellipsoid with normal gravity removed (= gravity disturbance), which is then
cropped to the DRC window and written to data/processed/<model>_drc.nc.

Run:  python src/analysis/synth_gravity.py [XGM2019e_2159 | EIGEN-6C4]
"""
import gc
import sys
import zipfile
from pathlib import Path

import numpy as np
import pyshtools as pysh
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
MODEL = sys.argv[1] if len(sys.argv) > 1 else "XGM2019e_2159"
SRC = ROOT / "data" / "raw" / "gravity_sh" / f"{MODEL}.zip"
OUT = ROOT / "data" / "processed" / ("xgm2019e_drc.nc" if MODEL.startswith("XGM") else
                                      f"{MODEL.lower().replace('-', '')}_drc.nc")
W, S, E, N = 11.5, -14.0, 32.0, 6.0
A_WGS84, F_WGS84 = 6378137.0, 1 / 298.257223563
OMEGA = 7.292115e-5

gfc = SRC.with_suffix(".gfc")
if not gfc.exists():
    with zipfile.ZipFile(SRC) as z:
        name = next(n for n in z.namelist() if n.endswith(".gfc"))
        gfc.write_bytes(z.read(name))

clm = pysh.SHGravCoeffs.from_file(gfc, format="icgem", errors=False, omega=OMEGA)
lmax = clm.lmax
print("loaded", gfc.name, "lmax", lmax, flush=True)
grids = clm.expand(a=A_WGS84, f=F_WGS84, lmax=lmax, normal_gravity=True, extend=False)
total = grids.total            # SHGrid: |g| - normal gravity on the ellipsoid, m/s^2
lats, lons = total.lats(), total.lons()
data = total.data
del grids
gc.collect()
lat_sel = (lats >= S) & (lats <= N)
lon_sel = (lons >= W) & (lons <= E)
sub = data[np.ix_(lat_sel, lon_sel)] * 1e5                     # mGal
da = xr.DataArray(sub, coords={"latitude": lats[lat_sel], "longitude": lons[lon_sel]},
                  dims=("latitude", "longitude"), name="gravity_disturbance").sortby("latitude")
da.attrs.update(units="mGal", model=MODEL, lmax=int(lmax),
                surface="WGS84 ellipsoid (h = 0)", note="gravity disturbance, pyshtools synthesis")
OUT.parent.mkdir(parents=True, exist_ok=True)
da.to_dataset().to_netcdf(OUT)
print(da.shape, float(da.min()), float(da.max()), "->", OUT)
