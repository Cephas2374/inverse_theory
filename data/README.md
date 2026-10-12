# Data

`raw/` is filled by `python src/download_data.py`; `processed/` by the analysis scripts;
`archives/` holds scanned historical documents (PDF/TIFF) for the text-mining pipeline.

## Automatic (src/download_data.py)
| Key | Source | Licence |
|---|---|---|
| boundaries | geoBoundaries COD ADM0–2 | CC-BY 4.0 |
| mrds | USGS Mineral Resources Data System | public domain |
| gravity | ensaio global gravity + topography (10 arc-min) | CC-BY |
| emag2 | NOAA EMAG2v3 magnetic anomaly grid | public domain |
| dem | Copernicus GLO-30 via Microsoft Planetary Computer | Copernicus licence |
| sentinel2 | Sentinel-2 L2A via Planetary Computer | Copernicus licence |

## Manual (registration needed) → put files in `raw/<name>/`
- WGM2012 gravity grids: https://bgi.obs-mip.fr (free account)
- XGM2019e and EIGEN-6C4 spherical-harmonic coefficients (`XGM2019e_2159.zip`, `EIGEN-6C4.zip`)
  → `raw/gravity_sh/`: ICGEM, https://icgem.gfz-potsdam.de/tom_longtime (no registration)
- USGS Central African Copperbelt deposits, tonnage/grade and permissive tracts
  (SIR 2010-5090-J and -T, public domain) → `raw/usgs_copper/`:
  https://pubs.usgs.gov/publication/sir20105090J, https://pubs.usgs.gov/publication/sir20105090T
- 3-arc-minute global relief tiles (GMT `earth_relief_03m_g`, SRTM15+; stored as elevation/2)
  → `raw/relief/`
- EnMAP / PRISMA hyperspectral: DLR / ASI portals
- EMIT hyperspectral: NASA Earthdata (free account)
- IPIS artisanal mining sites, eastern DRC: https://ipisresearch.be (open data portal)
- Historical documents: RMCA Tervuren archives, Gallica/BnF, Internet Archive, USGS Minerals Yearbook DRC chapters
