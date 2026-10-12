# inverse_theory: deep mineral and hydrocarbon mapping in the DRC from open data

Code for graph-regularised Bayesian inversion of open geodata (global gravity, EMAG2v3
magnetics, occurrence databases) to test how far open information can see below the
surface of the Democratic Republic of the Congo. It accompanies the manuscript *Seeing below
the surface: graph-regularised Bayesian joint inversion of open geodata, remote sensing and
historical archives for deep mineral and hydrocarbon mapping in the DRC* (C. Bisimwa
Mwimangire, HfT Stuttgart; in preparation for *Computers & Geosciences*).

## What it does

- **`drcinv`**, a small inversion library:
  - prism gravity, prism magnetic (induced) and spherical tesseroid kernels;
  - Tikhonov, fabric-graph Laplacian, minimum-support, total-variation and sparse-graph
    regularisation (IRLS);
  - discrepancy-principle trade-off and randomise-then-optimise (RTO) posterior sampling;
  - cross-gradient joint gravity–magnetic inversion (Gauss–Newton).
- **Synthetic benchmark.** A Copperbelt-like model compares seven regularisation theories at
  equal misfit and four station spacings. It shows that mass balance is the robust tonnage
  estimator.
- **National analysis.** It harmonises MRDS, IPIS and OpenStreetMap occurrences and analyses
  them against gravity, magnetics and terrain:
  - spectral depth to magnetic sources;
  - geophysical signatures of each commodity group;
  - Louvain mining districts;
  - prospectivity, compared across five learners under spatially blocked cross-validation.
- **Field inversions.**
  - *Katanga Copperbelt:* Cu–Co endowment from XGM2019e and EIGEN-6C4 gravity, run through a
    24-member ensemble of processing and modelling choices. It is validated against USGS
    deposit tonnages and a geology-matched (Roan Group) background.
  - *Cuvette Centrale:* depth to basement from tesseroid-kernel gravity inversion combined
    with local magnetic spectral depths, plus a joint gravity–magnetic inversion.

Main findings, briefly:
- *Katanga:* open gravity resolves the Copperbelt as a dense belt, but most of that excess
  mass is the Roan host sequence. It does not rank sub-districts by contained copper.
- *Cuvette Centrale:* basement lies at a median of about 7 km (3–10 km). This is
  consistent with seismic estimates and with wells that ended in sediments.
- *Data needs:* depth and ore tonnage need station spacings near 1 km.

## Layout

```
src/drcinv/          inversion library (mesh, forward, graph, inversion, joint)
src/download_data.py fetch open data into data/raw
src/analysis/        national grids, EMAG2 extraction, SH gravity synthesis, analysis, figures
src/experiments/     synthetic benchmark, field inversions (v1, v2), validation, figures
tests/               forward-model checks (point-mass far field, Bouguer slab)
data/README.md       data sources and licences
environment.yml      conda-forge environment (Python 3.11)
setup.ps1, activate.ps1  project-local install (micromamba) on Windows
```

## Setup

The scripts keep every tool, cache and temporary file inside the project folder.

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1   # once: micromamba + environment
. .\activate.ps1                                     # every session
```

On Linux or macOS, create the environment from `environment.yml` with conda or mamba.

## Reproducing the results

Run the scripts in this order. Each writes to `data/processed`, `results/` or `paper/`.

```bash
python src/download_data.py                      # open data -> data/raw (see data/README.md)
python src/analysis/extract_emag2.py             # EMAG2v3 -> DRC grid
python src/analysis/synth_gravity.py XGM2019e_2159
python src/analysis/synth_gravity.py EIGEN-6C4   # needs the ICGEM .zip in data/raw/gravity_sh
python src/analysis/regional_data.py             # national grids + harmonised occurrences
python src/analysis/regional_analysis.py
python src/analysis/regional_figures.py

python src/experiments/synthetic_copperbelt.py   # synthetic benchmark (~1 h)
python src/experiments/synthetic_figures.py

python src/experiments/field_inversions.py       # field inversions, single configuration
python src/experiments/field_v2.py katanga       # main runs + 24-member choice ensemble
python src/experiments/field_v2.py cuvette       # tesseroids, ensemble, joint inversion
python src/experiments/katanga_validation.py     # sub-districts vs USGS tonnages
python src/experiments/cuvette_analysis.py
python src/experiments/field_figures.py
```

`--quick` options exist for smoke tests. Some data must be fetched by hand; they are listed
in `data/README.md`:
- XGM2019e and EIGEN-6C4 coefficients from ICGEM;
- USGS SIR 2010-5090-J/T deposit and tract files;
- 3-arc-minute global relief tiles.

## Data

No data are stored in this repository. All inputs are open, and their sources and licences
are listed in `data/README.md`. Well information for the Cuvette Centrale is used only as
published depth bounds (Kadima et al., 2011); well coordinates are not public.

## Citation

Please cite the manuscript once published. Until then, cite this repository.
