# inverse_theory: how deep can open data see in the DRC?

Code for benchmarked and validated potential-field inversion of open geodata (global gravity,
EMAG2v3 magnetics, relief, occurrence databases) in the Democratic Republic of the Congo and
the Zambian Copperbelt. It accompanies the manuscript *How deep can open data see?
Benchmarked and validated potential-field inversion for copper–cobalt endowment and basin
architecture in the Democratic Republic of the Congo* (C. Bisimwa Mwimangire, HfT Stuttgart;
in preparation for *Computers & Geosciences*).

## What it does

- **`drcinv`**, a small inversion library:
  - prism gravity, prism magnetic (induced) and spherical tesseroid kernels;
  - Tikhonov, fabric-graph Laplacian, minimum-support, total-variation and sparse-graph
    regularisation (IRLS);
  - discrepancy-principle trade-off and randomise-then-optimise (RTO) posterior sampling;
  - cross-gradient joint gravity–magnetic inversion (Gauss–Newton).
- **Synthetic benchmark.** Seven regularisation theories are compared at equal misfit over
  four station spacings, three ore-body geometries and eight noise realisations.
- **National analysis.** MRDS, IPIS and OpenStreetMap occurrences are harmonised and analysed
  against gravity, magnetics and terrain:
  - spectral depth to magnetic sources;
  - geophysical signatures of each commodity group;
  - Louvain mining districts;
  - prospectivity under spatially blocked cross-validation.
- **Field inversions with choice ensembles**, which propagate processing and modelling
  choices into every result:
  - *Copperbelt, Katanga and Zambia:* XGM2019e and EIGEN-6C4 gravity, validated against USGS
    deposit tonnages and a geology-matched (Roan Group) background on both sides of the belt;
  - *Cuvette Centrale:* depth to basement from tesseroid-kernel gravity and local magnetic
    spectral depths, plus a joint gravity–magnetic inversion and depth sections.

Main findings, briefly:
- *Theory choice:* the best regularisation theory depends on data density and on the target
  geometry. The ℓ2 fabric-graph model keeps mass-balance tonnage within 15 % everywhere.
- *Copperbelt:* open gravity maps basin and basement architecture. It shows dense Roan rafts
  in Katanga and a mass deficit in Zambia. It does not rank sub-districts by contained copper
  (Spearman −0.54 over 14 sub-districts).
- *Cuvette Centrale:* basement lies at a median of about 7 km (3–10 km), consistent with
  seismic interpretation and wells.
- *Data needs:* ore tonnage and deep geometry need station spacings near 1 km.

## Layout

```
src/drcinv/          inversion library (mesh, forward, graph, inversion, joint)
src/download_data.py fetch open data into data/raw
src/analysis/        national grids, EMAG2 extraction, SH gravity synthesis, analysis, figures
src/experiments/     synthetic benchmark and robustness, field inversions, validation, figures
src/tools_*.py       reference check (Crossref), appendix check, contact sheet
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
python src/experiments/synthetic_multiseed.py --geometry baseline --seeds 0-7   # also steep, thin-deep
python src/experiments/synthetic_multiseed.py --summarise                        # robustness table + figure

python src/experiments/field_inversions.py       # field inversions, single configuration
python src/experiments/field_v2.py katanga       # main runs + 24-member choice ensemble
python src/experiments/field_v2.py zambia        # Zambian Copperbelt (second validation belt)
python src/experiments/field_v2.py cuvette       # tesseroids, ensemble, joint inversion
python src/experiments/katanga_validation.py     # sub-districts vs USGS tonnages
python src/experiments/zambia_validation.py      # Zambia + pooled test over both belts
python src/experiments/cuvette_analysis.py
python src/experiments/profiles.py               # depth sections along the arc / across the basin
python src/experiments/field_figures.py
```

`--quick` options exist for smoke tests. Some data must be fetched by hand; they are listed
in `data/README.md`:
- XGM2019e and EIGEN-6C4 coefficients from ICGEM;
- USGS SIR 2010-5090-J/T deposit and tract files;
- 3-arc-minute global relief tiles.

`src/tools_check_refs.py` checks the bibliography against Crossref, and
`src/tools_check_appendix.py` checks the appendix commodity table against the occurrence
databases.

## Data

No data are stored in this repository. All inputs are open, and their sources and licences
are listed in `data/README.md`. Well information for the Cuvette Centrale is used only as
published depth bounds (Kadima et al., 2011); well coordinates are not public.

## Licence and citation

The code is released under the MIT licence (see `LICENSE`). Data used by the scripts keep
their own licences, listed in `data/README.md`. Please cite the manuscript once it is
published; until then, cite this repository.
