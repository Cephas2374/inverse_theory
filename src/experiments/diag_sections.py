"""Quick diagnostic: x-z sections at y = 6 km of truth and MAP models (not a paper figure)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
r = np.load(ROOT / "results" / "synthetic" / "main_500m.npz")
nx, ny, nz = 30, 30, 15
iy = int(sys.argv[2]) if len(sys.argv) > 2 else 15
keys = ["truth"] + [k for k in ("map_A", "map_B", "map_C", "map_D") if k in r]
fig, axs = plt.subplots(len(keys), 1, figsize=(7, 2.0 * len(keys)), sharex=True)
for ax, k in zip(axs, keys):
    sec = r[k].reshape(nx, ny, nz)[:, iy, :].T
    im = ax.imshow(sec, extent=[0, 12, 3, 0], cmap="RdBu_r", vmin=-0.35, vmax=0.35, aspect="auto")
    ax.contour(np.linspace(0.2, 11.8, nx), np.linspace(0.1, 2.9, nz),
               r["truth"].reshape(nx, ny, nz)[:, iy, :].T > 0.3, levels=[0.5], colors="k", linewidths=0.8)
    ax.set_title(f"{k}  min={r[k].min():.2f} max={r[k].max():.2f}", fontsize=9)
    ax.set_ylabel("depth (km)")
axs[-1].set_xlabel("x (km)")
fig.colorbar(im, ax=axs, shrink=0.6, label="drho (g/cc)")
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".cache" / "preview" / "diag.png"
fig.savefig(out, dpi=80)
print(out)
