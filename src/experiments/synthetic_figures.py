"""Paper figures + LaTeX table for the synthetic benchmark (reads results/synthetic)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from drcinv import TensorMesh  # noqa: E402

RES = ROOT / "results" / "synthetic"
FIG = ROOT / "paper" / "figures"
GEN = ROOT / "paper" / "sections" / "generated"

# validated categorical order (dataviz reference palette): blue, orange, aqua, yellow
COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#c98500",
       "E": "#e87ba4", "F": "#008300", "G": "#4a3aa7"}
NAME = {"A": "Smooth", "B": "+ Graph", "C": "+ Archive", "D": "Full (l2)",
        "E": "Compact", "F": "TV", "G": "Sparse graph"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})
MM = 1 / 25.4
SINGLE, DOUBLE = 90 * MM, 190 * MM

mesh = TensorMesh(0, 0, 0, 400, 400, 200, 30, 30, 15)
NX, NY, NZ = mesh.shape
IY = 17                                       # section at y = 6.8-7.2 km
r = np.load(RES / "main_500m.npz")
S = json.loads((RES / "summary.json").read_text())
EXT = [0, 12, 3, 0]
VMAX = 0.35


def save(fig, name):
    """Vector PDF for LaTeX plus a 300-dpi PNG for quick viewing, both in paper/figures."""
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def section(v):
    return np.asarray(v).reshape(NX, NY, NZ)[:, IY, :].T


def outline(ax, color=INK, lw=0.7):
    xs, zs = np.linspace(0.2, 11.8, NX), np.linspace(0.1, 2.9, NZ)
    ax.contour(xs, zs, section(r["truth"]) > 0.3, levels=[0.5], colors=color, linewidths=lw)


def fig_setup():
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 62 * MM),
                               gridspec_kw={"width_ratios": [1, 1.6]})
    obs, d = r["obs"], r["d"]
    n = int(np.sqrt(len(d)))
    im = a.imshow(d.reshape(n, n).T, origin="lower", extent=[0, 12, 0, 12], cmap="RdBu_r",
                  vmin=-np.abs(d).max(), vmax=np.abs(d).max())
    a.plot(*np.array(r["collars"]).T / 1000, "o", ms=4, mfc="white", mec=INK, mew=0.8,
           label="Historical drill holes")
    a.axhline(IY * 0.4 + 0.2, color=INK2, lw=0.6, ls="--")
    a.set(xlabel="Easting (km)", ylabel="Northing (km)", title="(a) Gravity data, 0.5 km spacing")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.colorbar(im, ax=a, shrink=0.8, label="$g_z$ (mGal)")
    im = b.imshow(section(r["truth"]), extent=EXT, cmap="RdBu_r", vmin=-VMAX, vmax=VMAX, aspect="auto")
    b.text(5.8, 2.15, "Mineralised package\n+0.35 g/cm$^3$, dip 20°", fontsize=7, color=INK)
    b.text(7.9, 1.05, "Granite cupola\n−0.12 g/cm$^3$", fontsize=7, color=INK)
    b.set(xlabel="Easting (km)", ylabel="Depth (km)", title="(b) True model, section at 7 km northing")
    fig.colorbar(im, ax=b, shrink=0.8, label="Δρ (g/cm$^3$)")
    fig.tight_layout()
    save(fig, "syn_setup")


def fig_sections():
    keys = [k for k in "ABCDEFG" if f"map_{k}" in r]
    nrow = (len(keys) + 2) // 2
    fig, axs = plt.subplots(nrow, 2, figsize=(DOUBLE, 34 * MM * nrow), sharex=True, sharey=True,
                            gridspec_kw={"hspace": 0.5, "wspace": 0.08})
    axs = axs.ravel()
    letters = "abcdefgh"
    for i, k in enumerate(keys):
        im = axs[i].imshow(section(r[f"map_{k}"]), extent=EXT, cmap="RdBu_r", vmin=-VMAX / 2,
                           vmax=VMAX / 2, aspect="auto")
        outline(axs[i])
        axs[i].set_title(f"({letters[i]}) {k}: {NAME[k]}", loc="left")
    best = keys[-1]
    sd = r[f"samples_{best}"].std(0)
    ax = axs[len(keys)]
    im2 = ax.imshow(section(sd), extent=EXT, cmap="Blues", vmin=0, aspect="auto")
    outline(ax)
    ax.set_title(f"({letters[len(keys)]}) Posterior s.d., {best}: {NAME[best]}", loc="left")
    for a in axs[len(keys) + 1:]:
        a.set_visible(False)
    for a in axs[::2]:
        a.set_ylabel("Depth (km)")
    for a in axs[-2:]:
        a.set_xlabel("Easting (km)")
    fig.colorbar(im, ax=axs[:len(keys)].tolist(), shrink=0.5, pad=0.015, label="Δρ (g/cm$^3$)")
    fig.colorbar(im2, ax=[ax], shrink=0.9, pad=0.03, label="σ (g/cm$^3$)")
    save(fig, "syn_sections")


def fig_tonnage():
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 60 * MM))
    true_t = np.array(S["true_tonnage_Mt"]) / 1000
    labels = [f"{a/1000:g}–{b/1000:g}" for a, b in S["depth_slices_m"]]
    x = np.arange(len(labels))
    a = axs[0]
    runs = {v["config"]: v for k, v in S["runs"].items() if v["spacing_m"] == 500}
    off = np.linspace(-0.36, 0.36, len(runs))
    for o, (k, v) in zip(off, sorted(runs.items())):
        p10, p50, p90 = (np.array(v[f"tonnage_p{q}"]) / 1000 for q in (10, 50, 90))
        a.errorbar(x + o, p50, yerr=[p50 - p10, p90 - p50], fmt="o", ms=3.5, color=COL[k],
                   elinewidth=1.2, capsize=0, label=NAME[k])
    a.scatter(x, true_t, marker="_", s=260, color=INK, linewidths=1.6, label="True", zorder=5)
    a.set(xticks=x, xticklabels=labels, xlabel="Depth slice (km)", ylabel="Tonnage (Gt)",
          title="(a) Tonnage by depth, P10–P90")
    a.legend(frameon=False, loc="upper left", ncol=3, columnspacing=0.6, handletextpad=0.2,
             fontsize=6.5)
    a.set_ylim(top=a.get_ylim()[1] * 1.45)
    a.grid(axis="y", color=GRID, lw=0.5)

    sweep = {}
    for v in S["runs"].values():
        sweep.setdefault(v["config"], []).append(v)
    for k in ("A", "D", "G"):
        vs = sorted(sweep.get(k, []), key=lambda v: v["spacing_m"])
        if not vs:
            continue
        sp = [v["spacing_m"] / 1000 for v in vs]
        axs[1].plot(sp, [abs(v["centroid_depth_error_m"]) for v in vs], "-o", color=COL[k],
                    lw=2, ms=4, label=NAME[k])
        tot = np.array([v["total_tonnage_p10_p50_p90"] for v in vs]) / true_t.sum() / 1000
        axs[2].errorbar(sp, tot[:, 1], yerr=[tot[:, 1] - tot[:, 0], tot[:, 2] - tot[:, 1]],
                        fmt="-o", color=COL[k], lw=2, ms=4, elinewidth=1.2, label=NAME[k])
    axs[1].set(xscale="log", xlabel="Station spacing (km)", ylabel="|Depth-centroid error| (m)",
               title="(b) Depth error vs. data spacing", ylim=(0, None))
    axs[2].axhline(1, color=INK, lw=0.8, ls="--")
    axs[2].set(xscale="log", xlabel="Station spacing (km)", ylabel="Recovered / true tonnage",
               title="(c) Total tonnage vs. data spacing")
    for ax in axs[1:]:
        ax.set_xticks([0.5, 1, 2, 4], ["0.5", "1", "2", "4"])
        ax.minorticks_off()
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "syn_tonnage")


def table():
    rows = []
    true_tot = sum(S["true_tonnage_Mt"]) / 1000
    for k, v in sorted(S["runs"].items(), key=lambda kv: (kv[1]["spacing_m"], kv[1]["config"])):
        p10, p50, p90 = (t / 1000 for t in v["total_tonnage_p10_p50_p90"])
        name = NAME[v["config"]].replace("l2", "$\\ell_2$")
        rows.append(
            f"{v['spacing_m']/1000:g} & {v['config']} & {name} & {v['chi2_per_datum']:.2f} & "
            f"{v['rmse']:.3f} & {v['centroid_depth_error_m']:+.0f} & {v['ore_iou']:.2f} & "
            f"{v.get('peak_drho', float('nan')):.2f} & {p50:.1f} [{p10:.1f}, {p90:.1f}] \\\\")
    tex = (
        "\\begin{table*}[t]\n\\centering\\small\n"
        "\\caption{Synthetic benchmark. $\\chi^2/N$: data misfit per datum; RMSE: model error; "
        "$\\Delta z_c$: error of the depth centroid of the positive anomaly in the prospect sector; "
        "IoU: overlap of cells above half the sector maximum with the true package; peak: maximum "
        "recovered $\\Delta\\rho$ in the sector (true 0.35); tonnage: "
        f"mass-balance estimate P50 [P10, P90] (true {true_tot:.1f}~Gt).}}\n"
        "\\label{tab:synthetic}\n"
        "\\begin{tabular*}{\\tblwidth}{@{\\extracolsep{\\fill}}lllrrrrrl@{}}\n\\toprule\n"
        "Spacing (km) & & Method & $\\chi^2/N$ & RMSE (g/cm$^3$) & $\\Delta z_c$ (m) & IoU & Peak "
        "& Tonnage (Gt) \\\\\n\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular*}\n\\end{table*}\n")
    (GEN / "synthetic_table.tex").write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    for p in (FIG, GEN):
        p.mkdir(parents=True, exist_ok=True)
    fig_setup()
    fig_sections()
    fig_tonnage()
    table()
    print("figures + table written")
