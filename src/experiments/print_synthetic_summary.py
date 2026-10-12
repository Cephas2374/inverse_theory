"""Print the synthetic benchmark summary as a compact table."""
import json
from pathlib import Path

S = json.loads((Path(__file__).resolve().parents[2] / "results" / "synthetic" / "summary.json").read_text())
print("true total (Gt):", round(sum(S["true_tonnage_Mt"]) / 1000, 2))
for k, v in S["runs"].items():
    mass = [round(t / 1000, 1) for t in v["total_tonnage_p10_p50_p90"]]
    thr = [round(t / 1000, 1) for t in v.get("threshold_tonnage_p10_p50_p90", [0, 0, 0])]
    print(f"{k:7s} chi2={v['chi2_per_datum']:.2f} rmse={v['rmse']:.4f} "
          f"dz={v['centroid_depth_error_m']:+5.0f} iou={v['ore_iou']:.2f} "
          f"peak={v.get('peak_drho', float('nan')):.2f} mass={mass} thr={thr}")
