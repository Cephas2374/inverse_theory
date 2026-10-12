"""Print P50 tonnage per depth slice (Gt) for each 0.5 km run."""
import json
from pathlib import Path

S = json.loads((Path(__file__).resolve().parents[2] / "results" / "synthetic" / "summary.json").read_text())
print("slices", S["depth_slices_m"])
print("true  ", [round(t / 1000, 1) for t in S["true_tonnage_Mt"]])
for k, v in S["runs"].items():
    if v["spacing_m"] == 500:
        print(v["config"], "   ", [round(t / 1000, 1) for t in v["tonnage_p50"]])
