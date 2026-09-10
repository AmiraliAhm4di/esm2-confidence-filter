# Full-coverage reproduction check

from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSAY = "A4GRB6_PSEAI_Chen_2020"
SCORED = PROJECT_ROOT / "outputs" / "scored" / (ASSAY + ".csv")
BENCHMARKS = PROJECT_ROOT / "data" / "raw" / "DMS_substitutions_Spearman_DMS_level.csv"
OUTPUT = PROJECT_ROOT / "outputs" / "reproduction_smoke.csv"
TOLERANCE = 0.02

scored_table = pd.read_csv(SCORED)
benchmark_table = pd.read_csv(BENCHMARKS)
model_scores = scored_table["esm2_score"]
experimental_scores = scored_table["DMS_score"]

correlation = spearmanr(model_scores, experimental_scores)
local_spearman = correlation.statistic
benchmark_row = benchmark_table[benchmark_table["DMS ID"] == ASSAY]
published_spearman = benchmark_row["ESM2 (650M)"].iloc[0]
absolute_difference = abs(local_spearman - published_spearman)

if absolute_difference <= TOLERANCE:
    result = "PASS"
else:
    result = "FAIL"

report_data = {
    "DMS_id": ASSAY,
    "model": "ESM2 (650M)",
    "n_variants": len(scored_table),
    "local_spearman": local_spearman,
    "published_spearman": published_spearman,
    "absolute_difference": absolute_difference,
    "tolerance": TOLERANCE,
    "result": result
}
report = pd.DataFrame([report_data])
report.to_csv(OUTPUT, index=False)

print(f"Assay: {ASSAY}")
print(f"Variants: {len(scored_table)}")
print(f"Local Spearman: {local_spearman:.6f}")
print(f"Published Spearman: {published_spearman}")
print(f"Absolute difference: {absolute_difference:.6f}")
print(f"Tolerance: {TOLERANCE}")
print(f"Result: {result}")
print(f"Saved: {OUTPUT}")