# Three-assay reproduction check

from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORED_DIR = PROJECT_ROOT / "outputs" / "scored"
BENCHMARKS = (PROJECT_ROOT / "data" / "raw" / "DMS_substitutions_Spearman_DMS_level.csv")
OUTPUT = PROJECT_ROOT / "outputs" / "reproduction_three_assay.csv"
TOLERANCE = 0.02

assays = ["A4GRB6_PSEAI_Chen_2020", "AACC1_PSEAI_Dandage_2018", "GLPA_HUMAN_Elazar_2016",]

benchmark_table = pd.read_csv(BENCHMARKS)
report_rows = []
for assay in assays:
    scored_path = SCORED_DIR / (assay + ".csv")
    scored_table = pd.read_csv(scored_path)

    model_scores = scored_table["esm2_score"]
    experimental_scores = scored_table["DMS_score"]

    correlation = spearmanr(model_scores, experimental_scores)
    local_spearman = correlation.statistic

    benchmark_row = benchmark_table[benchmark_table["DMS ID"] == assay]
    published_spearman = benchmark_row["ESM2 (650M)"].iloc[0]
    published_variant_count = benchmark_row["Number of Mutants"].iloc[0]

    local_variant_count = len(scored_table)
    absolute_difference = abs(local_spearman - published_spearman)
    row_count_match = local_variant_count == published_variant_count

    if row_count_match == False:
        result = "ROW_COUNT_MISMATCH"
    elif absolute_difference <= TOLERANCE:
        result = "PASS"
    else:
        result = "FAIL"

    report_data = {
        "DMS_id": assay,
        "model": "ESM2 (650M)",
        "local_variants": local_variant_count,
        "published_variants": published_variant_count,
        "row_count_match": row_count_match,
        "local_spearman": local_spearman,
        "published_spearman": published_spearman,
        "absolute_difference": absolute_difference,
        "tolerance": TOLERANCE,
        "result": result,
    }

    report_rows.append(report_data)



report = pd.DataFrame(report_rows)
report.to_csv(OUTPUT, index=False)

print(report.to_string(index=False))
print(f"\nSaved: {OUTPUT}")