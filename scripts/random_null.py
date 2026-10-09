# Random Abstention

from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
import math
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
SCORED = OUTPUTS / "scored_variants_all.csv"
CURVE = OUTPUTS / "entropy_coverage.csv"
RANDOM_DRAWS = OUTPUTS / "random_null_draws.csv"
RANDOM_SUMMARY = OUTPUTS / "random_null_summary.csv"

coverage_percent = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
draws = 1000
seed = 42

draw_columns = ["DMS_id", "requested_coverage_percent", "draw_id", "total_variants", "retained_variants", "actual_coverage", "rho", "delta_rho",]

summary_columns = [
    "DMS_id", "requested_coverage_percent", "total_variants",
    "retained_variants", "actual_coverage", "draws", "seed",
    "sampling_unit", "with_replacement", "rho_full", "entropy_rho",
    "entropy_delta_rho", "random_rho_median", "random_rho_q025",
    "random_rho_q975", "random_delta_rho_median",
    "random_delta_rho_q025", "random_delta_rho_q975",
]


def retained_count(n, coverage):
    number_to_keep = n * coverage / 100
    number_to_keep = math.ceil(number_to_keep)
    return number_to_keep


def build_random_null(scored_table, entropy_curve, draws=1000, seed=42):
    report_rows = []
    summary_rows = []
    assays = scored_table["DMS_id"].unique().tolist()
    assays.sort()
    rng = np.random.default_rng(seed)

    for assay in assays:
        assay_table = scored_table[scored_table["DMS_id"] == assay].copy()
        assay_table = assay_table.sort_values("source_row")
        assay_table = assay_table.reset_index(drop=True)
        assay_entropy = entropy_curve[entropy_curve["DMS_id"] == assay]
        n = len(assay_table)

        model_scores = assay_table["esm2_score"].to_numpy(dtype=float)
        experimental_scores = assay_table["DMS_score"].to_numpy(dtype=float)
        rho_full = float(spearmanr(model_scores, experimental_scores).statistic)

        kept_by_coverage = {}
        entropy_rho = {}
        random_rho = {}
        for coverage in coverage_percent:
            entropy_row = assay_entropy[
                assay_entropy["requested_coverage_percent"] == coverage
            ].iloc[0]
            kept = retained_count(n, coverage)

            # Match entropy counts
            if kept != entropy_row["retained_variants"] or n != entropy_row["total_variants"]:
                raise ValueError(assay + ": entropy and random counts do not match")

            kept_by_coverage[coverage] = kept
            entropy_rho[coverage] = float(entropy_row["rho"])
            random_rho[coverage] = []

        for draw_id in range(1, draws + 1):
            random_order = rng.permutation(n)

            for coverage in coverage_percent:
                kept = kept_by_coverage[coverage]
                if coverage == 100:
                    rho = rho_full
                else:
                    retained_rows = random_order[:kept]
                    model_retained = model_scores[retained_rows]
                    experimental_retained = experimental_scores[retained_rows]
                    rho = float(spearmanr(model_retained, experimental_retained).statistic)

                if pd.isna(rho):
                    raise ValueError(assay + ": undefined Spearman at " + str(coverage) + "%")

                random_rho[coverage].append(rho)
                report_data = {
                    "DMS_id": assay,
                    "requested_coverage_percent": coverage,
                    "draw_id": draw_id,
                    "total_variants": n,
                    "retained_variants": kept,
                    "actual_coverage": kept / n,
                    "rho": rho,
                    "delta_rho": rho - rho_full,
                }
                report_rows.append(report_data)

        for coverage in coverage_percent:
            values = np.array(random_rho[coverage])
            changes = values - rho_full
            kept = kept_by_coverage[coverage]

            summary_data = {
                "DMS_id": assay,
                "requested_coverage_percent": coverage,
                "total_variants": n,
                "retained_variants": kept,
                "actual_coverage": kept / n,
                "draws": draws,
                "seed": seed,
                "sampling_unit": "variant_within_assay",
                "with_replacement": False,
                "rho_full": rho_full,
                "entropy_rho": entropy_rho[coverage],
                "entropy_delta_rho": entropy_rho[coverage] - rho_full,
                "random_rho_median": np.median(values),
                "random_rho_q025": np.quantile(values, 0.025),
                "random_rho_q975": np.quantile(values, 0.975),
                "random_delta_rho_median": np.median(changes),
                "random_delta_rho_q025": np.quantile(changes, 0.025),
                "random_delta_rho_q975": np.quantile(changes, 0.975),
            }

            summary_rows.append(summary_data)

    draws_table = pd.DataFrame(report_rows, columns=draw_columns)
    summary_table = pd.DataFrame(summary_rows, columns=summary_columns)
    return draws_table, summary_table


if __name__ == "__main__":
    scored_table = pd.read_csv(SCORED)
    entropy_curve = pd.read_csv(CURVE)
    draws_table, summary_table = build_random_null(
        scored_table, entropy_curve, draws, seed
    )

    draws_table.to_csv(RANDOM_DRAWS, index=False)
    summary_table.to_csv(RANDOM_SUMMARY, index=False)

    print(f"Wrote {RANDOM_DRAWS}: {len(draws_table)} matched draws")
    print(f"Wrote {RANDOM_SUMMARY}: {len(summary_table)} assay/coverage summaries")
