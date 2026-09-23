# Entropy Abstention

from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
import math
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
SCORED = OUTPUTS / "scored_variants_all.csv"
CURVE = OUTPUTS / "entropy_coverage.csv"
RANKING = OUTPUTS / "entropy_ranking.csv"

coverage_percent = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

def retained_count(n, coverage):
    number_to_keep = n * coverage / 100
    number_to_keep = math.ceil(number_to_keep)
    return number_to_keep


def build_results(scored_table, coverages):
    report_rows = []
    ranked_tables = []
    assays = scored_table["DMS_id"].unique().tolist()
    assays.sort()

    for assay in assays:
        assay_table = scored_table[scored_table["DMS_id"] == assay].copy()

        # Lowest entropy first then source row for tie
        ranked_table = assay_table.sort_values(["entropy", "source_row"])
        ranked_table = ranked_table.reset_index(drop=True)
        n = len(ranked_table)
        kept_50 = retained_count(n, 50)

        ranks = []
        retained_at_50 = []
        for rank in range(1, n + 1):
            ranks.append(rank)
            if rank <= kept_50:
                retained_at_50.append(True)
            else:
                retained_at_50.append(False)

        ranked_table["entropy_rank"] = ranks
        ranked_table["retained_at_50"] = retained_at_50


        model_scores = ranked_table["esm2_score"]
        experimental_scores = ranked_table["DMS_score"]
        correlation = spearmanr(model_scores, experimental_scores)
        rho_full = float(correlation.statistic)
        rho_by_coverage = {}
        for coverage in coverages:
            kept = retained_count(n, coverage)
            retained_table = ranked_table.iloc[:kept]

            model_scores = retained_table["esm2_score"]
            experimental_scores = retained_table["DMS_score"]
            correlation = spearmanr(model_scores, experimental_scores)
            rho = float(correlation.statistic)

            # Constant scores have no Spearman correlation
            if pd.isna(rho):
                raise ValueError(assay + ": undefined Spearman at " + str(coverage) + "%")

            rho_by_coverage[coverage] = rho

        rho_50 = rho_by_coverage[50]
        delta_rho_50 = rho_50 - rho_full

        for coverage in coverages:
            kept = retained_count(n, coverage)
            rho = rho_by_coverage[coverage]

            report_data = {
                "DMS_id": assay,
                "requested_coverage_percent": coverage,
                "total_variants": n,
                "retained_variants": kept,
                "actual_coverage": kept / n,
                "rho": rho,
                "rho_full": rho_full,
                "delta_rho": rho - rho_full,
                "rho_50": rho_50,
                "delta_rho_50": delta_rho_50,
            }
            report_rows.append(report_data)

        ranking_columns = [
            "DMS_id", "source_row", "mutation", "entropy", "entropy_rank", "retained_at_50"
        ]
        ranked_tables.append(ranked_table[ranking_columns])

    curve = pd.DataFrame(report_rows)
    ranking = pd.concat(ranked_tables, ignore_index=True)
    return curve, ranking


def self_test():
    sample = pd.DataFrame({
        "DMS_id": ["A"] * 5 + ["B"] * 5,
        "source_row": [20, 10, 30, 40, 50] * 2,
        "mutation": ["A1C", "A2C", "A3C", "A4C", "A5C"] * 2,
        "esm2_score": [1, 2, 3, 4, 5] * 2,
        "DMS_score": [1, 2, 3, 5, 4] + [5, 4, 3, 2, 1],
        "entropy": [0.1, 0.1, 0.2, 0.8, 0.9] * 2,
    })

    curve, ranking = build_results(sample, [50, 100])
    assert retained_count(5, 50) == 3
    assert retained_count(5, 10) == 1

    assay_ranking = ranking[ranking["DMS_id"] == "A"]
    assert assay_ranking["source_row"].tolist() == [10, 20, 30, 40, 50]

    half_table = curve[curve["requested_coverage_percent"] == 50]
    assay_a = half_table[half_table["DMS_id"] == "A"].iloc[0]
    assay_b = half_table[half_table["DMS_id"] == "B"].iloc[0]

    assert assay_a["retained_variants"] == 3
    assert assay_a["actual_coverage"] == 0.6
    assert abs(assay_a["rho_50"] - 1.0) < 1e-10
    assert abs(assay_b["rho_50"] - (-1.0)) < 1e-10
    assert abs(assay_a["rho_full"] - 0.9) < 1e-10
    assert abs(assay_a["delta_rho_50"] - 0.1) < 1e-10
    assert len(curve) == 4


    reordered_sample = sample.sort_values(["DMS_id", "source_row"], ascending=False)
    second_curve, second_ranking = build_results(reordered_sample, [50, 100])
    assert curve.equals(second_curve)
    assert ranking.equals(second_ranking)

    print("Synthetic abstention self-check: PASS")




if __name__ == "__main__":
    self_test()

    if "--self-test" not in sys.argv:
        scored_table = pd.read_csv(SCORED)
        curve, ranking = build_results(scored_table, coverage_percent)

        curve.to_csv(CURVE, index=False)
        ranking.to_csv(RANKING, index=False)

        print(f"Wrote {CURVE}: {len(curve)} assay/coverage results")
        print(f"Wrote {RANKING}: {len(ranking)} ranked variants")
        print("50% retention uses ceil(n/2); ties use ascending source_row.")
