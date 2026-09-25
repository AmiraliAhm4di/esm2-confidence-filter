# Random Abstention Tests

from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
import math
import numpy as np
from scripts.abstain import build_results
from scripts.random_null import build_random_null


def example_inputs():
    sample = {
        "DMS_id": [],
        "source_row": [],
        "mutation": [],
        "entropy": [],
        "esm2_score": [],
        "DMS_score": [],
    }

    for assay in ["A", "B"]:
        for position in range(1, 21):
            if assay == "A":
                experimental_score = position
                if position == 19:
                    experimental_score = 20
                if position == 20:
                    experimental_score = 19
            else:
                experimental_score = 21 - position

            sample["DMS_id"].append(assay)
            sample["source_row"].append(21 - position)
            sample["mutation"].append("A" + str(position) + "C")
            sample["entropy"].append(position / 20)
            sample["esm2_score"].append(position)
            sample["DMS_score"].append(experimental_score)

    scored_table = pd.DataFrame(sample)
    coverages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    entropy_curve, ranking = build_results(scored_table, coverages)
    return scored_table, entropy_curve



def test_draw_count_and_matched_coverage():
    scored_table, entropy_curve = example_inputs()
    draws_table, summary_table = build_random_null(scored_table, entropy_curve)

    assert len(draws_table) == 20000
    assert len(summary_table) == 20

    for assay in ["A", "B"]:
        for coverage in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            assay_draws = draws_table[draws_table["DMS_id"] == assay]
            selected_draws = assay_draws[
                assay_draws["requested_coverage_percent"] == coverage
            ]
            assay_entropy = entropy_curve[entropy_curve["DMS_id"] == assay]
            entropy_row = assay_entropy[
                assay_entropy["requested_coverage_percent"] == coverage
            ].iloc[0]


            assert len(selected_draws) == 1000
            assert selected_draws["draw_id"].tolist() == list(range(1, 1001))
            assert (selected_draws["retained_variants"] == entropy_row["retained_variants"]).all()


            if coverage == 100:
                assert (selected_draws["delta_rho"] == 0).all()


def test_same_seed_gives_same_results():
    scored_table, entropy_curve = example_inputs()
    first_draws, first_summary = build_random_null(scored_table, entropy_curve, 11, 42)
    reordered_table = scored_table.sort_values(["DMS_id", "source_row"], ascending=False)
    reordered_curve = entropy_curve.sort_values(
        ["DMS_id", "requested_coverage_percent"], ascending=False
    )
    second_draws, second_summary = build_random_null(reordered_table, reordered_curve, 11, 42)

    assert first_draws.equals(second_draws)
    assert first_summary.equals(second_summary)


def test_null_band_uses_the_saved_draws():
    scored_table, entropy_curve = example_inputs()
    draws_table, summary_table = build_random_null(scored_table, entropy_curve, 11, 42)

    assay_draws = draws_table[draws_table["DMS_id"] == "A"]
    selected_draws = assay_draws[assay_draws["requested_coverage_percent"] == 50]
    assay_summary = summary_table[summary_table["DMS_id"] == "A"]
    summary_row = assay_summary[assay_summary["requested_coverage_percent"] == 50].iloc[0]

    changes = selected_draws["delta_rho"].to_numpy()
    values = selected_draws["rho"].to_numpy()

    assert abs(summary_row["random_rho_median"] - np.median(values)) < 1e-10
    assert abs(summary_row["random_rho_q025"] - np.quantile(values, 0.025)) < 1e-10
    assert abs(summary_row["random_rho_q975"] - np.quantile(values, 0.975)) < 1e-10
    assert abs(summary_row["random_delta_rho_q025"] - np.quantile(changes, 0.025)) < 1e-10
    assert abs(summary_row["random_delta_rho_q975"] - np.quantile(changes, 0.975)) < 1e-10
