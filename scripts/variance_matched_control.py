# Variance-matched control

from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
import math
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
SCORED = OUTPUTS / "scored_variants_all.csv"

coverage_percent = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
draws = 1000
seed = 42


def retained_count(n, coverage):
    number_to_keep = n * coverage / 100
    number_to_keep = math.ceil(number_to_keep)
    return number_to_keep


def match_positions(position_summary):
    position_summary = position_summary[[
        "DMS_id", "position", "entropy", "mean_abs_DMS_score"
    ]].copy()
    assays = position_summary["DMS_id"].unique().tolist()
    assays.sort()
    rng = np.random.default_rng(seed)
    tables = []

    for assay in assays:
        assay_table = position_summary[position_summary["DMS_id"] == assay].copy()
        assay_table = assay_table.sort_values("position").reset_index(drop=True)
        n = len(assay_table)

        assay_table["entropy_quartile"] = 0
        order = assay_table.sort_values(["entropy", "position"]).index.tolist()
        for rank, row_number in enumerate(order):
            assay_table.loc[row_number, "entropy_quartile"] = (4 * rank) // n

        assay_table["effect_quartile"] = 0
        order = assay_table.sort_values(["mean_abs_DMS_score", "position"]).index.tolist()
        for rank, row_number in enumerate(order):
            assay_table.loc[row_number, "effect_quartile"] = (4 * rank) // n

        assay_table["selected"] = False
        assay_table["cell_target"] = 0

        for effect_group in range(4):
            counts = []
            for entropy_group in range(4):
                cell = assay_table[
                    (assay_table["effect_quartile"] == effect_group)
                    & (assay_table["entropy_quartile"] == entropy_group)
                ]
                counts.append(len(cell))
            number_to_keep = min(counts)

            for entropy_group in range(4):
                cell = assay_table[
                    (assay_table["effect_quartile"] == effect_group)
                    & (assay_table["entropy_quartile"] == entropy_group)
                ]
                assay_table.loc[cell.index, "cell_target"] = number_to_keep
                if number_to_keep > 0:
                    candidates = cell.sort_values("position").index.to_numpy()
                    chosen = rng.choice(candidates, size=number_to_keep, replace=False)
                    assay_table.loc[chosen, "selected"] = True

        tables.append(assay_table)

    return pd.concat(tables, ignore_index=True)


if __name__ == "__main__":
    scored_table = pd.read_csv(SCORED)
    if scored_table["position"].isna().any():
        raise ValueError("Some scored variants have no position")

    position_rows = []
    position_groups = scored_table.groupby(["DMS_id", "position"])
    for (assay, position), position_table in position_groups:
        if not np.isfinite(position_table["entropy"]).all() or position_table["entropy"].nunique() != 1:
            raise ValueError(assay + ": inconsistent or nonfinite entropy at position " + str(position))
        position_rows.append([
            assay,
            position,
            position_table["entropy"].iloc[0],
            position_table["DMS_score"].abs().mean(),
        ])
    position_summary = pd.DataFrame(position_rows, columns=[
        "DMS_id", "position", "entropy", "mean_abs_DMS_score"
    ])

    selection_table = match_positions(position_summary)
    selected_positions = selection_table[selection_table["selected"]]
    matched_table = scored_table.merge(
        selected_positions[["DMS_id", "position"]],
        on=["DMS_id", "position"], how="inner"
    )

    balance_rows = []
    curve_rows = []
    report_rows = []
    summary_rows = []
    assays = scored_table["DMS_id"].unique().tolist()
    assays.sort()

    for assay in assays:
        all_positions = selection_table[selection_table["DMS_id"] == assay]
        assay_positions = selected_positions[selected_positions["DMS_id"] == assay]
        all_variants = scored_table[scored_table["DMS_id"] == assay]
        assay_table = matched_table[matched_table["DMS_id"] == assay]

        assay_positions = assay_positions.sort_values(["entropy", "position"])
        half = len(assay_positions) // 2
        low_positions = assay_positions.iloc[:half]
        high_positions = assay_positions.iloc[half:]
        low_variants = assay_table[
            assay_table["position"].isin(low_positions["position"])
        ]["DMS_score"]
        high_variants = assay_table[
            assay_table["position"].isin(high_positions["position"])
        ]["DMS_score"]

        if half > 0:
            low_effects = low_positions["mean_abs_DMS_score"].to_numpy()
            high_effects = high_positions["mean_abs_DMS_score"].to_numpy()
            low_mean = float(np.mean(low_effects))
            high_mean = float(np.mean(high_effects))
            low_effect_variance = float(np.var(low_effects))
            high_effect_variance = float(np.var(high_effects))
            pooled_sd = math.sqrt((low_effect_variance + high_effect_variance) / 2)
            difference = high_mean - low_mean
            if pooled_sd > 0:
                smd = abs(difference) / pooled_sd
            elif difference == 0:
                smd = 0.0
            else:
                smd = math.inf
        else:
            low_mean = math.nan
            high_mean = math.nan
            smd = math.nan

        if len(low_variants) > 0:
            low_variance = float(np.var(low_variants))
        else:
            low_variance = math.nan
        if len(high_variants) > 0:
            high_variance = float(np.var(high_variants))
        else:
            high_variance = math.nan

        if low_variance == 0 and high_variance == 0:
            variance_ratio = 1.0
        elif low_variance == 0:
            variance_ratio = math.inf
        else:
            variance_ratio = high_variance / low_variance

        reasons = []
        if len(low_positions) < 10 or len(high_positions) < 10:
            reasons.append("FEWER_THAN_10_POSITIONS_PER_HALF")
        if len(assay_table) < 100:
            reasons.append("FEWER_THAN_100_VARIANTS")
        if pd.isna(smd) or smd > 0.1:
            reasons.append("EFFECT_SMD_GT_0.1")
        if pd.isna(variance_ratio) or variance_ratio < 0.8 or variance_ratio > 1.25:
            reasons.append("DMS_VARIANCE_RATIO_OUTSIDE_0.8_TO_1.25")

        balance_pass = len(reasons) == 0
        problem = ";".join(reasons)

        if balance_pass:
            ranked_table = assay_table.sort_values(["entropy", "source_row"])
            ranked_table = ranked_table.reset_index(drop=True)
            random_table = assay_table.sort_values("source_row")
            random_table = random_table.reset_index(drop=True)
            n = len(assay_table)

            correlation = spearmanr(ranked_table["esm2_score"], ranked_table["DMS_score"])
            rho_full = float(correlation.statistic)
            if pd.isna(rho_full):
                problem = "UNDEFINED_FULL_SPEARMAN"

            entropy_rho = {}
            assay_curve = []
            assay_draws = []
            if problem == "":
                for coverage in coverage_percent:
                    kept = retained_count(n, coverage)
                    retained_table = ranked_table.iloc[:kept]
                    correlation = spearmanr(retained_table["esm2_score"], retained_table["DMS_score"])
                    rho = float(correlation.statistic)
                    if pd.isna(rho):
                        problem = "UNDEFINED_ENTROPY_SPEARMAN_AT_" + str(coverage)
                        break
                    entropy_rho[coverage] = rho
                    assay_curve.append([
                        assay, coverage, n, kept, kept / n, rho, rho_full, rho - rho_full
                    ])

            if problem == "":
                random_rho = {}
                for coverage in coverage_percent:
                    random_rho[coverage] = []
                rng = np.random.default_rng(seed)

                for draw_id in range(1, draws + 1):
                    random_order = rng.permutation(n)
                    for coverage in coverage_percent:
                        kept = retained_count(n, coverage)
                        if coverage == 100:
                            rho = rho_full
                        else:
                            retained_rows = random_order[:kept]
                            retained_table = random_table.iloc[retained_rows]
                            correlation = spearmanr(retained_table["esm2_score"], retained_table["DMS_score"])
                            rho = float(correlation.statistic)
                        if pd.isna(rho):
                            problem = "UNDEFINED_RANDOM_SPEARMAN_AT_" + str(coverage) + "_DRAW_" + str(draw_id)
                            break
                        random_rho[coverage].append(rho)
                        assay_draws.append([
                            assay, coverage, draw_id, n, kept, kept / n, rho, rho - rho_full
                        ])
                    if problem != "":
                        break

            if problem == "":
                curve_rows.extend(assay_curve)
                report_rows.extend(assay_draws)
                for coverage in coverage_percent:
                    values = np.array(random_rho[coverage])
                    changes = values - rho_full
                    kept = retained_count(n, coverage)
                    summary_rows.append([
                        assay, coverage, n, kept, kept / n, draws, seed,
                        "variant_within_assay", False, rho_full,
                        entropy_rho[coverage], entropy_rho[coverage] - rho_full,
                        np.median(values), np.quantile(values, 0.025), np.quantile(values, 0.975),
                        np.median(changes), np.quantile(changes, 0.025), np.quantile(changes, 0.975),
                    ])

        balance_rows.append({
            "original_positions": len(all_positions),
            "matched_positions": len(assay_positions),
            "position_retention": len(assay_positions) / len(all_positions),
            "original_variants": len(all_variants),
            "matched_variants": len(assay_table),
            "variant_retention": len(assay_table) / len(all_variants),
            "low_entropy_positions": len(low_positions),
            "high_entropy_positions": len(high_positions),
            "low_entropy_variants": len(low_variants),
            "high_entropy_variants": len(high_variants),
            "low_mean_abs_effect": low_mean,
            "high_mean_abs_effect": high_mean,
            "abs_effect_smd": smd,
            "low_DMS_score_variance": low_variance,
            "high_DMS_score_variance": high_variance,
            "variance_ratio_high_over_low": variance_ratio,
            "balance_pass": balance_pass,
            "failure_reasons": problem,
            "DMS_id": assay,
            "usable": balance_pass and problem == "",
        })

    balance_table = pd.DataFrame(balance_rows)
    curve = pd.DataFrame(curve_rows, columns=[
        "DMS_id", "requested_coverage_percent", "total_variants",
        "retained_variants", "actual_coverage", "rho", "rho_full", "delta_rho",
    ])
    random_draws = pd.DataFrame(report_rows, columns=[
        "DMS_id", "requested_coverage_percent", "draw_id", "total_variants",
        "retained_variants", "actual_coverage", "rho", "delta_rho",
    ])
    random_summary = pd.DataFrame(summary_rows, columns=[
        "DMS_id", "requested_coverage_percent", "total_variants",
        "retained_variants", "actual_coverage", "draws", "seed",
        "sampling_unit", "with_replacement", "rho_full", "entropy_rho",
        "entropy_delta_rho", "random_rho_median", "random_rho_q025",
        "random_rho_q975", "random_delta_rho_median",
        "random_delta_rho_q025", "random_delta_rho_q975",
    ])

    balance_table.to_csv(OUTPUTS / "controlled_balance.csv", index=False)
    selection_table.to_csv(OUTPUTS / "controlled_position_selection.csv", index=False)
    curve.to_csv(OUTPUTS / "controlled_entropy_coverage.csv", index=False)
    random_draws.to_csv(OUTPUTS / "controlled_random_null_draws.csv", index=False)
    random_summary.to_csv(OUTPUTS / "controlled_random_null_summary.csv", index=False)

    print("Wrote the controlled results to", OUTPUTS)
    print(balance_table["usable"].sum(), "assays passed balance and had defined correlations")
