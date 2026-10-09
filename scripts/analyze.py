# Abstention analysis

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"

seed = 42
bootstrap_samples = 10000
random_draws = 1000
coverages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def bootstrap_ci(values):
    values = np.array(values, dtype=float)
    if len(values) < 2:
        raise ValueError("Need at least two assays for a bootstrap interval")
    if not np.isfinite(values).all():
        raise ValueError("Need finite values for a bootstrap interval")

    rng = np.random.default_rng(seed)
    means = []
    for repeat in range(bootstrap_samples):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))

    low, high = np.quantile(means, [0.025, 0.975])
    return low, high
def calculate_aurc(rhos, actual_coverage):
    area = 0
    for i in range(1, len(rhos)):
        left_risk = 1 - rhos[i - 1]
        right_risk = 1 - rhos[i]
        width = actual_coverage[i] - actual_coverage[i - 1]
        area += (left_risk + right_risk) * width / 2
    return area


def main():

    reproduction = pd.read_csv(OUTPUTS / "reproduction_all.csv")
    balance = pd.read_csv(OUTPUTS / "controlled_balance.csv")
    entropy = pd.read_csv(OUTPUTS / "entropy_coverage.csv")
    random_results = pd.read_csv(OUTPUTS / "random_null_draws.csv")
    controlled_entropy = pd.read_csv(OUTPUTS / "controlled_entropy_coverage.csv")
    controlled_random = pd.read_csv(OUTPUTS / "controlled_random_null_draws.csv")
    confound = pd.read_csv(OUTPUTS / "confound_assay_summary.csv")

    passed = reproduction[reproduction["result"] == "PASS"]
    usable = balance[balance["usable"] == True]
    all_assays = sorted(reproduction["DMS_id"].tolist())
    controlled_assays = sorted(usable["DMS_id"].tolist())
    if len(controlled_assays) < 2:
        raise ValueError("Need at least two usable assays after matching")

    groups = {
        "uncontrolled_all": all_assays,
        "uncontrolled_control_eligible": controlled_assays,
        "controlled_usable": controlled_assays,
    }
    if len(passed) >= 2:
        groups["uncontrolled_benchmark_pass"] = sorted(passed["DMS_id"].tolist())

    headline_rows = []
    assay_rows = []
    coverage_rows = []

    for analysis, assays in groups.items():
        if analysis == "controlled_usable":
            curve_table = controlled_entropy
            draw_table = controlled_random
        else:
            curve_table = entropy
            draw_table = random_results

        rng = np.random.default_rng(seed)

        group_rows = []
        random_delta_sum = np.zeros(random_draws)
        random_aurc_sum = np.zeros(random_draws)
        random_rho_sum = {}
        entropy_rhos = {}
        actual_coverages = {}

        for coverage in coverages:
            random_rho_sum[coverage] = np.zeros(random_draws)
            entropy_rhos[coverage] = []
            actual_coverages[coverage] = []

        for assay in assays:
            curve = curve_table[curve_table["DMS_id"] == assay]
            curve = curve.sort_values("requested_coverage_percent")
            draws = draw_table[draw_table["DMS_id"] == assay]
            full_row = curve[curve["requested_coverage_percent"] == 100].iloc[0]
            half_row = curve[curve["requested_coverage_percent"] == 50].iloc[0]
            rho_full = full_row["rho"]
            delta = half_row["rho"] - rho_full
            x = curve["actual_coverage"].to_numpy()
            rhos = curve["rho"].to_numpy()
            entropy_aurc = calculate_aurc(rhos, x)

            # keep random curve together
            draw_order = rng.permutation(random_draws)
            random_rhos = {}
            for coverage in coverages:
                saved = draws[draws["requested_coverage_percent"] == coverage]
                saved = saved.sort_values("draw_id")
                if saved["draw_id"].tolist() != list(range(1, random_draws + 1)):
                    raise ValueError(assay + ": invalid draw IDs at " + str(coverage) + "%")
                values = saved["rho"].to_numpy()
                if not np.isfinite(values).all():
                    raise ValueError(assay + ": nonfinite random correlations at " + str(coverage) + "%")
                random_rhos[coverage] = values[draw_order]
                random_rho_sum[coverage] += random_rhos[coverage]

                row = curve[curve["requested_coverage_percent"] == coverage].iloc[0]
                entropy_rhos[coverage].append(row["rho"])

                actual_coverages[coverage].append(row["actual_coverage"])

            random_delta = random_rhos[50] - rho_full
            random_aurcs = []
            for draw in range(random_draws):
                draw_rhos = []
                for coverage in coverages:
                    draw_rhos.append(random_rhos[coverage][draw])
                random_aurcs.append(calculate_aurc(draw_rhos, x))

            random_delta_sum += random_delta
            random_aurc_sum += np.array(random_aurcs)
            random_mean = np.mean(random_delta)
            row = {
                "analysis": analysis,
                "DMS_id": assay,
                "total_variants": full_row["total_variants"],
                "retained_variants_50": half_row["retained_variants"],
                "actual_coverage_50": half_row["actual_coverage"],
                "rho_full": rho_full,
                "rho_50": half_row["rho"],
                "delta_rho_50": delta,
                "random_delta_mean": random_mean,
                "excess_over_random": delta - random_mean,
                "entropy_aurc": entropy_aurc,
                "random_aurc_mean": np.mean(random_aurcs),
            }
            group_rows.append(row)
            assay_rows.append(row)

        report = pd.DataFrame(group_rows)
        n = len(assays)
        aggregate_delta = random_delta_sum / n
        aggregate_aurc = random_aurc_sum / n
        null_low, null_high = np.quantile(aggregate_delta, [0.025, 0.975])
        delta_low, delta_high = bootstrap_ci(report["delta_rho_50"])
        excess_low, excess_high = bootstrap_ci(report["excess_over_random"])
        aurc_low, aurc_high = bootstrap_ci(report["entropy_aurc"])
        random_auc_low, random_auc_high = np.quantile(aggregate_aurc, [0.025, 0.975])
        mean_delta = report["delta_rho_50"].mean()
        # must beat zero and random
        positive = delta_low > 0 and excess_low > 0 and mean_delta > null_high
        if positive:
            outcome = "positive"
        elif delta_high < 0:
            outcome = "negative"
        else:
            outcome = "null_or_inconclusive"

        headline_rows.append({
            "analysis": analysis,
            "n_assays": n,
            "mean_rho_full": report["rho_full"].mean(),
            "mean_rho_50": report["rho_50"].mean(),
            "mean_delta_rho_50": mean_delta,
            "delta_ci_low": delta_low,
            "delta_ci_high": delta_high,
            "random_delta_mean": np.mean(aggregate_delta),
            "random_delta_q025": null_low,
            "random_delta_q975": null_high,
            "mean_excess_over_random": report["excess_over_random"].mean(),
            "excess_ci_low": excess_low,
            "excess_ci_high": excess_high,
            "mean_entropy_aurc": report["entropy_aurc"].mean(),
            "aurc_ci_low": aurc_low,
            "aurc_ci_high": aurc_high,
            "mean_random_aurc": np.mean(aggregate_aurc),
            "random_aurc_q025": random_auc_low,
            "random_aurc_q975": random_auc_high,
            "positive": positive,
            "outcome": outcome,
        })

        for coverage in coverages:
            mean_random_rhos = random_rho_sum[coverage] / n
            low, high = np.quantile(mean_random_rhos, [0.025, 0.975])
            mean_rho = np.mean(entropy_rhos[coverage])
            coverage_rows.append({
                "analysis": analysis,
                "requested_coverage_percent": coverage,
                "mean_actual_coverage": np.mean(actual_coverages[coverage]),
                "mean_entropy_rho": mean_rho,
                "entropy_risk": 1 - mean_rho,
                "random_rho_median": np.median(mean_random_rhos),
                "random_rho_q025": low,
                "random_rho_q975": high,
            })

    headline = pd.DataFrame(headline_rows)
    per_assay = pd.DataFrame(assay_rows)
    all_result = headline[headline["analysis"] == "uncontrolled_all"].iloc[0]
    before = headline[headline["analysis"] == "uncontrolled_control_eligible"].iloc[0]
    after = headline[headline["analysis"] == "controlled_usable"].iloc[0]

    before_assays = per_assay[per_assay["analysis"] == "uncontrolled_control_eligible"]
    after_assays = per_assay[per_assay["analysis"] == "controlled_usable"]
    changes = []
    for assay in controlled_assays:
        before_row = before_assays[before_assays["DMS_id"] == assay].iloc[0]
        after_row = after_assays[after_assays["DMS_id"] == assay].iloc[0]
        changes.append(after_row["excess_over_random"] - before_row["excess_over_random"])
    change_low, change_high = bootstrap_ci(changes)

    confound_values = confound["spearman_entropy_vs_mean_abs_effect"]
    confound_low, confound_high = bootstrap_ci(confound_values)
    eligible_confound = confound[confound["DMS_id"].isin(controlled_assays)]
    eligible_values = eligible_confound["spearman_entropy_vs_mean_abs_effect"]
    eligible_low, eligible_high = bootstrap_ci(eligible_values)
    # ci doesn't cross zero
    association = eligible_low > 0 or eligible_high < 0

    proxy_pattern = False
    if before["positive"] and not after["positive"]:
        if change_high < 0 and association:
            proxy_pattern = True

    if after["positive"]:
        conclusion = "The entropy filter helped in this matched subset"
    elif proxy_pattern:
        conclusion = "the earlier gain got smaller after matching so entropy may be picking up which positions tolerate changes"
    
    elif after["outcome"] == "negative":
        conclusion = "the entropy filter made the rankings worse after matching"
    else:
        conclusion = "we did not see a clear improvement after matching"

    if change_low <= 0 <= change_high:
        change_interpretation = "The paired interval includes zero, so a change in the advantage after matching was not clearly established."
    elif change_low > 0:
        change_interpretation = "The paired interval supports a larger advantage after matching."
    else:
        change_interpretation = "The paired interval supports a smaller advantage after matching."

    memo = [
        (
            f"At full coverage the Spearman results were close enough to the published "
            f"ESM-2 results for {len(passed)} of {len(all_assays)} assays. The other "
            f"{len(all_assays) - len(passed)} could not be compared directly because "
            "their published results include multiple mutations."
        ),
        (
            f"Across all {len(all_assays)} assays before matching, the mean Spearman "
            f"change at 50% requested coverage was "
            f"{all_result['mean_delta_rho_50']:+.4f} "
            f"(95% assay-bootstrap interval [{all_result['delta_ci_low']:+.4f}, "
            f"{all_result['delta_ci_high']:+.4f}])."
        ),
        (
            f"For the same {len(controlled_assays)} assays, the mean Spearman change "
            f"at 50% requested coverage before matching was {before['mean_delta_rho_50']:+.4f}. "
            f"The 95% range of changes from random retention was [{before['random_delta_q025']:+.4f}, "
            f"{before['random_delta_q975']:+.4f}] before matching and "
            f"[{after['random_delta_q025']:+.4f}, {after['random_delta_q975']:+.4f}] "
            f"after matching. Entropy's advantage over random changed by "
            f"{np.mean(changes):+.4f} (95% assay-bootstrap interval "
            f"[{change_low:+.4f}, {change_high:+.4f}]). {change_interpretation}"
        ),
        (
            f"Within each assay, we compared entropy with the mean absolute experimental "
            f"score at each position which gave an average Spearman of "
            f"{confound_values.mean():+.4f} (95% assay-bootstrap interval "
            f"[{confound_low:+.4f}, {confound_high:+.4f}]), matching kept "
            f"{len(controlled_assays)} of {len(all_assays)} assays and "
            f"{int(usable['matched_variants'].sum())} of "
            f"{int(usable['original_variants'].sum())} variants from those assays."
        ),
        (
            f"Within the {len(controlled_assays)} matched assays, the mean Spearman "
            f"change at 50% requested coverage was "
            f"{after['mean_delta_rho_50']:+.4f} (95% assay-bootstrap interval "
            f"[{after['delta_ci_low']:+.4f}, {after['delta_ci_high']:+.4f}]), and mean "
            f"AURC over {coverages[0]}-{coverages[-1]}% requested coverage was "
            f"{after['mean_entropy_aurc']:.4f} versus {after['mean_random_aurc']:.4f} "
            f"for random retention (lower is better). {conclusion}."
        ),
    ]
    memo_1 = "\n".join(memo) + "\n"

    headline.to_csv(OUTPUTS / "headline_table.csv", index=False)
    per_assay.to_csv(OUTPUTS / "headline_per_assay.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(OUTPUTS / "headline_coverage.csv", index=False)
    with open(OUTPUTS / "results_memo.txt", "w", encoding="utf-8") as file:
        file.write(memo_1)

    print("Wrote", OUTPUTS / "headline_table.csv")
    print("Wrote", OUTPUTS / "results_memo.txt")


if __name__ == "__main__":
    main()
