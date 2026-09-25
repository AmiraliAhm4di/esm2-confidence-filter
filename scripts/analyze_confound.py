# Entropy and experimental effect size by position

from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
SCORED = OUTPUTS / "scored_variants_all.csv"
POSITION_OUTPUT = OUTPUTS / "position_confound.csv"
SUMMARY_OUTPUT = OUTPUTS / "confound_assay_summary.csv"
SCATTER_OUTPUT = OUTPUTS / "entropy_effect_scatter.png"
DISTRIBUTION_OUTPUT = OUTPUTS / "entropy_effect_distributions.png"

scored_table = pd.read_csv(SCORED)

if scored_table["position"].isna().any():
    raise ValueError("Some scored variants have no position")

scored_table["abs_DMS_score"] = scored_table["DMS_score"].abs()

position_rows = []
position_groups = scored_table.groupby(["DMS_id", "position"])

for (assay, position), position_table in position_groups:
    report_data = {
        "DMS_id": assay,
        "position": position,
        "n_variants": len(position_table),
        "mean_abs_DMS_score": position_table["abs_DMS_score"].mean(),
        "sd_DMS_score": np.std(position_table["DMS_score"], ddof=0),
        "entropy": position_table["entropy"].iloc[0],
    }
    position_rows.append(report_data)

position_summary = pd.DataFrame(position_rows)


assays = position_summary["DMS_id"].unique().tolist()
assays.sort()

if len(assays) != 20:
    raise ValueError("Need 20 scored assays for this analysis")

summary_rows = []


for assay in assays:
    assay_table = position_summary[position_summary["DMS_id"] == assay]
    correlation = spearmanr(assay_table["entropy"], assay_table["mean_abs_DMS_score"])

    report_data = {
        "DMS_id": assay,
        "n_positions": len(assay_table),
        "n_variants": assay_table["n_variants"].sum(),
        "spearman_entropy_vs_mean_abs_effect": correlation.statistic,
    }
    summary_rows.append(report_data)

summary_table = pd.DataFrame(summary_rows)
position_summary.to_csv(POSITION_OUTPUT, index=False)
summary_table.to_csv(SUMMARY_OUTPUT, index=False)


# One scatter plot for each assay
plt.figure(figsize=(17, 16))

for index, assay in enumerate(assays):
    assay_table = position_summary[position_summary["DMS_id"] == assay]
    plt.subplot(5, 4, index + 1)
    plt.scatter(assay_table["entropy"], assay_table["mean_abs_DMS_score"], s=7, alpha=0.35)

    rho = summary_table[summary_table["DMS_id"] == assay]["spearman_entropy_vs_mean_abs_effect"].iloc[0]
    plt.title(f"{assay}\nn={len(assay_table)} positions; rho={rho:.2f}", fontsize=8)
    plt.xlabel("Positional entropy", fontsize=8)
    plt.ylabel("Mean |DMS_score|", fontsize=8)
    plt.tick_params(labelsize=7)

plt.suptitle("Entropy vs. experimental effect magnitude, one point per position", fontsize=13)
plt.tight_layout()
plt.savefig(SCATTER_OUTPUT, dpi=180)
plt.close()

entropy_groups = []
effect_groups = []
for assay in assays:
    assay_table = position_summary[position_summary["DMS_id"] == assay]
    entropy_groups.append(assay_table["entropy"].to_numpy())
    effect_groups.append(assay_table["mean_abs_DMS_score"].to_numpy())


plt.figure(figsize=(16, 9))
plt.subplot(1, 2, 1)
plt.boxplot(entropy_groups, vert=False)
plt.yticks(range(1, 21), assays, fontsize=8)
plt.title("Positional entropy")

plt.subplot(1, 2, 2)
plt.boxplot(effect_groups, vert=False)
plt.yticks(range(1, 21), assays, fontsize=8)
plt.title("Position mean |DMS_score|")

plt.suptitle("Per-assay distributions; each position has equal weight", fontsize=13)
plt.tight_layout()
plt.savefig(DISTRIBUTION_OUTPUT, dpi=180)
plt.close()


print(f"Saved {len(position_summary)} positions across {len(assays)} assays to {OUTPUTS}")
