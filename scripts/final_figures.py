# Entropy-Gated ESM-2 Final Analysis

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
outputs = PROJECT_ROOT / "outputs"

headline = pd.read_csv(outputs / "headline_table.csv")
coverage = pd.read_csv(outputs / "headline_coverage.csv")
per_assay = pd.read_csv(outputs / "headline_per_assay.csv")
confound = pd.read_csv(outputs / "confound_assay_summary.csv")
balance = pd.read_csv(outputs / "controlled_balance.csv")
reproduction = pd.read_csv(outputs / "reproduction_all.csv")

usable_assays = balance[balance["usable"] == True]["DMS_id"].tolist()
if len(usable_assays) == 0:
    raise ValueError("No assays passed the variance-matched control.")

groups = [
    ("uncontrolled_all", "All assays"),
    ("uncontrolled_control_eligible", "Before matching"),
    ("controlled_usable", "After matching"),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for i in range(3):
    group_name, title = groups[i]
    rows = coverage[coverage["analysis"] == group_name]
    rows = rows.sort_values("requested_coverage_percent")
    if rows.empty:
        raise ValueError("Missing coverage results for " + group_name)

    x = rows["mean_actual_coverage"].to_numpy() * 100
    entropy_risk = 1 - rows["mean_entropy_rho"].to_numpy()
    random_risk = 1 - rows["random_rho_median"].to_numpy()

    random_low = 1 - rows["random_rho_q975"].to_numpy()
    random_high = 1 - rows["random_rho_q025"].to_numpy()

    ax = axes[0, i]
    ax.fill_between(x, random_low, random_high, color="gray", alpha=0.25,
                    label="Random 95% band")
    ax.plot(x, random_risk, color="gray", linestyle="--",
            label="Random median")
    ax.plot(x, entropy_risk, color="teal", marker="o",
            label="Entropy filter")
    ax.axvline(50, color="lightgray", linestyle=":")
    ax.set_title(title)
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Risk (1 - Spearman)")
    ax.set_xlim(5, 103)
    ax.grid(alpha=0.2)

axes[0, 0].legend(fontsize=8)

confound = confound.sort_values("spearman_entropy_vs_mean_abs_effect")
bar_colors = []
for assay in confound["DMS_id"]:
    if assay in usable_assays:
        bar_colors.append("teal")
    else:
        bar_colors.append("lightgray")

ax = axes[1, 0]
ax.barh(range(len(confound)),
        confound["spearman_entropy_vs_mean_abs_effect"], color=bar_colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks([])
ax.set_xlabel("Within-assay Spearman rho")
ax.set_title("Entropy vs. mutation effect")
ax.text(0.02, 0.04, "Teal = assays retained by matching",
        transform=ax.transAxes, fontsize=8)
ax.grid(axis="x", alpha=0.2)

ax = axes[1, 1]
for assay in usable_assays:
    before_rows = per_assay[
        (per_assay["analysis"] == "uncontrolled_control_eligible")
        & (per_assay["DMS_id"] == assay)
    ]
    after_rows = per_assay[
        (per_assay["analysis"] == "controlled_usable")
        & (per_assay["DMS_id"] == assay)
    ]
    before = before_rows["excess_over_random"].iloc[0]
    after = after_rows["excess_over_random"].iloc[0]
    ax.plot([0, 1], [before, after], color="gray", marker="o")

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks([0, 1], ["Before", "After"])
ax.set_xlim(-0.3, 1.3)
ax.set_ylabel("Entropy delta rho minus mean random delta rho")
ax.set_title("Before vs. after matching")
ax.grid(axis="y", alpha=0.2)

usable_balance = balance[balance["usable"] == True]
original_count = int(usable_balance["original_variants"].sum())
matched_count = int(usable_balance["matched_variants"].sum())

ax = axes[1, 2]
ax.barh([1, 0], [original_count, matched_count],
        color=["gray", "teal"])
ax.set_yticks([1, 0], ["Before", "Matched"])
ax.set_xlabel("Number of variants")
ax.set_title("Variants left after matching")
ax.set_xlim(0, original_count * 1.25)
ax.text(original_count, 1, f"  {original_count:,}", va="center")
ax.text(matched_count, 0, f"  {matched_count:,}", va="center")
ax.grid(axis="x", alpha=0.2)

fig.suptitle("Entropy filtering results", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.95])

figure_file = outputs / "final_analysis.png"
fig.savefig(figure_file, dpi=200)
plt.close(fig)

selected_names = [
    "uncontrolled_all", "uncontrolled_control_eligible", "controlled_usable"
]

headline_rows = headline[headline["analysis"].isin(selected_names)]
print("Reproduction results:")
print(reproduction["result"].value_counts().to_string())
print("\nMain results:")
print(headline_rows.to_string(index=False))
print(f"\nMatching kept {len(usable_assays)} of {len(balance)} assays and "
      f"{matched_count:,} of {original_count:,} variants.")

print("Figure saved to", figure_file)
