# ProteinGym Assay Selection

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
METADATA = DATA_RAW / "DMS_substitutions.csv"
ASSAYS_DIR = DATA_RAW / "DMS_ProteinGym_substitutions"
BENCHMARKS = DATA_RAW / "DMS_substitutions_Spearman_DMS_level.csv"
CANDIDATE_OUTPUT = PROJECT_ROOT / "outputs" / "candidate_pool.csv"
EXCLUSIONS_OUTPUT = PROJECT_ROOT / "outputs" / "excluded_candidates.csv"
SELECTED_CANDIDATES = PROJECT_ROOT / "outputs" / "final_selection.csv"

metadata_table = pd.read_csv(METADATA)
benchmark_table = pd.read_csv(BENCHMARKS)

downloaded_assays = set()
metadata_assays = set()
amount_downloaded = 0

for assays in ASSAYS_DIR.glob("*.csv"):
    amount_downloaded += 1
    downloaded_assays.add(assays.name)

metadata_assays.update(metadata_table["DMS_filename"].dropna())

amino_acids = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

valid_seq = 0

for value in metadata_table["target_seq"]:
    if not isinstance(value, str):
        continue
    if value == "":
        continue

    valid = True

    for i in value:
        if i not in amino_acids:
            valid = False
            break

    if valid:
        valid_seq += 1

metadata_dms = set()
benchmark_dms = set()

metadata_dms.update(metadata_table["DMS_id"].dropna())
benchmark_dms.update(benchmark_table["DMS ID"].dropna())

import random
import math

seed = 42
min_var = 100
selection = 20

candidate_rows = []
excluded_rows = []

for index, metadata_row in metadata_table.iterrows():
    dms_id = metadata_row["DMS_id"]
    dms_filename = metadata_row["DMS_filename"]
    uniprot_id = metadata_row["UniProt_ID"]
    target_seq = metadata_row["target_seq"]

    exclusion_reasons = []
    esm2_spearman = None
    eligible_var_count = 0
    eligibility = False

    if pd.isna(dms_filename):
        exclusion_reasons.append("missing_filename")
    else:
        metadata_match_count = (metadata_table["DMS_filename"] == dms_filename).sum()

        if dms_filename not in downloaded_assays:
            exclusion_reasons.append("missing_assay_file")

        if metadata_match_count != 1:
            exclusion_reasons.append("metadata_match_not_unique")

    target_valid = True

    if not isinstance(target_seq, str):
        target_valid = False

    elif target_seq == "":
        target_valid = False

    else:
        for amino_acid in target_seq:
            if amino_acid not in amino_acids:
                target_valid = False
                break

    if not target_valid:
        exclusion_reasons.append("invalid_target_seq")

    if pd.isna(uniprot_id):
        exclusion_reasons.append("missing_uniprot_id")

    matching_benchmark_rows = benchmark_table[
        benchmark_table["DMS ID"] == dms_id
    ]

    if len(matching_benchmark_rows) == 0:
        exclusion_reasons.append("missing_esm2_benchmark")

    elif len(matching_benchmark_rows) > 1:
        exclusion_reasons.append("multiple_esm2_benchmarks")

    else:
        esm2_spearman = matching_benchmark_rows.iloc[0][
            "ESM2 (650M)"
        ]

        if pd.isna(esm2_spearman):
            exclusion_reasons.append("missing_esm2_benchmark")

    assay_path = ASSAYS_DIR / str(dms_filename)

    if assay_path.exists() and target_valid:
        assay_table = pd.read_csv(assay_path)

        if "mutant" not in assay_table.columns:
            exclusion_reasons.append("missing_mutant_column")

        elif "DMS_score" not in assay_table.columns:
            exclusion_reasons.append("missing_dms_score_column")

        else:
            eligibility = True
            eligible_mutations = set()

            for variant_index, variant_row in assay_table.iterrows():
                mutant_value = variant_row["mutant"]
                dms_score = variant_row["DMS_score"]

                if not isinstance(mutant_value, str):
                    continue

                if ":" in mutant_value:
                    continue

                if len(mutant_value) < 3:
                    continue

                wild_type = mutant_value[0]
                mutant_amino_acid = mutant_value[-1]
                position_text = mutant_value[1:-1]

                if wild_type not in amino_acids:
                    continue

                if mutant_amino_acid not in amino_acids:
                    continue

                if wild_type == mutant_amino_acid:
                    continue

                if not position_text.isdigit():
                    continue

                position = int(position_text)

                if position < 1:
                    continue

                if position > len(target_seq):
                    continue

                if target_seq[position - 1] != wild_type:
                    continue

                if pd.isna(dms_score):
                    continue

                try:
                    numeric_dms_score = float(dms_score)
                except (TypeError, ValueError):
                    continue

                if not math.isfinite(numeric_dms_score):
                    continue

                eligible_mutations.add(mutant_value)

            eligible_var_count = len(eligible_mutations)

    if eligibility:
        if eligible_var_count < min_var:
            exclusion_reasons.append("fewer_than_100_eligible_variants")

    if len(exclusion_reasons) == 0:
        candidate_rows.append(
            {
                "DMS_id": dms_id,
                "DMS_filename": dms_filename,
                "UniProt_ID": uniprot_id,
                "target_seq": target_seq,
                "seq_len": len(target_seq),
                "eligible_variant_count": eligible_var_count,
                "published_esm2_spearman": esm2_spearman,
            }
        )

    else:
        excluded_rows.append(
            {
                "DMS_id": dms_id,
                "DMS_filename": dms_filename,
                "UniProt_ID": uniprot_id,
                "eligible_variant_count": eligible_var_count,
                "exclusion_reason": ";".join(exclusion_reasons),
            }
        )

candidate_table = pd.DataFrame(candidate_rows)

excluded_table = pd.DataFrame(
    excluded_rows,
    columns=[
        "DMS_id",
        "DMS_filename",
        "UniProt_ID",
        "eligible_variant_count",
        "exclusion_reason",
    ],
)

unique_proteins = sorted(candidate_table["UniProt_ID"].unique())

if len(unique_proteins) < selection:
    raise ValueError(
        f"Only {len(unique_proteins)} eligible proteins were found. "
        f"At least {selection} are required."
    )

random_generator = random.Random(seed)

selected_proteins = random_generator.sample(unique_proteins, selection)

selected_rows = []

for protein in selected_proteins:
    protein_assays = candidate_table[
        candidate_table["UniProt_ID"] == protein
    ]

    protein_assays = protein_assays.sort_values("DMS_id")
    protein_assays = protein_assays.reset_index(drop=True)

    selected_assay_index = random_generator.randrange(
        len(protein_assays)
    )

    selected_assay = protein_assays.iloc[selected_assay_index].to_dict()



    selected_rows.append(selected_assay)

selected_table = pd.DataFrame(selected_rows)
selected_table["random_seed"] = seed
selected_table = selected_table.sort_values("DMS_id")

CANDIDATE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

candidate_table.to_csv(CANDIDATE_OUTPUT, index=False)

excluded_table.to_csv(EXCLUSIONS_OUTPUT, index=False,)
selected_table.to_csv(SELECTED_CANDIDATES, index=False)


print(f"Downloaded assay files: {amount_downloaded}")
print(f"Valid target sequences: {valid_seq}")
print(f"Metadata IDs with benchmarks: {len(metadata_dms & benchmark_dms)}")
print(f"Eligible assays: {len(candidate_table)}")
print(f"Excluded assays: {len(excluded_table)}")
print(f"Eligible unique proteins: {len(unique_proteins)}")
print(f"Selected assays: {len(selected_table)}")
print(f"Selected unique proteins: {selected_table['UniProt_ID'].nunique()}")
print(f"Random seed: {seed}")
print()
print(f"Saved candidate pool to: {CANDIDATE_OUTPUT}")
print(f"Saved exclusions to: {EXCLUSIONS_OUTPUT}")
print(f"Saved final selection to: {SELECTED_CANDIDATES}")