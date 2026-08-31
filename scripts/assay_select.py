# ProteinGym Assay Selection

from pathlib import Path
import random

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

for assay in ASSAYS_DIR.glob("*.csv"):
    downloaded_assays.add(assay.name)

amino_acids = set("ARNDCEQGHILKMFPSTWYV")
benchmark_scores = benchmark_table.set_index("DMS ID")["ESM2 (650M)"]

seed = 42
min_var = 100
selection = 20

candidate_rows = []
excluded_rows = []



# Check every assay
for index, metadata_row in metadata_table.iterrows():
    dms_id = metadata_row["DMS_id"]
    dms_filename = metadata_row["DMS_filename"]
    uniprot_id = metadata_row["UniProt_ID"]
    target_seq = metadata_row["target_seq"]

    exclusion_reason = None
    eligible_var_count = 0
    esm2_spearman = None

    if pd.isna(dms_filename) or dms_filename not in downloaded_assays:
        exclusion_reason = "missing_assay_file"

    elif not isinstance(target_seq, str) or not target_seq:
        exclusion_reason = "invalid_target_seq"

    elif not set(target_seq).issubset(amino_acids):
        exclusion_reason = "invalid_target_seq"

    elif pd.isna(uniprot_id):
        exclusion_reason = "missing_uniprot_id"

    elif dms_id not in benchmark_scores.index:
        exclusion_reason = "missing_esm2_benchmark"

    else:
        esm2_spearman = benchmark_scores[dms_id]

        if pd.isna(esm2_spearman):
            exclusion_reason = "missing_esm2_benchmark"

    if exclusion_reason is None:
        assay_table = pd.read_csv(ASSAYS_DIR / dms_filename)
        eligible_mutations = set()

        for variant_index, variant_row in assay_table.iterrows():
            mutant_value = variant_row["mutant"]
            dms_score = variant_row["DMS_score"]

            if not isinstance(mutant_value, str):
                continue

            if ":" in mutant_value or len(mutant_value) < 3:
                continue

            wild_type = mutant_value[0]
            mutant_amino_acid = mutant_value[-1]
            position_text = mutant_value[1:-1]

            if not position_text.isdigit():
                continue

            if wild_type not in amino_acids or mutant_amino_acid not in amino_acids:
                continue

            if wild_type == mutant_amino_acid:
                continue

            position = int(position_text)

            if not 1 <= position <= len(target_seq):
                continue

            if target_seq[position - 1] != wild_type or pd.isna(dms_score):
                continue

            eligible_mutations.add(mutant_value)

        eligible_var_count = len(eligible_mutations)

        if eligible_var_count < min_var:
            exclusion_reason = "fewer_than_100_eligible_variants"

    if exclusion_reason is None:
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
                "exclusion_reason": exclusion_reason,
            }
        )

candidate_table = pd.DataFrame(candidate_rows)
excluded_table = pd.DataFrame(excluded_rows)

unique_proteins = sorted(candidate_table["UniProt_ID"].unique())


if len(unique_proteins) < selection:
    raise ValueError("Not enough eligible proteins")

random.seed(seed)
selected_proteins = random.sample(unique_proteins, selection)

selected_rows = []

for protein in selected_proteins:
    protein_assays = candidate_table[
        candidate_table["UniProt_ID"] == protein
    ].sort_values("DMS_id")

    selected_assay_index = random.randrange(len(protein_assays))
    selected_rows.append(protein_assays.iloc[selected_assay_index].to_dict())

selected_table = pd.DataFrame(selected_rows)
selected_table["random_seed"] = seed
selected_table = selected_table.sort_values("DMS_id")


CANDIDATE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

candidate_table.to_csv(CANDIDATE_OUTPUT, index=False)
excluded_table.to_csv(EXCLUSIONS_OUTPUT, index=False)
selected_table.to_csv(SELECTED_CANDIDATES, index=False)

print(f"""Eligible assays: {len(candidate_table)}""")
print(f"""Excluded assays: {len(excluded_table)}""")
print(f"""Selected assays: {len(selected_table)}""")
