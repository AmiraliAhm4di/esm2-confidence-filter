# ProteinGym Variant Processing

from pathlib import Path
import math
import pandas as pd

from scripts.parser import mutation_parser
from scripts.parser import mutation_validator
from scripts.parser import mutated_seq_validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSAYS_DIR = (PROJECT_ROOT / "data" / "raw" / "DMS_ProteinGym_substitutions")
FINAL_SELECTION = (PROJECT_ROOT / "outputs" / "final_selection.csv")
PARSED_OUTPUT = (PROJECT_ROOT / "outputs" / "parsed_variants.csv")
EXCLUDED_OUTPUT = (PROJECT_ROOT / "outputs" / "excluded_variants.csv")

selected_table = pd.read_csv(FINAL_SELECTION)

eligible_rows = []
excluded_rows = []
assays = 0
total_rows = 0


for index, selected_row in selected_table.iterrows():
    dms_id = selected_row["DMS_id"]
    target_seq = selected_row["target_seq"]
    dms_filename = selected_row["DMS_filename"]
    uniprot_id = selected_row["UniProt_ID"]

    assay_path = ASSAYS_DIR / dms_filename
    assay_table = pd.read_csv(assay_path)

    assays += 1
    total_rows += len(assay_table)


    for index, var_row in assay_table.iterrows():
        mutation = var_row["mutant"]

        try:
            wild_type, position, mutant = mutation_parser(mutation)

            mutation_validator(wild_type, position, mutant, target_seq)

            mutated_seq_validator(target_seq, var_row["mutated_sequence"], position, mutant)

            try:
                dms_score = float(var_row["DMS_score"])
            except (TypeError, ValueError):
                raise ValueError("invalid_dms_score")

            if not math.isfinite(dms_score):
                raise ValueError("invalid_dms_score")

            eligible_rows.append(
                {
                    "DMS_id": dms_id,
                    "DMS_filename": dms_filename,
                    "UniProt_ID": uniprot_id,
                    "source_row": index,
                    "mutation": mutation,
                    "wild_type": wild_type,
                    "position": position,
                    "mutant": mutant,
                    "DMS_score": dms_score,
                }
            )

        except ValueError as error:
            excluded_rows.append(
                {
                    "DMS_id": dms_id,
                    "DMS_filename": dms_filename,
                    "mutation": mutation,
                    "source_row": index,
                    "exclusion_reason": str(error),
                }
            )

parsed_table = pd.DataFrame(eligible_rows)
excluded_table = pd.DataFrame(excluded_rows)

parsed_table.to_csv(PARSED_OUTPUT, index=False)
excluded_table.to_csv(EXCLUDED_OUTPUT, index=False)

print("Selected assays:", len(selected_table))
print("Loaded assay files:", assays)
print("Total variant rows:", total_rows)
print("Excluded rows:", len(excluded_table))
print("Eligible rows:", len(parsed_table))

print("\nSaved parsed variants to:", PARSED_OUTPUT)
print("Saved excluded variants to:", EXCLUDED_OUTPUT)