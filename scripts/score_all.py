# ESM-2 Scoring for All Selected Assays

from pathlib import Path
import json
import math
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import time
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_SELECTION = PROJECT_ROOT / "outputs" / "final_selection.csv"
PARSED_OUTPUT = PROJECT_ROOT / "outputs" / "parsed_variants.csv"
CACHE_DIR = PROJECT_ROOT / "outputs" / "position_cache"
SCORED_DIR = PROJECT_ROOT / "outputs" / "scored"
MANIFEST_OUTPUT = PROJECT_ROOT / "outputs" / "scoring_manifest.csv"

MODEL = "facebook/esm2_t33_650M_UR50D"
BATCH_SIZE = 2

amino_acids = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

manifest_columns = [
    "DMS_id",
    "status",
    "started_at_utc",
    "finished_at_utc",
    "runtime_seconds",
    "batch_size",
    "gpu",
    "eligible_rows",
    "duplicate_exclusions",
    "scored_rows",
    "error",
]

def save_manifest(manifest_rows, selected_ids):
    ordered_rows = []

    for dms_id in selected_ids:
        if dms_id in manifest_rows:
            ordered_rows.append(manifest_rows[dms_id])

    manifest_table = pd.DataFrame(ordered_rows, columns=manifest_columns)
    manifest_table.to_csv(MANIFEST_OUTPUT, index=False)


CACHE_DIR.mkdir(parents=True, exist_ok=True)
SCORED_DIR.mkdir(parents=True, exist_ok=True)

selected_table = pd.read_csv(FINAL_SELECTION)
parsed_table = pd.read_csv(PARSED_OUTPUT)
selected_ids = selected_table["DMS_id"].tolist()

manifest_rows = {}
if MANIFEST_OUTPUT.exists():
    old_manifest = pd.read_csv(MANIFEST_OUTPUT)
    for index, old_row in old_manifest.iterrows():
        manifest_rows[old_row["DMS_id"]] = old_row.to_dict()



if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU")

gpu = torch.cuda.get_device_name(0)


# Find the assays that still need scoring
pending_ids = []

for dms_id in selected_ids:
    scored_output = SCORED_DIR / f"{dms_id}.csv"

    if scored_output.exists():
        if dms_id not in manifest_rows or manifest_rows[dms_id]["status"] != "success":
            existing_table = pd.read_csv(scored_output)
            duplicates_output = SCORED_DIR / f"{dms_id}_duplicates.csv"
            duplicate_count = 0
            if duplicates_output.exists():
                duplicate_table = pd.read_csv(duplicates_output)
                duplicate_count = len(duplicate_table)

            manifest_rows[dms_id] = {
                "DMS_id": dms_id,
                "status": "success",
                "started_at_utc": "",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "runtime_seconds": 0,
                "batch_size": BATCH_SIZE,
                "gpu": gpu,
                "eligible_rows": len(existing_table) + duplicate_count,
                "duplicate_exclusions": duplicate_count,
                "scored_rows": len(existing_table),
                "error": "existing scored output; not rerun",
            }
    else:
        pending_ids.append(dms_id)

save_manifest(manifest_rows, selected_ids)

print(f"Selected assays: {len(selected_ids)}")
print(f"Already scored: {len(selected_ids) - len(pending_ids)}")
print(f"Still pending: {len(pending_ids)}")


if len(pending_ids) > 0:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForMaskedLM.from_pretrained(MODEL)
    model = model.to("cuda")
    model = model.half()
    model.eval()

    aa_ids = []
    for amino_acid in amino_acids:
        aa_ids.append(tokenizer.convert_tokens_to_ids(amino_acid))


for dms_id in pending_ids:
    print()
    print(f"Starting: {dms_id}")

    start_time = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    eligible_rows = 0
    duplicate_count = 0
    scored_rows = 0
    status = "failed"
    error_text = ""

    try:
        selected_row = selected_table[selected_table["DMS_id"] == dms_id].iloc[0]
        sequence = selected_row["target_seq"]

        assay_table = parsed_table[parsed_table["DMS_id"] == dms_id].copy()
        eligible_rows = len(assay_table)

        if eligible_rows == 0:
            raise ValueError("No eligible variants found")

        comparison_columns = []
        for column in assay_table.columns:
            if column != "source_row":
                comparison_columns.append(column)

        duplicates = assay_table[assay_table.duplicated(subset=comparison_columns)].copy()
        duplicates["exclusion_reason"] = "exact_duplicate"
        duplicate_count = len(duplicates)

        duplicates_output = SCORED_DIR / f"{dms_id}_duplicates.csv"
        duplicates.to_csv(duplicates_output, index=False)

        assay_table = assay_table.drop_duplicates(subset=comparison_columns)

        if assay_table["mutation"].duplicated().any():
            raise ValueError("Repeated mutation")

        cache_output = CACHE_DIR / f"{dms_id}.json"
        scored_output = SCORED_DIR / f"{dms_id}.csv"

        identity = {
            "model": MODEL,
            "sequence": sequence,
            "amino_acids": amino_acids,
            "normalization": "20 amino acids; natural logarithm; fp16 model",
        }
        cache = {"identity": identity, "positions": {}}

        if cache_output.exists():
            with open(cache_output, encoding="utf-8") as file:
                cache = json.load(file)

            if cache["identity"] != identity:
                raise ValueError("Cache settings not match")

        positions = cache["positions"]
        remaining = []

        for position in range(1, len(sequence) + 1):
            if str(position) not in positions:
                remaining.append(position)

        print(f"Protein length: {len(sequence)}")
        print(f"Cached positions: {len(positions)}/{len(sequence)}")

        for start in range(0, len(remaining), BATCH_SIZE):
            batch_positions = remaining[start:start + BATCH_SIZE]
            masked_sequences = []

            for position in batch_positions:
                sequence_index = position - 1
                masked_sequence = (
                    sequence[:sequence_index]
                    + tokenizer.mask_token
                    + sequence[sequence_index + 1:]
                )
                masked_sequences.append(masked_sequence)

            inputs = tokenizer(masked_sequences, return_tensors="pt")
            input_ids = inputs["input_ids"].to("cuda")
            attention_mask = inputs["attention_mask"].to("cuda")

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            for batch_index in range(len(batch_positions)):
                position = batch_positions[batch_index]
                mask_positions = input_ids[batch_index] == tokenizer.mask_token_id
                mask_indices = mask_positions.nonzero()
                mask_index = mask_indices[0, 0].item()
                mask_logits = outputs.logits[batch_index, mask_index]

                aa_logits = mask_logits[aa_ids].float()
                aa_log_probs = torch.log_softmax(aa_logits, dim=0)
                positions[str(position)] = aa_log_probs.tolist()

            # Save after each batch so a stopped run can continue later
            with open(cache_output, "w", encoding="utf-8") as file:
                json.dump(cache, file)

            print(f"Saved positions: {len(positions)}/{len(sequence)}")

        position_entropy = {}

        for position in range(1, len(sequence) + 1):
            aa_log_probs = positions[str(position)]
            entropy = 0

            for log_probability in aa_log_probs:
                probability = math.exp(log_probability)
                entropy -= probability * log_probability

            position_entropy[position] = entropy

        variant_scores = []
        entropies = []

        for index, variant_row in assay_table.iterrows():
            wild_type = variant_row["wild_type"]
            position = int(variant_row["position"])
            mutant = variant_row["mutant"]

            aa_log_probs = positions[str(position)]
            wild_type_index = amino_acids.index(wild_type)
            mutant_index = amino_acids.index(mutant)
            variant_score = aa_log_probs[mutant_index] - aa_log_probs[wild_type_index]

            variant_scores.append(variant_score)
            entropies.append(position_entropy[position])

        assay_table["esm2_score"] = variant_scores
        assay_table["entropy"] = entropies

        assay_table.to_csv(scored_output, index=False)

        scored_rows = len(assay_table)
        status = "success"
        print(f"Finished: {dms_id} | Scored rows: {scored_rows}")

    except Exception as error:
        error_text = f"{type(error).__name__}: {error}"
        print(f"FAILED: {dms_id} | {error_text}")
        torch.cuda.empty_cache()

    finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    runtime_seconds = round(time.perf_counter() - start_time, 2)

    manifest_rows[dms_id] = {
        "DMS_id": dms_id,
        "status": status,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "runtime_seconds": runtime_seconds,
        "batch_size": BATCH_SIZE,
        "gpu": gpu,
        "eligible_rows": eligible_rows,
        "duplicate_exclusions": duplicate_count,
        "scored_rows": scored_rows,
        "error": error_text,
    }

    save_manifest(manifest_rows, selected_ids)

success_count = 0
failed_count = 0

for dms_id in selected_ids:
    if dms_id in manifest_rows:
        if manifest_rows[dms_id]["status"] == "success":
            success_count += 1
        if manifest_rows[dms_id]["status"] == "failed":
            failed_count += 1

print()
print(f"Successful assays: {success_count}/{len(selected_ids)}")
print(f"Failed assays: {failed_count}")
print(f"Manifest: {MANIFEST_OUTPUT}")
