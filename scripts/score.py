# ESM-2 Assay Scoring

from pathlib import Path
import json
import math
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_SELECTION = PROJECT_ROOT / "outputs" / "final_selection.csv"
PARSED_OUTPUT = PROJECT_ROOT / "outputs" / "parsed_variants.csv"
CACHE_DIR = PROJECT_ROOT / "outputs" / "position_cache"
SCORED_DIR = PROJECT_ROOT / "outputs" / "scored"

MODEL = "facebook/esm2_t33_650M_UR50D"
DMS_ID = "GLPA_HUMAN_Elazar_2016"
BATCH_SIZE = 2

amino_acids = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

CACHE_DIR.mkdir(parents=True, exist_ok=True)
SCORED_DIR.mkdir(parents=True, exist_ok=True)
CACHE_OUTPUT = CACHE_DIR / f"{DMS_ID}.json"
SCORED_OUTPUT = SCORED_DIR / f"{DMS_ID}.csv"

selected_table = pd.read_csv(FINAL_SELECTION)
selected_row = selected_table[selected_table["DMS_id"] == DMS_ID].iloc[0]
sequence = selected_row["target_seq"]

parsed_table = pd.read_csv(PARSED_OUTPUT)
assay_table = parsed_table[parsed_table["DMS_id"] == DMS_ID].copy()
if len(assay_table) == 0:
    raise ValueError("No eligible variants found for this assa")


# Ignore source row
comparison_columns = []
for column in assay_table.columns:
    if column != "source_row":
        comparison_columns.append(column)


duplicates = assay_table[assay_table.duplicated(subset=comparison_columns)].copy()
duplicates["exclusion_reason"] = "exact_duplicate"
duplicates.to_csv(SCORED_DIR / f"{DMS_ID}_duplicates.csv", index=False)
assay_table = assay_table.drop_duplicates(subset=comparison_columns)

if assay_table["mutation"].duplicated().any():
    raise ValueError("Repeated mutation with conflicting records")


identity = {
    "model": MODEL,
    "sequence": sequence,
    "amino_acids": amino_acids,
    "normalization": "20 amino acids; natural logarithm; fp16 model",
}
cache = {"identity": identity, "positions": {}}


if CACHE_OUTPUT.exists():
    with open(CACHE_OUTPUT, encoding="utf-8") as file:
        cache = json.load(file)
    if cache["identity"] != identity:
        raise ValueError("Cache settings do not match this run")



positions = cache["positions"]
remaining = []
for position in range(1, len(sequence) + 1):
    if str(position) not in positions:
        remaining.append(position)


print(f"Assay: {DMS_ID}, Protein length: {len(sequence)}")
print(f"Cached positions: {len(positions)}/{len(sequence)}")


tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForMaskedLM.from_pretrained(MODEL)
model = model.to("cuda")
model = model.half()
model.eval()

aa_ids = []
for aa in amino_acids:
    aa_ids.append(tokenizer.convert_tokens_to_ids(aa))


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

    # Read each masked sequences result
    for batch_index in range(len(batch_positions)):
        position = batch_positions[batch_index]
        mask_positions = inputs["input_ids"][batch_index] == tokenizer.mask_token_id
        mask_indices = mask_positions.nonzero()
        mask_index = mask_indices[0, 0].item()
        mask_logits = outputs.logits[batch_index, mask_index]

        aa_logits = mask_logits[aa_ids].float()
        aa_log_probs = torch.log_softmax(aa_logits, dim=0)
        positions[str(position)] = aa_log_probs.tolist()

    with open(CACHE_OUTPUT, "w", encoding="utf-8") as file:
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

for index, var_row in assay_table.iterrows():
    wild_type = var_row["wild_type"]
    position = int(var_row["position"])
    mutant = var_row["mutant"]

    aa_log_probs = positions[str(position)]
    wild_type_index = amino_acids.index(wild_type)
    mutant_index = amino_acids.index(mutant)
    variant_score = aa_log_probs[mutant_index] - aa_log_probs[wild_type_index]

    variant_scores.append(variant_score)
    entropies.append(position_entropy[position])

assay_table["esm2_score"] = variant_scores
assay_table["entropy"] = entropies


assay_table.to_csv(SCORED_OUTPUT, index=False)
print(f"Scored variants: {len(assay_table)}, Duplicate exclusions: {len(duplicates)}")
print(f"Saved: {SCORED_OUTPUT}")
