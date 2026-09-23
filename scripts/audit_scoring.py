# Audit the 20 scored assays and combine them

from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import hashlib
import json
import math
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr



PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
FINAL_SELECTION = OUTPUTS / "final_selection.csv"
PARSED_VARIANTS = OUTPUTS / "parsed_variants.csv"
EXCLUDED_VARIANTS = OUTPUTS / "excluded_variants.csv"
SCORING_MANIFEST = OUTPUTS / "scoring_manifest.csv"
BENCHMARKS = PROJECT_ROOT / "data" / "raw" / "DMS_substitutions_Spearman_DMS_level.csv"
SCORED_DIR = OUTPUTS / "scored"
CACHE_DIR = OUTPUTS / "position_cache"


COMBINED_OUTPUT = OUTPUTS / "scored_variants_all.csv"
REPRODUCTION_OUTPUT = OUTPUTS / "reproduction_all.csv"
INTEGRITY_OUTPUT = OUTPUTS / "scoring_integrity.json"
FREEZE_OUTPUT = OUTPUTS / "dataset_freeze.json"

MODEL = "facebook/esm2_t33_650M_UR50D"
amino_acids = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
tolerance = 0.02



def file_hash(path):
    contents = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(contents).hexdigest()


def verify_freeze():
    with open(FREEZE_OUTPUT, encoding="utf-8") as file:
        freeze = json.load(file)

    for filename, expected_hash in freeze["sha256"].items():
        path = PROJECT_ROOT / filename
        if not path.exists() or file_hash(path) != expected_hash:
            raise ValueError("Freeze mismatch: " + filename)

    print("FROZEN DATASET VERIFIED:", len(freeze["sha256"]), "files unchanged")


if "--verify-freeze" in sys.argv:
    verify_freeze()


else:
    if FREEZE_OUTPUT.exists() and "--dry-run" not in sys.argv:
        raise ValueError("Already frozen. Use --verify-freeze.")

    selected_table = pd.read_csv(FINAL_SELECTION)
    parsed_table = pd.read_csv(PARSED_VARIANTS)
    excluded_table = pd.read_csv(EXCLUDED_VARIANTS)
    manifest_table = pd.read_csv(SCORING_MANIFEST)
    benchmark_table = pd.read_csv(BENCHMARKS)

    selected_ids = selected_table["DMS_id"].tolist()
    if len(selected_ids) != 20 or len(set(selected_ids)) != 20:
        raise ValueError("Need 20 assays")
    if set(manifest_table["DMS_id"]) != set(selected_ids) or not (manifest_table["status"] == "success").all():
        raise ValueError("Some assays are not scored")


    combined_tables = []
    reproduction_rows = []
    frozen_files = [FINAL_SELECTION, PARSED_VARIANTS, EXCLUDED_VARIANTS, SCORING_MANIFEST, BENCHMARKS, PROJECT_ROOT / "METHODS.md"]
    total_duplicates = 0
    total_excluded_multi = 0

    for index, selected_row in selected_table.iterrows():
        dms_id = selected_row["DMS_id"]
        target_seq = selected_row["target_seq"]
        scored_path = SCORED_DIR / (dms_id + ".csv")
        duplicates_path = SCORED_DIR / (dms_id + "_duplicates.csv")
        cache_path = CACHE_DIR / (dms_id + ".json")

        for path in [scored_path, duplicates_path, cache_path]:
            if not path.exists():
                raise FileNotFoundError(path)
        scored_table = pd.read_csv(scored_path)
        duplicates_table = pd.read_csv(duplicates_path)
        parsed_assay = parsed_table[parsed_table["DMS_id"] == dms_id].copy()
        excluded_assay = excluded_table[excluded_table["DMS_id"] == dms_id]
        manifest_row = manifest_table[manifest_table["DMS_id"] == dms_id].iloc[0]
        benchmark_row = benchmark_table[benchmark_table["DMS ID"] == dms_id].iloc[0]

        comparison_columns = list(parsed_assay.columns)
        comparison_columns.remove("source_row")
        expected_scored = parsed_assay.drop_duplicates(subset=comparison_columns).reset_index(drop=True)
        expected_duplicates = parsed_assay[parsed_assay.duplicated(subset=comparison_columns)]

        if len(parsed_assay) != manifest_row["eligible_rows"] or len(scored_table) != manifest_row["scored_rows"]:
            raise ValueError(dms_id + ": manifest count mismatch")
        if len(scored_table) != len(expected_scored) or len(duplicates_table) != len(expected_duplicates):
            raise ValueError(dms_id + ": scored or duplicate count mismatch")
        if scored_table["source_row"].tolist() != expected_scored["source_row"].tolist():
            raise ValueError(dms_id + ": source rows differ")
        if duplicates_table["source_row"].tolist() != expected_duplicates["source_row"].tolist():
            raise ValueError(dms_id + ": duplicates differ")
        if scored_table["mutation"].duplicated().any():
            raise ValueError(dms_id + ": repeated mutation")

        # Check that input columns did not change
        for column in comparison_columns:
            if not scored_table[column].equals(expected_scored[column]):
                raise ValueError(dms_id + ": changed column: " + column)

        if not np.isfinite(scored_table[["DMS_score", "esm2_score", "entropy"]].to_numpy()).all():
            raise ValueError(dms_id + ": bad score")

        with open(cache_path, encoding="utf-8") as file:
            cache = json.load(file)
        if cache["identity"]["model"] != MODEL or cache["identity"]["sequence"] != target_seq:
            raise ValueError(dms_id + ": cache model or sequence mismatch")
        if cache["identity"]["amino_acids"] != amino_acids:
            raise ValueError(dms_id + ": amino acid order mismatch")

        positions = cache["positions"]
        if len(positions) != len(target_seq):
            raise ValueError(dms_id + ": incomplete cache")

        position_entropy = {}
        for position in range(1, len(target_seq) + 1):
            if str(position) not in positions:
                raise ValueError(dms_id + ": missing position " + str(position))
            log_probs = positions[str(position)]
            if len(log_probs) != 20:
                raise ValueError(dms_id + ": wrong cache size")
            entropy = 0
            for log_prob in log_probs:
                probability = math.exp(log_prob)
                entropy -= probability * log_prob
            position_entropy[position] = entropy

        for row_index, variant_row in scored_table.iterrows():
            position = int(variant_row["position"])
            wild_type = variant_row["wild_type"]
            mutant = variant_row["mutant"]
            mutation = variant_row["mutation"]

            if position < 1 or position > len(target_seq):
                raise ValueError(dms_id + ": position out of range")
            if target_seq[position - 1] != wild_type or mutation != wild_type + str(position) + mutant:
                raise ValueError(dms_id + ": mutation mismatch")

            log_probs = positions[str(position)]
            expected_score = log_probs[amino_acids.index(mutant)] - log_probs[amino_acids.index(wild_type)]
            if abs(variant_row["esm2_score"] - expected_score) > 1e-7:
                raise ValueError(dms_id + ": score/cache mismatch")
            if abs(variant_row["entropy"] - position_entropy[position]) > 1e-6:
                raise ValueError(dms_id + ": entropy/cache mismatch")


        local_spearman = float(spearmanr(scored_table["esm2_score"], scored_table["DMS_score"]).statistic)
        published_spearman = float(benchmark_row["ESM2 (650M)"])
        published_variants = int(benchmark_row["Number of Mutants"])
        absolute_difference = abs(local_spearman - published_spearman)
        excluded_multi = int((excluded_assay["exclusion_reason"] == "multi_mutant").sum())


        if len(scored_table) == published_variants:
            if absolute_difference > tolerance:
                raise ValueError(dms_id + ": benchmark difference > 0.02")
            result = "PASS"
            reason = "Count matches; Spearman within 0.02."
        else:
            if len(excluded_assay) != excluded_multi or len(scored_table) + excluded_multi != published_variants:
                raise ValueError(dms_id + ": benchmark count mismatch")
            result = "NOT_COMPARABLE_MULTI_MUTANT"
            reason = "Multi-mutants excluded here."

        reproduction_rows.append({
            "DMS_id": dms_id,
            "model": "ESM2 (650M)",
            "local_variants": len(scored_table),
            "published_variants": published_variants,
            "excluded_multi_mutants": excluded_multi,
            "local_spearman": local_spearman,
            "published_spearman": published_spearman,
            "absolute_difference": absolute_difference,
            "tolerance": tolerance,
            "result": result,
            "reason": reason,
        })

        combined_tables.append(scored_table)
        frozen_files.extend([scored_path, duplicates_path, cache_path])
        total_duplicates += len(duplicates_table)
        total_excluded_multi += excluded_multi


    combined_table = pd.concat(combined_tables, ignore_index=True)
    reproduction_table = pd.DataFrame(reproduction_rows)


    summary = {
        "assays": len(selected_ids),
        "scored_rows": len(combined_table),
        "exact_duplicates_excluded": total_duplicates,
        "multi_mutants_excluded": total_excluded_multi,
        "benchmark_pass": int((reproduction_table["result"] == "PASS").sum()),
        "benchmark_not_comparable_multi_mutant": int((reproduction_table["result"] == "NOT_COMPARABLE_MULTI_MUTANT").sum()),
        "benchmark_fail": int((reproduction_table["result"] == "FAIL").sum()),
        "max_absolute_difference_comparable": float(reproduction_table[reproduction_table["result"] == "PASS"]["absolute_difference"].max()),
        "integrity_status": "PASS",
    }

    print(json.dumps(summary, indent=2))
    print(reproduction_table[["DMS_id", "local_variants", "published_variants", "absolute_difference", "result"]].to_string(index=False))

    if "--dry-run" in sys.argv:
        print("DRY RUN: no files written")
    else:
        combined_table.to_csv(COMBINED_OUTPUT, index=False)
        reproduction_table.to_csv(REPRODUCTION_OUTPUT, index=False)
        with open(INTEGRITY_OUTPUT, "w", encoding="utf-8", newline="\n") as file:
            json.dump(summary, file, indent=2)
            file.write("\n")

        frozen_files.extend([COMBINED_OUTPUT, REPRODUCTION_OUTPUT, INTEGRITY_OUTPUT])
        frozen_hashes = {}
        for path in frozen_files:
            filename = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            frozen_hashes[filename] = file_hash(path)

        freeze = {
            "assays": len(reproduction_table),
            "rows": len(combined_table),
            "sha256": frozen_hashes,
        }
        with open(FREEZE_OUTPUT, "w", encoding="utf-8", newline="\n") as file:
            json.dump(freeze, file, indent=2)
            file.write("\n")

        print("WROTE", COMBINED_OUTPUT)
        print("WROTE", REPRODUCTION_OUTPUT)
        print("WROTE", FREEZE_OUTPUT)
        verify_freeze()
