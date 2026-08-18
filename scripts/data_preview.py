import csv
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"



metadata = pd.read_csv(DATA_RAW / "DMS_substitutions.csv")
assay = pd.read_csv(DATA_RAW / "DMS_ProteinGym_substitutions/A0A1I9GEU1_NEIME_Kennouche_2019.csv")


print(metadata.shape)
print(metadata.columns.tolist())
print(metadata.isna().sum())
print(metadata.head())

print(assay.head())
print(assay.shape)
print(assay.isna().sum())
print(assay["mutant"].iloc[0])
print(assay.columns.tolist())