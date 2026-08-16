# Methods Specification

### 1. Selecting the 20 Assays

The candidate pool will include ProteinGym v1.3 assays that follow these requirements:

- have an available assay file;
- match exactly one row in the metadata;
- contain a valid protein sequence (`target_seq`);
- have a published benchmark result for the ESM-2 650M model; and
- contain at least 100 usable single-amino-acid substitutions after the exclusion rules below are applied.

To stop proteins that have multiple assays from being represented more than once, eligible assays will be grouped by protein. Twenty proteins will be selected randomly without replacement using the fixed project seed. If a selected one has many eligible assays only one will be selected using the same seed.

### 2. Matching Assays to Metadata

Each assay file will be matched to `DMS_substitutions.csv` by comparing its filename with the `DMS_filename` column. The match must return exactly one metadata row so that the correct `target_seq`, `DMS_id`, and protein information are attached to the assay.
If a file has no match or multiple, it will be recorded and excluded. The resulting `DMS_id` will later be used to find the assay's published ESM-2 benchmark result.

### 3. Reading Mutations and Positions

Mutation strings are expected to contain a wild-type amino acid, a position, and a mutant amino acid. Protein positions will be treated as one-indexed like the first amino acid being on position 1. When accessing the sequence in Python, the position will be converted to a zero-based index by subtracting 1.The parsed position must fall within `target_seq` and the stated wild-type amino acid must match the amino acid at that position.

### 4. Choosing Which Variants to Include

Only single-amino-acid substitutions between the 20 standard amino acids will be included. Each row must have a valid mutation string, agree with `target_seq` and also have a numeric `DMS_score`.
Stop mutations, insertions or deletions, nonstandard amino acids, unchanged substitutions, multi-mutants, and ect along with rows that have missing scores will be excluded. If a row has an exact duplicate it will be kept only once.
When available, `mutated_sequence` will be used as an additional check that the stated substitution produces the supplied sequence. `DMS_score_bin` and `taxon` will not be used in the main analysis. Every excluded row will be saved with a reason for its exclusion.

### 5. Checking Score Direction

ProteinGym's normalized `DMS_score` will be used as provided where a higher value means that the mutant performed better for the phenotype measured by the assay. The ESM-2 score will be calculated as `log p(mutant) - log p(wild type)`. A higher score means the model considers the mutation more plausible.
Before running the confidence analysis, the full-coverage Spearman correlation will be compared with ProteinGym's published ESM-2 650M result for the same assay. If it doesn’t match, the data processing steps will be checked.

### 6. Amino-Acid Probabilities

ESM-2 outputs scores for amino acids as well as special tokens used by the model. For this project, I will only keep the scores corresponding to the 20 standard amino acids.
I will then normalize these scores so that they form probabilities that sum to 1. These probabilities will be used to compute the variant score and the entropy.

### 7. Random Seed

The project will use a random seed of 42 for assay selection and any later steps involving randomness. This will make the same random choices reproducible when the code is run again.

### 8. Comparing with Published Results

For each assay, the locally calculated full-coverage Spearman correlation has to be within 0.02 of ProteinGym's published ESM-2 650M result. This means:
`absolute(local Spearman - published Spearman) <= 0.02`
If an assay falls outside of what we are willing to tolerate, the data matching, mutation parsing, score direction, and ESM-2 scoring steps will be checked before any confidence-filtering results are used.
