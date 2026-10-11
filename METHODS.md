# Methods Specification

### 1. Assays

The candidate pool came from ProteinGym v1.3. It included assays with an available data file, a valid protein sequence and a published ESM-2 650M benchmark. Each had at least 100 unique and usable single-amino-acid substitutions.

To stop proteins with multiple assays from being represented more than once, I grouped the eligible assays by protein ID and randomly selected twenty proteins without replacement. Where a protein had several eligible assays, one was selected at random.

### 2. Metadata

The assay filenames linked the data to the protein sequences and assay IDs in `DMS_substitutions.csv`. I used entries with a single matching metadata row, leaving out missing files and ambiguous filename matches. The assay IDs also linked the data to the published benchmarks.

### 3. Mutation positions

Each mutation string gives the wild-type amino acid, its position, and the mutant amino acid. Positions are one-indexed, so the first amino acid is position 1. In Python, I subtract 1 from the position to get the zero-based index. I checked the position and wild-type amino acid against the reference protein sequence.

### 4. Included variants

I only included single-amino-acid substitutions between the 20 standard amino acids that had valid experimental scores.For each included variant, I checked the supplied mutant sequence against the original protein sequence to confirm that it contained only the stated substitution. Duplicate entries were kept once.

### 5. Score direction

ProteinGym's normalized experimental scores were used without changes. Higher scores meant that the mutant performed better for the trait measured in the assay. The ESM-2 score was calculated as `log p(mutant) - log p(wild type)` with higher scores meaning that the model considered the mutation more plausible.

### 6. Probabilities and entropy

ESM-2 was given the wild-type sequence with one amino acid masked at a time. From its predictions at the masked position, I took the scores for the 20 standard amino acids and converted them into probabilities using softmax.

Those probabilities were used for both the variant score and positional entropy. Entropy was calculated as `H = -sum(p * log(p))`, using natural logs. This gave one entropy value per position (measured in nats) shared by the substitutions at that position.

### 7. Random seed

Random choices used seed 42. It was just a random number I chose and doesn't have any significance so I could repeat the same selections.

### 8. Published benchmarks

The published correlations helped me check on whether the scoring was actually working. For assays with matching variant #, I compared Spearman correlation across all scored variants with ProteinGym's ESM-2 650M result with a tolerance of 0.02.

Five published benchmarks were excluded. Those assays stayed in the filtering analysis but their published correlations were not directly comparable with the single-substitution results.

### 9. Entropy filtering

Variants were ranked by entropy within each assay. Starting with the lowest values, I kept progressively larger subsets from 10% to 100% coverage in steps of 10%.

Spearman correlation measured agreement between model and experimental rankings in each subset. The main comparison used 50% coverage and measured the change from the correlation across all variants in the same dataset.

### 10. Random baseline

Random subsets provided a comparison for what happened when variants were removed without using entropy. Each entropy-filtered subset was compared with 1,000 random subsets of the same size, sampled without replacement. The random comparison was repeated on the matched data.

### 11. Experimental effects

Entropy could also be related to how strongly a position responded to mutation. That would make it harder to tell whether a change in correlation reflected better predictions or a different set of experimental effects.

For each position, I took the average absolute experimental score across its substitutions. Its relationship with entropy was measured using Spearman correlation.

### 12. Position matching

For the matched comparison, I tried to keep the experimental effects similar across entropy groups. Positions were split into four groups by entropy rank. Their average absolute experimental scores were used to form another four groups. Within each experimental-score group, I randomly sampled the same number of positions from each entropy group using the smallest group's count. The matched dataset contained all substitutions from those selected positions.

Equal position counts did not necessarily mean that the experimental scores were balanced. After selection, I compared the low and high entropy halves using their position-level average absolute effects and the variance of their individual variant scores. For the balance checks, I used an absolute standardized mean difference of 0.1 or less and a high to low variance ratio between 0.8 and 1.25.

The matched analysis also used a minimum of 10 positions in each half and 100 variants per assay. Six assays remained after these checks. Entropy filtering and the random baseline were then calculated on their matched variants.

### 13. Results and intervals

Since matching only left me with six assays, I reported their results before matching as well as after, alongside the result across all twenty. The summary used the mean change in Spearman at 50% coverage and weighted each assay equally. For matched data, the change was measured against all variants in the matched dataset.

The bootstrap intervals came from resampling assays with replacement 10,000 times and taking the middle 95% of the resulting means. The filter's advantage over random selection was calculated by subtracting the average random correlation from the filtered correlation at the same coverage. To compare that advantage before and after matching, I calculated the change for each of the same six assays and bootstrapped those paired changes.

The figure shows 1 minus the mean Spearman correlation so that lower values indicate better performance. The shaded bands covered the middle 95% of the average random retention results at each coverage level.

The area under the risk-coverage curve (AURC) summarized the same relationship over coverage. It was calculated using the trapezoidal rule with 1 minus Spearman and the actual coverage values from 10% to 100%.
