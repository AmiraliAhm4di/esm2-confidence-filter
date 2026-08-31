# ESM-2 Smoke Test

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


MODEL = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForMaskedLM.from_pretrained(MODEL)


model = model.to("cuda")
model = model.half()
model.eval()

amino_acids = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
aa_ids = [tokenizer.convert_tokens_to_ids(aa) for aa in amino_acids]

sequence = "MFKLLSKLLVYLTASIMAIASPLAFSVDSSGEYPTVSEIPVGEVRLYQIADGVWSHIATQSFDGAVYPSNGLIVRDGDELLLIDTAWGAKNTAALLAEIEKQIGLPVTRAVSTHFHDDRVGGVDVLRAAGVATYASPSTRRLAEVEGNEIPTHSLEGLSSSGDAVRFGPVELFYPGAAHSTDNLIVYVPSASVLYGGCAIYELSRTSAGNVADADLAEWPTSIERIQQHYPEAQFVIPGHGLPGGLDLLKHTTNVVKAHTNRSVVE"

wild_type = "M"
position = 1
mutant = "V"

sequence_index = position - 1


# Mask one amino
masked_sequence = (
    sequence[:sequence_index]
    + tokenizer.mask_token
    + sequence[sequence_index + 1:]
)

inputs = tokenizer(masked_sequence, return_tensors="pt")

mask_positions = inputs["input_ids"] == tokenizer.mask_token_id
mask_index = mask_positions.nonzero()[0, 1].item()



input_ids = inputs["input_ids"].to("cuda")
attention_mask = inputs["attention_mask"].to("cuda")

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

# Get prob for the 20 aa
mask_logits = outputs.logits[0, mask_index]

aa_logits = mask_logits[aa_ids].float()
aa_log_probs = torch.log_softmax(aa_logits, dim=0)
aa_probs = aa_log_probs.exp()

entropy = -(aa_probs * aa_log_probs).sum()

wild_type_index = amino_acids.index(wild_type)
mutant_index = amino_acids.index(mutant)

wild_type_probability = aa_probs[wild_type_index]
variant_score = aa_log_probs[mutant_index] - aa_log_probs[wild_type_index]


amino_acid_distribution = dict(zip(amino_acids, aa_probs.tolist()))


print(f"20-AA probabilities: {amino_acid_distribution}")
print(f"""
    Entropy: {entropy.item()}
    | Wild-type probability: {wild_type_probability.item()}
    | Variant score: {variant_score.item()}
""")