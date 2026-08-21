amino_acids = set("ACDEFGHIKLMNPQRSTVWY")

def mutation_parser(mutation):
    if not isinstance(mutation, str) or len(mutation) < 3:
        raise ValueError("invalid_mutant")
    
    if ":" in mutation:
        raise ValueError("multi_mutant")

    wild_type = mutation[0]
    position = mutation[1:-1]
    mutant = mutation[-1]

    if not wild_type.isalpha() or not mutant.isalpha() or not position.isdigit():
        raise ValueError("invalid_mutant")

    position = int(position)

    return wild_type, position, mutant

def mutation_validator(wild_type, position, mutant, target_seq):
    if wild_type not in amino_acids or mutant not in amino_acids:
        raise ValueError("unnatural_aminoacid")

    if not 1 <= position <= len(target_seq):
        raise ValueError("position_out_of_bounds")

    sequence = target_seq[position - 1]
    if not sequence == wild_type:
        raise ValueError("wildtype_mismatch")

    if wild_type == mutant:
        raise ValueError("same_aminoacid")

    return True

def mutated_seq_validator(target_seq, mutated_seq, position, mutant):
    if not isinstance(mutated_seq, str) or not mutated_seq:
        raise ValueError("missing_mutated_sequence")

    if len(mutated_seq) != len(target_seq):
        raise ValueError("mutated_length_mismatch")

    if mutated_seq[position - 1] != mutant:
        raise ValueError("mutant_position_mismatch")

    difference_positions = []

    for index, (original, changed) in enumerate(zip(target_seq, mutated_seq)):
        if original != changed:
            difference_positions.append(index)

    if difference_positions != [position - 1]:
        raise ValueError("unexpected_changes")

    return True
