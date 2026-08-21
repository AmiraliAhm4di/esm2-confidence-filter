# Mutation Parser Tests

import pytest
from scripts.parser import mutation_parser
from scripts.parser import mutation_validator
from scripts.parser import mutated_seq_validator

def test_parser():
    result = mutation_parser("A42G")
    assert result == ("A", 42, "G")


def test_first_and_last_positions():
    first_position = mutation_validator("M", 1, "V", "MAG")
    last_position = mutation_validator("G", 3, "A", "MAG")

    assert first_position is True
    assert last_position is True




def test_multi_mutant():
    with pytest.raises(ValueError):
        mutation_parser("A2G:G3A")


def test_wrong_wild_type():
    with pytest.raises(ValueError):
        mutation_validator("G", 2, "A", "MAG")



def test_mutated_sequence():
    valid_sequence = mutated_seq_validator("MAG", "MVG", 2, "V")

    assert valid_sequence is True

    with pytest.raises(ValueError):
     mutated_seq_validator("MAG", "MVA", 2, "V")