import numpy as np
import pytest

from crested.utils import (
    calculate_nucleotide_distribution,
    hot_encoding_to_sequence,
    one_hot_encode_sequence,
    reverse_complement,
)


def test_reverse_complement_string():
    assert reverse_complement("ACGT") == "ACGT"
    assert reverse_complement("TGCA") == "TGCA"
    assert reverse_complement("AAGGTTCC") == "GGAACCTT"


def test_reverse_complement_list_of_strings():
    assert reverse_complement(["ACGT", "TGCA"]) == ["ACGT", "TGCA"]
    assert reverse_complement(["AAGGTTCC", "CCGGAATT"]) == ["GGAACCTT", "AATTCCGG"]


def test_reverse_complement_one_hot_encoded_array():
    seq_1 = "ACGTA"
    seq_1_one_hot = one_hot_encode_sequence(seq_1)
    seq_1_rev = reverse_complement(seq_1)
    seq_1_rev_one_hot = one_hot_encode_sequence(seq_1_rev)

    np.testing.assert_array_equal(reverse_complement(seq_1_one_hot), seq_1_rev_one_hot)

    seq_2 = "AAGGTTCC"
    seq_2_rev = "GGAACCTT"
    seq_2_one_hot = one_hot_encode_sequence(seq_2, expand_dim=False)
    seq_2_rev_one_hot = reverse_complement(seq_2_one_hot)
    seq_2_seq_reversed = hot_encoding_to_sequence(seq_2_rev_one_hot)

    assert seq_2_seq_reversed == seq_2_rev


def test_nucleotide_distribution_total():
    seq = "ACTT" * 100
    distribution = calculate_nucleotide_distribution(seq, per_position=False)
    assert distribution[0] == 0.25, distribution[0]
    assert distribution[1] == 0.25, distribution[1]
    assert distribution[2] == 0.00, distribution[2]
    assert distribution[3] == 0.50, distribution[3]
    assert distribution.shape == (4,), distribution.shape


def test_nucleotide_distribution_pos():
    seq = ["ACTTG", "AATTG"]
    distribution = calculate_nucleotide_distribution(seq, per_position=True)
    assert distribution.shape == (5, 4), distribution.shape
    assert distribution[0][0] == 1.0, distribution[0][0]
    assert distribution[0][1] == 0.0, distribution[0][1]
    assert distribution[1][0] == 0.5, distribution[1][0]
    assert distribution[1][1] == 0.5, distribution[1][1]
    assert distribution[1][2] == 0.0, distribution[1][2]
    assert distribution[-1][0] == 0.0, distribution[-1][0]
    assert distribution[-1][2] == 1.0, distribution[-1][2]


def test_nucleotide_distribution_unequal_inputs():
    """Test when inputs have different lengths."""
    seq = ["ACTTG", "AATTG", "A"]
    with pytest.raises(ValueError):
        _ = calculate_nucleotide_distribution(seq, per_position=True)
