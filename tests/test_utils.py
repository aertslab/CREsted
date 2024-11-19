import numpy as np

from crested.utils import (
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
