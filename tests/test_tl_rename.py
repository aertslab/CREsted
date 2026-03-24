"""Test aliases of renamed functions in the tl module."""

import crested


def test_enhancer_design_in_silico_evolution(keras_model, adata, genome):
    # one model
    seqs = crested.tl.enhancer_design_in_silico_evolution(n_mutations=2, target=0, model=keras_model, n_sequences=1)
    assert len(seqs) == 1, len(seqs)
    assert len(seqs[0]) == keras_model.input_shape[1], len(seqs[0])
