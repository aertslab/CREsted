Cerberus
============

.. sidebar:: Model Features

   - **Genome**: *hg38/mm10*
   - **Type**: Track prediction
   - **Parameters**: 134M/126M
   - **Size**: 534MB/502MB
   - **Input shape**: (786432, 4)
   - **Output shape**: (24576, 8361)/(24576, 3102)

The **Cerberus** model is a large model trained on bulk and pseudobulked single-cell RNA-seq, 3' RNA-seq, CAGE, DNase, ATAC, ChIP-seq and CLIP data from a wide variety of human and mouse tissues and cell types (ENCODE, FANTOM, CZI, BICCN, CATlas, recount3 and others).
It predicts 24576 bins of 32bp, covering the full 786432 bp input sequence.

It was originally provided based on the Baskerville package (PyTorch), and its weights and architecture have been ported to CREsted.
Cerberus is an ensemble of 8 models, each trained with a different held-out fold of sequences tiled across the genome.
The original model has a shared trunk and two organism-specific heads, which are provided as two specific models for human and mouse here,
resulting in models `cerberus_human_fold0`-`cerberus_human_fold7` and `cerberus_mouse_fold0`-`cerberus_mouse_fold7`. `'cerberus_human'` and `'cerberus_mouse'` are aliases for `'fold0'`.

The model is a CNN+Hydra (bidirectional Mamba-2 state space) model using the :func:`~crested.tl.zoo.cerberus` architecture, without attention or upsampling.
Predictions are in the transformed (clipped, scaled and square-rooted) units of the training data; the transforms per track are listed in the original `target tables <https://github.com/calico/baskerville/tree/main/releases/cerberus>`_.

Details of the data and the model can be found in the original publication.

-------------------

.. warning::

    The Cerberus architecture uses custom layers that are serialized inside the CREsted package. To ensure that the model is loaded correctly, make sure that CREsted and specifically `crested.tl` is imported before loading the model.
    If it still refuses to load, add `Hydra` as a custom object, as in the example.

.. note::

    The Hydra layers run on any Keras backend and on CPU, but this is slow and memory hungry for 786 kb inputs.
    For practical use, run the model on a GPU with the PyTorch backend and `mamba-ssm <https://github.com/state-spaces/mamba>`_ installed, which uses its fused Triton scan kernel.
    When training or finetuning on the PyTorch backend, compile the model with `jit_compile=True` for a ~2x speedup.

.. admonition:: Citation

    Kelley, D. R., Yuan, H., Huang, X. & Linder, J. Cerberus: bidirectional state space blocks improve accuracy and efficiency of regulatory sequence models. bioRxiv (2026).

.. admonition:: License

    The original model is licensed under the `Apache License, version 2.0 <https://github.com/calico/baskerville/blob/main/LICENSE>`_.

Usage
-------------------

.. code-block:: python
    :linenos:

    import crested
    import keras

    # download model
    model_path, output_names = crested.get_model("cerberus_human_fold0")

    # load model
    model = crested.utils.load_model(model_path)

    # load the model with custom_objects as fallback
    # from crested.tl.zoo.utils import Hydra
    # model = crested.utils.load_model(
    #     model_path,
    #     custom_objects={
    #         'Hydra': Hydra
    #     }
    # )

    # make predictions
    sequence = "A" * 786432
    predictions = crested.tl.predict(sequence, model)
    print(predictions.shape)
