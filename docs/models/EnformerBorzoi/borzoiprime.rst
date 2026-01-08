Borzoi Prime
============

.. sidebar:: Model Features

   - **Genome**: *modified hg38/mm10*
   - **Type**: Track prediction
   - **Parameters**: 185M
   - **Size**: 654MB
   - **Input shape**: (524288, 4)
   - **Output shape**: (12288, 5431)/(12288, 1774)

The **Borzoi Prime** model is a large model trained on pseudobulked scRNA tracks as well as base Borzoi bulk ENCODE and FANTOM DNase, CAGE and RNA-seq data from a wide variety of human and mouse tissues.
It predicts 12288 bins of 16bp, corresponding to the core 196608 bp of the input sequence.

It was originally provided based on the Baskerville package, and its weights and architecture have been ported to CREsted.
The model was trained on sequences tiled across the genome, which can be found in the `original Borzoi repository <https://github.com/calico/borzoi/tree/main/data>`_, with fold 4 as validation set, test 3 as test set, and the rest as training set. All replicates are trained on the same data folds.
Note that the original human training used a modified hg38 genome, where the allele with maximum frequency in gnomAD was substituted at each position.
The original model has a shared trunk and two organism-specific heads, which are provided as two specific models for human and mouse here, resulting in models `borzoiprime_human_rep0`-`borzoiprime_human_rep3` and `borzoiprime_mouse_rep0`-`borzoiprime_mouse_rep3`.

The model is a CNN+Transformer+Upsampling model using the :func:`~crested.tl.zoo.borzoi_prime` architecture.

Details of the data and the model can be found in the original publication.

-------------------

.. warning::

    The Borzoi (Prime) architecture uses custom layers that are serialized inside the CREsted package. To ensure that the model is loaded correctly, make sure that CREsted is imported before loading the model.

.. admonition:: Citation

    Linder, J., Yuan, H. and Kelley, D.R. Predicting cell type-specific coverage profiles from DNA sequence. bioRxiv (2025). https://doi.org/10.1101/2025.06.10.658961

.. admonition:: License

    The original model is licensed under the `Apache License, version 2.0 <https://github.com/calico/borzoi-paper/blob/main/LICENSE>`_.

Usage
-------------------

.. code-block:: python
    :linenos:

    import crested
    import keras

    # download model
    model_path, output_names = crested.get_model("borzoiprime_human_rep0")

    # load model
    model = keras.models.load_model(model_path)

    # make predictions
    sequence = "A" * 524288
    predictions = crested.tl.predict(sequence, model)
    print(predictions.shape)
