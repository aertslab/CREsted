Borzoi
============

.. sidebar:: Model Features

   - **Genome**: *hg38/mm10*
   - **Type**: Track Prediction
   - **Parameters**: 235M
   - **Size**: 666MB
   - **Input shape**: (524288, 4)
   - **Output shape**: (6144, 7611)/(6144, 2608)

The **Borzoi** model is a large model trained on bulk ENCODE and FANTOM DNase, ChIP-seq, CAGE and RNA-seq data from a wide variety of human and mouse tissues.
It predicts 6144 bins of 32bp, corresponding to the core 196608 bp of the input sequence.

It was originally provided based on the Baskerville package, and its weights and architecture have been ported to CREsted.
The model was trained on sequences tiled across the genome, which can be found in the `original repository <https://github.com/calico/borzoi/tree/main/data>`_, with fold 4 as validation set, test 3 as test set, and the rest as training set. All replicates are trained on the same data folds.
The original model has a shared trunk and two organism-specific heads, which are provided as two specific models for human and mouse here, resulting in models `borzoi_human_rep0`-`borzoi_human_rep3` and `borzoi_mouse_rep0`-`borzoi_mouse_rep3`.

The model is a CNN+Transformer+Upsampling model using the :func:`~crested.tl.zoo.borzoi` architecture.

Details of the data and the model can be found in the original publication.

-------------------

.. admonition:: Citation

    Linder, J., Srivastava, D., Yuan, H. et al. Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation. Nat Genet (2025). https://doi.org/10.1038/s41588-024-02053-6

Usage
-------------------

.. code-block:: python
    :linenos:

    import crested
    import keras

    # download model
    model_path, output_names = crested.get_model("borzoi_human_rep0")

    # load model
    model = keras.models.load_model(model_path)

    # make predictions
    sequence = "A" * 524288
    predictions = crested.tl.predict(sequence, model)
    print(predictions.shape)
