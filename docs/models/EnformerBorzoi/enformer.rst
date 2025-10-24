Enformer
============

.. sidebar:: Model Features

   - **Genome**: *hg38/mm10*
   - **Type**: Track Prediction
   - **Parameters**: 236M
   - **Size**: 873MB
   - **Input shape**: (196608, 4)
   - **Output shape**: (896, 5313)/(896, 1643)

The **Enformer** model is a large model trained on bulk ENCODE and FANTOM DNase, ChIP-seq, and CAGE data from a wide variety of human and mouse tissues.
It predicts 896 bins of 128bp, corresponding to the core 114688 bp of the input sequence.

It was originally provided based on the Sonnet package, and its weights and architecture have been ported to CREsted.
The model was trained on sequences tiled across the genome, which can be downloaded from the `original authors' Google Cloud bucket <https://console.cloud.google.com/storage/browser/basenji_barnyard2>`_.
The original model has a shared trunk and two organism-specific heads, which are provided as two specific models for human and mouse here, resulting in models `enformer_human` and `enformer_mouse`.

The model is a CNN+Transformer model using the :func:`~crested.tl.zoo.enformer` architecture.

Details of the data and the model can be found in the original publication.

-------------------

.. admonition:: Citation

    Avsec, Ž., Agarwal, V., Visentin, D. et al. Effective gene expression prediction from sequence by integrating long-range interactions. Nat Methods 18, 1196–1203 (2021). https://doi.org/10.1038/s41592-021-01252-x

Usage
-------------------

.. code-block:: python
    :linenos:

    import crested
    import keras

    # download model
    model_path, output_names = crested.get_model("enformer_human")

    # load model
    model = keras.models.load_model(model_path)

    # make predictions
    sequence = "A" * 196608
    predictions = crested.tl.predict(sequence, model)
    print(predictions.shape)
