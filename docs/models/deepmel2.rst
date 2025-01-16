DeepMEL2
============

.. sidebar:: Model Features

   - **Genome**: hg38
   - **Type**: Topic Classification
   - **Parameters**: 6.4M
   - **Size**: 23MB
   - **Input shape**: (500, 4)
   - **Output shape**: (47,)

The **DeepMEL2** model is a topic classification model that is very similar to the original **DeepMEL1** model, but trained on more ATAC-seq samples (n=30) with a larger model architecture.

Using `pycistopic <https://pycistopic.readthedocs.io/en/latest/>`_, binarized topics per region were extracted for 47 target topics representing melanoma-specific coaccessible regions.

These sets of regions were used as input for a DL model, where 500bp one-hot encoded (ACGT) DNA sequences were used to predict the topic set to which the region belongs.

The model is a hybrid CNN-RNN multiclass classifier which is very similar to :func:`~crested.tl.zoo.deeptopic_lstm` with addition of a reverse complement layer in the first layer of the model.

DeepMEL2 uses 300 filters in its first convolutional layer instead of 128 in DeepMEL1, this time also initialized with JASPAR PWMs.

Details of the data and model can be found in the original publication.

-------------------

.. admonition:: Citation

    Atak, Z.K., Taskiran, I.I. et al. Interpretation of allele-specific chromatin accessibility using cell state-aware deep learning. Genome Res. 31, 1082–1096 (2021). https://doi.org/10.1101/gr.260851.120

Usage
-------------------

.. code-block:: python
    :linenos:

    import crested
    import keras

    # download model
    model_path, output_names = crested.get_model("DeepMEL2")

    # load model
    model = keras.models.load_model(model_path)

    # make predictions
    sequence = "A" * 500
    predictions = crested.tl.predict(sequence, model)
    print(predictions.shape)
