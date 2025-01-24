DeepLiver Accessibility
=======================

.. sidebar:: Model Features

   - **Genome**: *mm10*
   - **Type**: Topic Classification
   - **Parameters**: 12.2M
   - **Size**: 44MB
   - **Input shape**: (500, 4)
   - **Output shape**: (82,)

The **DeepLiver Accessibility** model is a topic classification model trained on 220K annotated regions from hepatocytes in the mouse liver.

Using `pycistopic <https://pycistopic.readthedocs.io/en/latest/>`_, binarized topics per region were extracted for 82 target topics. These sets of regions were then used as input for a DL model, where 500bp one-hot encoded (ACGT) DNA sequences were used to predict the topic set to which the region belongs.

The model is a hybrid CNN-RNN multiclass classifier which is very similar to :func:`~crested.tl.zoo.deeptopic_lstm` with addition of a reverse complement layer in the first layer of the model.

Details of the data and model can be found in the original publication.

-------------------

.. admonition:: Citation

    Bravo González-Blas, C., Matetovici, I., Hillen, H. et al. Single-cell spatial multi-omics and deep learning dissect enhancer-driven gene regulatory networks in liver zonation. Nat Cell Biol 26, 153-167 (2024). https://doi.org/10.1038/s41556-023-01316-4

Usage
-------------------

.. code-block:: python
    :linenos:

    import crested
    import keras

    # download model
    model_path, output_names = crested.get_model("DeepLiver_accessibility")

    # load model
    model = keras.models.load_model(model_path)

    # make predictions
    sequence = "A" * 500
    predictions = crested.tl.predict(sequence, model)
    print(predictions.shape)

-------------------

.. warning::

    DeepLiver_Accessibility was originally trained using Tensorflow 1 as the backend.
    Even though the model architecture and weights are exactly the same, there will be slight differences in the output compared to the original model due to backend changes between Tensorflow 1 and 2.
    Overall the correlation between the original and the Keras 3 model is very high (0.99+), but if you want the exact same outputs and contribution plots as in the original publication, you should use an older, compatible environment which you can find in `kipoi <https://kipoi.org/models/DeepLiver/>`_.
