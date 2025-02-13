DeepFlyBrain
============

.. sidebar:: Model Features

   - **Genome**: *dm6*
   - **Type**: Topic Classification
   - **Parameters**: 3.2M
   - **Size**: 12MB
   - **Input shape**: (500, 4)
   - **Output shape**: (81,)

The **DeepFlyBrain** model is a topic classification model trained on KCs, T-Neurons, and Glia cells from the adult fly brain (17K cells total).

Using `pycistopic <https://pycistopic.readthedocs.io/en/latest/>`_, binarized topics per region were extracted for 81 target topics. These sets of regions were then used as input for a DL model, where 500bp one-hot encoded (ACGT) DNA sequences were used to predict the topic set to which the region belongs.

The model is a hybrid CNN-RNN multiclass classifier which is very similar to :func:`~crested.tl.zoo.deeptopic_lstm` with addition of a reverse complement layer in the first layer of the model.

Details of the data and model can be found in the original publication.

-------------------

.. admonition:: Citation

    Janssens, J., Aibar, S., Taskiran, I.I. et al. Decoding gene regulation in the fly brain. Nature 601, 630–636 (2022). https://doi.org/10.1038/s41586-021-04262-z

Usage
-------------------

.. code-block:: python
    :linenos:

    import crested
    import keras

    # download model
    model_path, output_names = crested.get_model("DeepFlyBrain")

    # load model
    model = keras.models.load_model(model_path)

    # make predictions
    sequence = "A" * 500
    predictions = crested.tl.predict(sequence, model)
    print(predictions.shape)

-------------------

.. warning::

    DeepFlyBrain was originally trained using Tensorflow 1 as the backend.
    Even though the model architecture and weights are exactly the same, there will be slight differences in the output compared to the original model due to backend changes between Tensorflow 1 and 2.
    Overall the correlation between the original and the Keras 3 model is very high (0.99+), but if you want the exact same outputs and contribution plots as in the original publication, you should use an older, compatible environment which you can find in `kipoi <https://kipoi.org/models/DeepFlyBrain/>`_.
