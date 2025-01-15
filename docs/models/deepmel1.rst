DeepMEL1
============

.. sidebar:: Model Features

   - **Genome**: hg38
   - **Type**: Topic Classification
   - **Parameters**: 3.4M
   - **Size**: 13MB
   - **Input shape**: (500, 4)
   - **Output shape**: (24,)

The **DeepMEL1** model is a topic classification model trained on 339K ATAC-seq peaks from 16 human samples with the goal of investigating MEL- and MES-state enhancer logic.

Using `pycistopic <https://pycistopic.readthedocs.io/en/latest/>`_, binarized topics per region were extracted for 24 target topics, where topic 4 and topic 7 represent MEL- and MES-specific enhancers.
In addition, two topics have regions that are generally accessible accross all cell lines (topic 1 and topic 19).

These sets of regions were used as input for a DL model, where 500bp one-hot encoded (ACGT) DNA sequences were used to predict the topic set to which the region belongs.

The model is a hybrid CNN-RNN multiclass classifier which is very similar to :func:`~crested.tl.zoo.deeptopic_lstm` with addition of a reverse complement layer in the first layer of the model.

Details of the data and model can be found in the original publication.

-------------------

.. admonition:: Citation

    Minoye, L., Taskiran, I.I. et al. Cross-species analysis of enhancer logic using deep learning. Genome Res. 30, 1815–1834 (2020). https://doi.org/10.1101/gr.260844.120

Usage
-------------------

.. code-block:: python
    :linenos:

    import crested
    import keras

    # download model
    model_path, output_names = crested.get_model("DeepMEL1")

    # load model
    model = keras.models.load_model(model_path)

    # make predictions
    sequence = "A" * 500
    predictions = crested.tl.predict(sequence, model)
    print(predictions.shape)
