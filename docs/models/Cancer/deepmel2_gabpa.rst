DeepMEL2 GABPA
==============

.. sidebar:: Model Features

   - **Genome**: hg38
   - **Type**: Topic Classification
   - **Parameters**: 6.4M
   - **Size**: 23MB
   - **Input shape**: (500, 4)
   - **Output shape**: (48,)

The **DeepMEL2 GABPA** model is a topic classification model that is trained on the same data and with the same architecture as DeepMEL2.

On top of the 47 topics from DeepMEL2, this model includes a 48th class in which regions in input data were labeled as 1 if it overlaps with GABPA ChIP-seq peaks.

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
    model_path, output_names = crested.get_model("DeepMEL2_gabpa")

    # load model
    model = keras.models.load_model(model_path)

    # make predictions
    sequence = "A" * 500
    predictions = crested.tl.predict(sequence, model)
    print(predictions.shape)
