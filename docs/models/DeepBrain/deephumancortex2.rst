DeepHumanCortex2
================

.. sidebar:: Model Features

   - **Genome**: *hg38*
   - **Type**: Topic Classification
   - **Parameters**: 13.9M
   - **Size**: 47MB
   - **Input shape**: (500, 4)
   - **Output shape**: (14,)


The **DeepHumanCortex2** model is a topic classification model, fine-tuned with differential accessible regions (DARs) to make cell type level predictions for cell types in the human motor cortex. The dataset was obtained from Bakken et al., 2021(Science).

After pretraining on topics, obtained through `pycistopic <https://pycistopic.readthedocs.io/en/latest/>`_, DARs were calculated per cell type and used as cell type representation. These sets of regions were then used as input to the model, where 500bp one-hot encoded DNA sequences were used to predict the cell type(s) to which the regions belong.

The model is a CNN multiclass classifier which is uses the :func:`~crested.tl.zoo.deeptopic_cnn` architecture.

Details of the data and the model can be found in the original publication.

-------------------

.. admonition:: Citation

    Hecker, N., Kempynck, N. et al. Enhancer-driven cell type comparison reveals similarities between the mammalian and bird pallium. bioRxiv (2024). https://doi.org/10.1101/2024.04.17.589795

Usage
-------------------

.. code-block:: python
    :linenos:

    import crested
    import keras

    # download model
    model_path, output_names = crested.get_model("DeepHumanCortex2")

    # load model
    model = keras.models.load_model(model_path)

    # make predictions
    sequence = "A" * 500
    predictions = crested.tl.predict(sequence, model)
    print(predictions.shape)
