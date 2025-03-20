DeepChickenBrain2
=================

.. sidebar:: Model Features

   - **Genome**: *galGal6*
   - **Type**: Peak Regression
   - **Parameters**: 6.3M
   - **Size**: 23MB
   - **Input shape**: (2114, 4)
   - **Output shape**: (20,)


The **DeepChickenBrain2** model is a peak regression model fine-tuned to cell type-specific regions for cell types in the chicken telencephalon.

After pretraining on all consensus peaks, the model was fine-tuned to specific peaks obtained with the :func:`~crested.pp.filter_regions_on_specificity` function. These sets of regions were then used as input to the model, where 2114bp one-hot encoded DNA sequences were used to per cell type the mean peak accessibility over the center 1000 bp of the peak.

Peak heights were normalized across cell types with the :func:`~crested.pp.normalize_peaks` function.

The model is a CNN multiclass regression model that uses the :func:`~crested.tl.zoo.dilated_cnn` architecture.

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
    model_path, output_names = crested.get_model("DeepChickenBrain2")

    # load model
    model = keras.models.load_model(model_path)

    # make predictions
    sequence = "A" * 500
    predictions = crested.tl.predict(sequence, model)
    print(predictions.shape)
