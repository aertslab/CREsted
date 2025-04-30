DeepPBMC
============

.. sidebar:: Model Features

   - **Genome**: *hg38*
   - **Type**: Peak Regression
   - **Parameters**: 18.9M
   - **Size**: 49MB
   - **Input shape**: (2114, 4)
   - **Output shape**: (7,)

The **DeepPBMC** model is a peak regression model trained to predict genomic region accessibility over seven cell types from human PBMC data.

The model is pre-trained on a set of 278K consensus peaks, followed by fine-tuning on 51K cell type-specific peaks.

The model is a CNN multiclass regression model using the :func:`~crested.tl.zoo.dilated_cnn` architecture.

Details of the data and the model can be found in the original publication.

-------------------

.. admonition:: Citation

   Kempynck, N., De Winter, S., et al. CREsted: modeling genomic and synthetic cell type-specific enhancers across tissues and species. Zenodo. https://doi.org/10.5281/zenodo.13918932

Usage
-------------------

.. code-block:: python
    :linenos:

    import crested
    import keras

    # download model
    model_path, output_names = crested.get_model("DeepPBMC")

    # load model
    model = keras.models.load_model(model_path)

    # make predictions
    sequence = "A" * 2114
    predictions = crested.tl.predict(sequence, model)
    print(predictions.shape)
