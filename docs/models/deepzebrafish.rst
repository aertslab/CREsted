DeepZebraFish
=============

.. sidebar:: Model Features

   - **Genome**: *GRCz11*
   - **Type**: Peak Regression
   - **Parameters**: 77.6M
   - **Size**: 207MB
   - **Input shape**: (2114, 4)
   - **Output shape**: (639,)

The **DeepZebraFish** model is a peak regression model trained on a scATAC-seq dataset of the developing zebrafish embryo.
This dataset comprises 20 developmental stages, and 639 cell type-timepoint-combinations that were used as separate classes for the model.

The model was trained on a set of 793K consensus peaks and fine-tuned on 89K cell type-timepoint-specific peaks.

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
    model_path, output_names = crested.get_model("DeepZebraFish")

    # load model
    model = keras.models.load_model(model_path)

    # make predictions
    sequence = "A" * 2114
    predictions = crested.tl.predict(sequence, model)
    print(predictions.shape)
