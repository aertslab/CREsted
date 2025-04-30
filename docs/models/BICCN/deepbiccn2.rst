DeepBICCN2
============

.. sidebar:: Model Features

   - **Genome**: *mm10*
   - **Type**: Peak Regression
   - **Parameters**: 18.9M
   - **Size**: 60MB
   - **Input shape**: (2114, 4)
   - **Output shape**: (19,)

The **DeepBICCN2** model is a peak regression model fine-tuned to cell type-specific regions for cell types in the mouse cortex. Differently to **DeepBICCN**, this model is trained on Tn5 cut-site counts instead of mean coverage.

After pretraining on all consensus peaks, the model was fine-tuned to specific peaks. Specific peaks were determined through the ratio of highest and second highest peak, and the ratio of the second and third highest peak. These sets of regions were then used as input to the model, where 2114bp one-hot encoded DNA sequences were used to per cell type predict the Tn5 cut-site counts over the center 1000 bp of the peak.

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
    model_path, output_names = crested.get_model("DeepBICCN2")

    # load model
    model = keras.models.load_model(model_path)

    # make predictions
    sequence = "A" * 2114
    predictions = crested.tl.predict(sequence, model)
    print(predictions.shape)
