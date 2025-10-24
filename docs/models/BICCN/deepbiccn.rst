DeepBICCN
============

.. sidebar:: Model Features

   - **Genome**: *mm10*
   - **Type**: Peak Regression
   - **Parameters**: 6.3M
   - **Size**: 23MB
   - **Input shape**: (2114, 4)
   - **Output shape**: (19,)

The **DeepBICCN** model is a peak regression model fine-tuned to cell type-specific regions for cell types in the mouse cortex. It was used in the BICCN Challenge, to predict in vivo activity of a large set of validated enhancers. The selected model was the one that had the highest ranking out of all submitted sequence-models.

After pretraining on all consensus peaks, the model was fine-tuned to specific peaks. Specific peaks were determined through the ratio of highest and second highest peak, and the ratio of the second and third highest peak. These sets of regions were then used as input to the model, where 2114bp one-hot encoded DNA sequences were used to per cell type the mean peak accessibility over the center 1000 bp of the peak.

The model is a CNN multiclass regression model using the :func:`~crested.tl.zoo.dilated_cnn` architecture.

Details of the data and the model can be found in the original publication.

-------------------

.. admonition:: Citation

    Johansen, N.J., Kempynck, N. et al. Evaluating Methods for the Prediction of Cell Type-Specific Enhancers in the Mammalian Cortex. bioRxiv (2024). https://doi.org/10.1101/2024.08.21.609075

Usage
-------------------

.. code-block:: python
    :linenos:

    import crested
    import keras

    # download model
    model_path, output_names = crested.get_model("DeepBICCN")

    # load model
    model = keras.models.load_model(model_path)

    # make predictions
    sequence = "A" * 500
    predictions = crested.tl.predict(sequence, model)
    print(predictions.shape)
