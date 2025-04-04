MouseCortexHydrop
=================

.. sidebar:: Model Features

   - **Genome**: *mm10*
   - **Type**: Peak Regression
   - **Parameters**: 18.9M
   - **Size**: 44MB
   - **Input shape**: (2114, 4)
   - **Output shape**: (17,)

The **MouseCortexHydrop** model is a peak regression model fine-tuned to cell type-specific regions for cell types in the mouse cortex. It was trained in the same way and on the same cell types as the **DeepBICCN** model to show similarities between the Hydrop and 10x technologies.

After pretraining on all consensus peaks, the model was fine-tuned to specific peaks. Specific peaks were determined through the ratio of highest and second highest peak, and the ratio of the second and third highest peak. These sets of regions were then used as input to the model, where 2114bp one-hot encoded DNA sequences were used to per cell type predict the Tn5 cut-site counts over the center 1000 bp of the peak.

The model is a CNN multiclass regression model using the :func:`~crested.tl.zoo.dilated_cnn` architecture.

Details of the data and the model can be found in the original publication.

-------------------

.. admonition:: Citation

   Dickmanken, H., Wojno, M., Theunis, K., Eksi, E. C., Mahieu, L., Christiaens, V., Kempynck, N., De Rop, F., Roels, N., Spanier, K. I., Vandepoel, R., Hulselmans, G., Poovathingal, S., Aerts, S. HyDrop v2: Scalable atlas construction for training sequence-to-function models. bioRxiv doi: 10.1101/2025.04.02.646792

Usage
-------------------

.. code-block:: python
    :linenos:

    import crested
    import keras

    # download model
    model_path, output_names = crested.get_model("MouseCortexHydrop")

    # load model
    model = keras.models.load_model(model_path)

    # make predictions
    sequence = "A" * 2114
    predictions = crested.tl.predict(sequence, model)
    print(predictions.shape)
