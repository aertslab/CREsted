ASAP_SN
=======

.. sidebar:: Model Features

   - **Genome**: *T2T-CHM13v2.0*
   - **Type**: Peak regression
   - **Parameters**: 6.3M
   - **Size**: ~53MB
   - **Input shape**: (2114, 4)
   - **Output shape**: (9,)

The **ASAP_SN** model is a peak regression model fine-tuned to cell type-specific regions for 9 cell types in the human substantia nigra (SN).

After pretraining on all 977K consensus peaks, the model was fine-tuned to cell type-specific peaks, defined as regions with a Gini index 1 standard deviation above the mean across all regions (194K peaks). Peak heights were calculated by taking mean coverage in the central 1000bp of resized peaks, normalized across cell types using min-max normalization with the top 3% of constitutive peaks per cell type. 2114bp one-hot encoded DNA sequences (centered on consensus peak summits) were used to predict mean peak accessibilities across cell types. Train/validation/test split was done by chromosome (chr2 and chr22 = validation, chr4 and chr9 = test).

The model is a CNN multiclass regression model using the :func:`~crested.tl.zoo.dilated_cnn` architecture.

Details of the data and the model can be found in the original publication.

-------------------

.. admonition:: Citation

   Sigalova, O.M., Pančíková, A., De Man, J., Theunis, K. et al. Modeling cis-regulatory variation in human brain enhancers across a large Parkinson's Disease cohort. bioRxiv (2026). https://doi.org/10.64898/2026.03.15.711881

Usage
-------------------

.. code-block:: python
    :linenos:

    import crested
    import keras

    # download model
    model_path, output_names = crested.get_model("ASAP_SN")

    # load model
    model = crested.utils.load_model(model_path)

    # make predictions
    sequence = "A" * 2114
    predictions = crested.tl.predict(sequence, model)
    print(predictions.shape)
