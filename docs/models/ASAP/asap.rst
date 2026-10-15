ASAP
====

.. sidebar:: Model Features

   - **Genome**: *T2T-CHM13v2.0*
   - **Type**: Peak regression
   - **Parameters**: 6.3M
   - **Size**: ~25MB
   - **Input shape**: (2114, 4)
   - **Output shape**: (33,)

The **ASAP** model is a peak regression model fine-tuned to cell type-specific regions for 33 cell types across two human brain regions: 24 cell types from the cingulate cortex (CC) and 9 cell types from the substantia nigra (SN). Output classes are suffixed with their brain region (e.g. ``Astro_CC``, ``Astro_SN``).

This model supersedes the separate :doc:`ASAP_CC <asap_cc>` and :doc:`ASAP_SN <asap_sn>` models.

The model was first pretrained on all 2.12M consensus peaks of the combined CC and SN datasets, and then fine-tuned to 243K cell type-specific peaks, selected as the top 10,000 most specific regions per cell type (proportion-based specificity score, after deduplication). Peak heights were calculated by taking the cut site counts in the central 1000bp of resized peaks from pseudobulk cut site bigWigs, normalized across cell types using min-max normalization with the top 3% of constitutive peaks per cell type. 2114bp one-hot encoded DNA sequences (centered on consensus peak summits) were used to predict peak accessibilities across cell types. Train/validation/test split was done by chromosome (chr2 and chr22 = validation, chr4 and chr9 = test).

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
    model_path, output_names = crested.get_model("ASAP")

    # load model
    model = crested.utils.load_model(model_path)

    # make predictions
    sequence = "A" * 2114
    predictions = crested.tl.predict(sequence, model)
    print(predictions.shape)
