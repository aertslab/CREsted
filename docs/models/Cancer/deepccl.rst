DeepCCL
============

.. sidebar:: Model Features

   - **Genome**: *hg38*
   - **Type**: Peak Regression
   - **Parameters**: 12.6M
   - **Size**: 47MB
   - **Input shape**: (2114, 4)
   - **Output shape**: (8,)

The **DeepCCL** model is a peak regression model trained on an ATAC-seq dataset of cancer cell lines, using two ENCODE deeply-profiled-cell lines, namely HepG2 and GM12878; three melanoma cell lines (2 mesenchymal-like, MM029 and MM099, and one melanocytic-like, MM001); and three GBM cell lines (A172, M059J, and LN229).

The model was trained on a set of 415K consensus peaks and fine-tuned on 207K cell type-specific peaks, where peak heights were normalized across cell types with the normalize_peaks() function.

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
    model_path, output_names = crested.get_model("DeepCCL")

    # load model
    model = keras.models.load_model(model_path)

    # make predictions
    sequence = "A" * 2114
    predictions = crested.tl.predict(sequence, model)
    print(predictions.shape)
