BorzoiBICCN
============

.. sidebar:: Model Features

   - **Genome**: *mm10*
   - **Type**: Peak Regression
   - **Parameters**: 511M
   - **Size**: 1.6GB
   - **Input shape**: (2048, 4)
   - **Output shape**: (19,)

The **BorzoiBICCN** model is Borzoi model that is (double) fine-tuned to perform peak regression on motor cortex cell types from the BICCN dataset. The model was first fine-tuned on all consensus peaks (440K regions),
and further fine-tuned on a training set of cell type-specific peaks (73K regions).

The model is a CNN multiclass regression model using the :func:`~crested.tl.zoo.borzoi` architecture, with an added dense layer after the final embedding layer to predict peak heights over cell types.

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
    model_path, output_names = crested.get_model("BorzoiBICCN")

    # load model
    model = keras.models.load_model(model_path)

    # make predictions
    sequence = "A" * 2048
    predictions = crested.tl.predict(sequence, model)
    print(predictions.shape)

-------------------

.. warning::

    The borzoi architecture uses custom layers that are serialized inside the CRESted package. To ensure that the model is loaded correctly, make sure that crested is imported before loading the model.
