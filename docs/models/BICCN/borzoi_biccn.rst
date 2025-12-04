BorzoiBICCN
============

.. sidebar:: Model Features

   - **Genome**: *mm10*
   - **Type**: Peak regression
   - **Parameters**: 511M
   - **Size**: 1.6GB
   - **Input shape**: (2048, 4)
   - **Output shape**: (19,)

The **BorzoiBICCN** model is a Borzoi model (Linder et al., 2025) that is (double) fine-tuned to perform peak regression on motor cortex cell types from the BICCN dataset. The model was first fine-tuned on all consensus peaks (440K regions),
and further fine-tuned on a training set of cell type-specific peaks (73K regions).

The model is a CNN multiclass regression model using the :func:`~crested.tl.zoo.borzoi` architecture, with a shrunk input size (2048bp) and an added dense layer after the final embedding layer to predict peak heights over cell types.

Details of the data and the model can be found in the original publication. The training data can be downloaded with `crested.get_dataset('mouse_cortex_bigwig_cut_sites')`.

-------------------

.. warning::

    The Borzoi architecture uses custom layers that are serialized inside the CREsted package. To ensure that the model is loaded correctly, make sure that CREsted is imported before loading the model.

.. admonition:: Citation

   Kempynck, N., De Winter, S., et al. CREsted: modeling genomic and synthetic cell type-specific enhancers across tissues and species. bioRxiv (2025). https://doi.org/10.1101/2025.04.02.646812

.. admonition:: Data source

   Zemke, N.R., Armand, E.J., et al. Conserved and divergent gene regulatory programs of the mammalian neocortex. Nature (2023). https://doi.org/10.1038/s41586-023-06819-6

.. admonition:: License

    The original Borzoi model is licensed under the `Apache License, version 2.0 <https://github.com/calico/borzoi/blob/main/LICENSE>`_.

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
