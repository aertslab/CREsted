STEAM_evolution_aware
=====================

.. sidebar:: Model Features

   - **Genome**: *mm10*
   - **Type**: Peak regression
   - **Parameters**: 6.3M
   - **Size**: ~67MB
   - **Input shape**: (2114, 4)
   - **Output shape**: (32,)

The **STEAM_evolution_aware** model is an intermediate model from the evolutionary transfer learning study. Like :doc:`STEAM_v1 <steam_v1>`, it predicts cell-class-specific chromatin accessibility across 32 cell classes of the developing mouse embryo (E10–P0), but it is trained only on the mouse windows that survive evolutionary-coherence filtering against syntenic orthologs across 240 mammalian genomes, before the full synteny-supervised ortholog expansion of STEAM_v1.

It is provided to illustrate the contribution of evolutionary filtering to the final STEAM approach. Evolutionary information enters this model only through *window selection* — the training sequences themselves are all mm10. 2114bp one-hot encoded DNA sequences are used to predict normalized Tn5 cut-site accessibility (log-scaled) over the central region of each window, per cell class.

The model is a CNN multiclass regression model using the :func:`~crested.tl.zoo.dilated_cnn` architecture. Like the other STEAM models, it maps DNA sequence directly to accessibility and is genome-agnostic at inference (applicable to any mammalian genome).

Details of the data and the model can be found in the original publication.

-------------------

.. admonition:: Citation

   Qiu, C., Daza, R.M., Welsh, I.C. et al. Evolutionary transfer learning enables organism-wide inference of mammalian enhancer landscapes (2026). https://doi.org/10.62329/hxkk6249

Usage
-------------------

.. code-block:: python
    :linenos:

    import crested
    import keras

    # download model
    model_path, output_names = crested.get_model("STEAM_evolution_aware")

    # load model
    model = crested.utils.load_model(model_path)

    # make predictions
    sequence = "A" * 2114
    predictions = crested.tl.predict(sequence, model)
    print(predictions.shape)
