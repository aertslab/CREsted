DeepMouseBrain1
===============

.. sidebar:: Model Features

   - **Genome**: *mm10*
   - **Type**: Topic Classification
   - **Parameters**: 12.9M
   - **Size**: 44MB
   - **Input shape**: (500, 4)
   - **Output shape**: (23,)


The **DeepMouseBrain1** model is a topic classification model, fine-tuned with differential accessible regions (DARs) to make cell type level predictions for cell types in the mouse neocortex and hippocampus. The dataset was obtained from Bravo González-Blas & De Winter et al., 2023 (Nature Methods).

After pretraining on topics, obtained through `pycistopic <https://pycistopic.readthedocs.io/en/latest/>`_, DARs were calculated per cell type and used as cell type representation. These sets of regions were then used as input to the model, where 500bp one-hot encoded DNA sequences were used to predict the cell type(s) to which the regions belong.

The model is a CNN multiclass classifier which uses the :func:`~crested.tl.zoo.deeptopic_cnn` architecture.

Details of the data and the model can be found in the original publication.

-------------------

.. admonition:: Citation

    Hecker, N., Kempynck, N. et al. Enhancer-driven cell type comparison reveals similarities between the mammalian and bird pallium. bioRxiv (2024). https://doi.org/10.1101/2024.04.17.589795

Usage
-------------------

.. code-block:: python
    :linenos:

    import crested
    import keras

    # download model
    model_path, output_names = crested.get_model("DeepMouseBrain1")

    # load model
    model = keras.models.load_model(model_path)

    # make predictions
    sequence = "A" * 500
    predictions = crested.tl.predict(sequence, model)
    print(predictions.shape)
