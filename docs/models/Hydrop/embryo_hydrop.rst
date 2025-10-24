EmbryoHydrop
============

.. sidebar:: Model Features

   - **Genome**: *dm6*
   - **Type**: Peak Regression
   - **Parameters**: 9.6M
   - **Size**: 34MB
   - **Input shape**: (500, 4)
   - **Output shape**: (20,)

The **Embryo10x** and **EmbryoHydrop** models are peak regression models trained on on ATAC coverage from the same cell types that were captured using different technologies to show the similarities between these methods.

Both models were trained using the same preprocessing steps and model architecture.

For preprocessing, the regions in chromosome 2R were evenly divided into two to use as validation and test sets. The remaining chromosomes were used for training.
Peak heights were normalized per chromosome to a target mean accessibility of 0.5. After normalization, z-scores of peak heights were calculated per region.
For each cell type, top 3000 regions with the highest z-scores were kept and the accessibility values of all the other regions were set to zero for that cell type

For model training, CosineMSELoss (from `crested.tl.losses`) was used along with the default optimizer and metrics from default CREsted peak regression configuration.
The model is a CNN multiclass regression model using the :func:`~crested.tl.zoo.deeptopic_cnn` architecture with a softplus output activation  and the following parameters: filters=500, conv_do=0.5

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
    model_path, output_names = crested.get_model("EmbryoHydrop")

    # load model
    model = keras.models.load_model(model_path)

    # make predictions
    sequence = "A" * 500
    predictions = crested.tl.predict(sequence, model)
    print(predictions.shape)
