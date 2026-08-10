STEAM_v1
========

.. sidebar:: Model Features

   - **Genome**: *mm10* + orthologs from 240 Zoonomia genomes
   - **Type**: Peak regression
   - **Parameters**: 6.3M
   - **Size**: ~59MB
   - **Input shape**: (2114, 4)
   - **Output shape**: (32,)

The **STEAM_v1** model is the final, evolution-augmented peak regression model from the evolutionary transfer learning study. It predicts cell-class-specific chromatin accessibility across 32 cell classes of the developing mouse embryo (E10–P0), profiled by single-cell ATAC-seq across 3.9 million nuclei.

STEAM (Synteny-aware Transfer learning for Enhancer Activity Modeling) expands the mm10 mouse training corpus with synteny-anchored enhancer orthologs from up to 241 mammalian genomes, in a synteny-supervised manner. The 2114bp mouse windows are lifted over to the 240 `Zoonomia <https://doi.org/10.1126/science.abn3943>`_ genomes; orthologs are retained when syntenically conserved (in at least half of the species) and concordant with the evolution-naive predictions (Pearson r ≥ 0.6), and are assigned the chromatin accessibility profile of their corresponding mouse window. This increases the effective training scale by up to ~195-fold (0.3M → ~58M sequences) and improves generalization for organism-wide inference of mammalian enhancer landscapes. 2114bp one-hot encoded DNA sequences are used to predict normalized Tn5 cut-site accessibility (log-scaled) over the central region of each window, per cell class.

Multi-species training is done with **unmodified CREsted**: the qualifying ortholog sequences are concatenated into a single custom genome FASTA, and added to the training :class:`~anndata.AnnData` as additional regions that inherit their parent mouse window's labels. One :func:`~crested.register_genome` call over this combined genome then lets the standard :class:`~crested.tl.data.AnnDataModule` / :class:`~crested.tl.Crested` pipeline fetch and one-hot encode every species' sequence on the fly, with chromosome-based train/validation/test splits shared between mouse and orthologs to avoid leakage.

The model is a CNN multiclass regression model using the :func:`~crested.tl.zoo.dilated_cnn` architecture. Because it maps DNA sequence directly to accessibility, it is genome-agnostic at inference and can be applied to any mammalian genome — in the paper it is used to predict enhancer landscapes in human (hg38, hg19) and mouse (mm10).

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
    model_path, output_names = crested.get_model("STEAM_v1")

    # load model
    model = crested.utils.load_model(model_path)

    # make predictions
    sequence = "A" * 2114
    predictions = crested.tl.predict(sequence, model)
    print(predictions.shape)
