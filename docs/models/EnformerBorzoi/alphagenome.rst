AlphaGenome
============

.. sidebar:: Model Features

   - **Genome**: *hg38/mm10*
   - **Type**: Track prediction
   - **Parameters**: 421M/407M
   - **Size**: 1.7GB/1.6GB
   - **Input shape**: (1048576, 4)
   - **Output shape**: per head (1048576, tracks) at 1bp and/or (8192, tracks) at 128bp; splice sites (1048576, 5); splice site usage (1048576, 734)/(1048576, 180); contact maps (512, 512, 28)/(512, 512, 8)

The **AlphaGenome** model is a large model trained on bulk RNA-seq, CAGE, PRO-cap, DNase, ATAC, histone and TF ChIP-seq
and Hi-C/Micro-C data from a wide variety of human and mouse tissues and cell types (ENCODE, GTEx, FANTOM5, 4DN).
From a 1 Mb input sequence, it predicts tracks at 1 bp and/or 128 bp resolution, splice sites and their usage at 1 bp resolution, and contact maps at 2048 bp resolution, covering the full input.

It was originally provided by Google DeepMind (JAX), and its weights were ported to PyTorch by `alphagenome-pytorch <https://github.com/genomicsxai/alphagenome-pytorch>`_,
from which they have been ported to CREsted.

Available models
-------------------

The models are named `alphagenome_{organism}[_full]_{weights}`, combining three choices:

- **Organism** (`human` or `mouse`): the original AlphaGenome is a single model for both organisms, with a shared trunk, organism embeddings and organism-specific output heads.
  In CREsted, it is split into one model per organism, with the organism embedding built into the weights, so the model only needs a sequence as input.
  The human and mouse models therefore share the same trunk weights, but have their own output heads and tracks.
  The mouse models have fewer tracks, and no PRO-cap outputs.
- **Full or simplified** (no suffix or `_full`): the full models contain all ported output heads: the genomic track heads, the splice site heads and the contact map head.
  The default models only contain the genomic track heads (ATAC, DNase, PRO-cap, CAGE, RNA-seq, ChIP-seq).
  Their track predictions are identical to the full models', but they skip computing the splice site and contact map outputs, of which especially the 1 bp splice site usage output is large.
  Use these unless you need the splicing or contact map predictions.
  Note that both still internally compute the pairwise representation of the sequence, as it is part of the model's transformer tower.
- **Weights** (`all_folds` or `fold0`-`fold3`): the `all_folds` model was trained on all data and is recommended.
  The four fold models were each trained with a different held-out fold of the genome, e.g. to evaluate on unseen regions or to ensemble.
  The names without weights suffix are aliases for `all_folds`: `'alphagenome_human'`, `'alphagenome_mouse'`, `'alphagenome_human_full'` and `'alphagenome_mouse_full'`.

This gives 20 models in total. Their outputs and number of tracks per output are:

.. list-table::
   :header-rows: 1

   * - Output
     - Resolution
     - Human tracks
     - Mouse tracks
     - In base models
   * - `atac_1bp`, `atac_128bp`
     - 1 bp, 128 bp
     - 167
     - 18
     - yes
   * - `dnase_1bp`, `dnase_128bp`
     - 1 bp, 128 bp
     - 305
     - 67
     - yes
   * - `procap_1bp`, `procap_128bp`
     - 1 bp, 128 bp
     - 12
     - (none)
     - yes
   * - `cage_1bp`, `cage_128bp`
     - 1 bp, 128 bp
     - 546
     - 188
     - yes
   * - `rna_seq_1bp`, `rna_seq_128bp`
     - 1 bp, 128 bp
     - 667
     - 173
     - yes
   * - `chip_tf`
     - 128 bp
     - 1617
     - 127
     - yes
   * - `chip_histone`
     - 128 bp
     - 1116
     - 183
     - yes
   * - `splice_sites`
     - 1 bp
     - 5
     - 5
     - no
   * - `splice_site_usage`
     - 1 bp
     - 734
     - 180
     - no
   * - `contact_maps`
     - 2048 bp (pairwise)
     - 28
     - 8
     - no

The `_full` models have 421M (human) and 407M (mouse) parameters, the base models 420M and 407M; all are ~0.75 GB to download.

Architecture and outputs
------------------------

The model uses the :func:`~crested.tl.zoo.alphagenome` architecture: a convolutional U-Net around a transformer tower that also builds a pairwise representation of the sequence.
It has one output per head and resolution, named `'{head}_1bp'`/`'{head}_128bp'` (`'atac'`, `'dnase'`, `'procap'` (human only), `'cage'`, `'rna_seq'`), `'chip_tf'` and `'chip_histone'` (128 bp only),
followed in the full models by `'splice_sites'` (probabilities of donor/acceptor on the +/- strand or no splice site), `'splice_site_usage'` (fraction of transcripts using each splice site per tissue and strand) and `'contact_maps'`.
The splice junction head is not included. Padding tracks of the original heads are removed.
The output class names are `'{head}/{track name}'`, with the strand appended for stranded tracks, listing each head's tracks once in the order of the outputs.

Track predictions are in the model's scaled units, in which it was trained. To convert them to the experimental scale as the original model does,
soft-clipped values above 10 are expanded (`(x + 10)**2 / 40`), RNA-seq predictions are raised to the power 1/0.75,
and all are multiplied by the track's mean (the `nonzero_mean` column of the `AlphaGenome track metadata <https://www.alphagenomedocs.com/exploring_model_metadata.html>`_) and the resolution in bp.

Details of the data and the model can be found in the original publication.

-------------------

.. warning::

    The AlphaGenome architecture uses custom layers that are serialized inside the CREsted package. To ensure that the model is loaded correctly, make sure that CREsted and specifically `crested.tl` is imported before loading the model.

.. note::

    The model is large: predicting a full 1 Mb sequence requires a GPU with substantial memory.
    The model can be rebuilt for shorter sequences (multiples of 2048 bp) with :func:`~crested.tl.zoo.alphagenome` and the same head arguments, and the weights transferred with `model.set_weights()`, although the model was trained on 1 Mb sequences.

.. admonition:: Citation

    Avsec, Ž., Latysheva, N., Cheng, J. et al. Advancing regulatory variant effect prediction with AlphaGenome. Nature 649, 1206–1218 (2026). https://doi.org/10.1038/s41586-025-10014-0

.. admonition:: License

    The model parameters, output, and any derivatives thereof are subject to `Google DeepMind's AlphaGenome Model Terms <https://deepmind.google.com/science/alphagenome/model-terms>`_, which **forbid commercial usage**.
    The original model code and architecture is licensed under the `Apache License, version 2.0 <https://github.com/google-deepmind/alphagenome_research/blob/main/LICENSE>`_.

Usage
-------------------

.. code-block:: python
    :linenos:

    import crested
    import keras

    # download model
    model_path, output_names = crested.get_model("alphagenome_human")

    # load model
    model = crested.utils.load_model(model_path)

    # track names per head
    head_classes = {}
    for name in output_names:
        head, track = name.split("/", 1)
        head_classes.setdefault(head, []).append(track)

    # make predictions: one array per output
    sequence = "A" * 1048576
    predictions = crested.tl.predict(sequence, model)
    for output_name, prediction in zip(model.output_names, predictions):
        print(output_name, prediction.shape)
