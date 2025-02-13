"""Full analysis pipeline tests."""

import os

import genomepy
import keras

import crested


def test_peak_regression(adata):
    crested.pp.change_regions_width(adata, width=600)
    crested.pp.train_val_test_split(
        adata, strategy="region", val_size=0.1, test_size=0.1
    )
    if not os.path.exists("tests/data/genomes/hg38.fa"):
        genomepy.install_genome(
            "hg38", annotation=False, provider="UCSC", genomes_dir="tests/data/genomes"
        )

    if os.path.exists("tests/data/test_pipeline"):
        import shutil

        shutil.rmtree("tests/data/test_pipeline")

    datamodule = crested.tl.data.AnnDataModule(
        adata,
        genome="tests/data/genomes/hg38/hg38.fa",
        batch_size=2,
        always_reverse_complement=True,
        deterministic_shift=True,
        max_stochastic_shift=3,
    )
    model_architecture = crested.tl.zoo.simple_convnet(
        seq_len=600,
        num_classes=10,
        first_filters=5,
        num_conv_blocks=1,
        num_dense_blocks=1,
        filters=5,
    )
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    loss = crested.tl.losses.CosineMSELogLoss(max_weight=1.5)
    metrics = [
        keras.metrics.MeanSquaredError(),
    ]
    config = crested.tl.TaskConfig(optimizer, loss, metrics)
    trainer = crested.tl.Crested(
        data=datamodule,
        model=model_architecture,
        config=config,
        project_name="tests/data/test_pipeline",
        run_name="test_peak_regression",
        logger=None,
    )
    trainer.fit(epochs=1)

    trainer.load_model(
        "tests/data/test_pipeline/test_peak_regression/checkpoints/01.keras",
        compile=True,
    )

    test_metrics = trainer.test(return_metrics=True)
    assert isinstance(test_metrics, dict)
    trainer.predict(adata, model_name="01")

    trainer.predict_regions(region_idx=["chr1:1000-1600", "chr2:2000-2600"])
    sequence = "ATGCGTacGT" * 60
    trainer.predict_sequence(sequence=sequence)

    scores, one_hot_encoded_sequences = trainer.calculate_contribution_scores_sequence(
        sequence, class_names=["cell_1", "cell_2"], method="integrated_grad"
    )

    assert scores.shape == (1, 2, 600, 4)
    assert one_hot_encoded_sequences.shape == (1, 600, 4)

    scores, one_hot_encoded_sequences = trainer.calculate_contribution_scores_regions(
        region_idx=["chr1:1000-1600", "chr2:2000-2600"],
        class_names=[],
        method="integrated_grad",
    )
    intermediate, seqs = trainer.enhancer_design_in_silico_evolution(
        target_class="cell_1", n_sequences=1, n_mutations=2, return_intermediate=True
    )

    scores, seqs = trainer.calculate_contribution_scores_enhancer_design(
        intermediate, class_names=["cell_1"], method="expected_integrated_grad"
    )
    crested.pl.patterns.enhancer_design_steps_contribution_scores(
        intermediate, scores, seqs, show=False
    )
    # test continue training
    trainer_2 = crested.tl.Crested(
        data=datamodule,
        model=model_architecture,
        config=config,
        project_name="tests/data/test_pipeline",
        run_name="test_peak_regression",
        logger=None,
    )
    trainer_2.fit(epochs=2)
    assert os.path.exists(
        "tests/data/test_pipeline/test_peak_regression/checkpoints/02.keras"
    )
