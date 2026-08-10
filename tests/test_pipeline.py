"""Full analysis pipeline tests."""

import os

import keras

import crested


def test_peak_regression(adata, genome):
    crested.pp.change_regions_width(adata, width=600)
    crested.pp.train_val_test_split(
        adata, strategy="region", val_size=0.1, test_size=0.1
    )

    if os.path.exists("tests/data/test_pipeline"):
        import shutil

        shutil.rmtree("tests/data/test_pipeline")

    datamodule = crested.tl.data.AnnDataModule(
        adata,
        genome=genome,
        batch_size=2,
        always_reverse_complement=True,
        deterministic_shift=True,
        max_stochastic_shift=3,
    )
    model = crested.tl.zoo.simple_convnet(
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
        model=model,
        config=config,
        project_name="tests/data/test_pipeline",
        run_name="test_peak_regression",
        logger=None,
    )
    trainer.fit(epochs=1)

    model_path = "tests/data/test_pipeline/test_peak_regression/checkpoints/01.keras"
    trainer.load_model(
        model_path,
        compile=True,
    )

    test_metrics = trainer.test(return_metrics=True)
    assert isinstance(test_metrics, dict)

    model = keras.models.load_model(model_path, compile=False)
    crested.tl.predict(adata, model, genome=genome)
    crested.tl.predict(["chr1:1000-1600", "chr2:2000-2600"], model, genome=genome)
    sequence = "ATGCGTacGT" * 60
    crested.tl.predict(sequence, model)

    scores, one_hot_encoded_sequences = crested.tl.contribution_scores(
        sequence, target_idx=[1, 2], model=model, method="integrated_grad"
    )
    assert scores.shape == (1, 2, 600, 4)
    assert one_hot_encoded_sequences.shape == (1, 600, 4)

    scores, one_hot_encoded_sequences = crested.tl.contribution_scores(
        ["chr1:1000-1600", "chr2:2000-2600"], target_idx=[1, 2], model=model, method="expected_integrated_grad", genome=genome
    )
    intermediate, seqs = crested.tl.design.in_silico_evolution(
        n_mutations=2, target=1, model=model, n_sequences=1, return_intermediate=True
    )

    intermediate_seqs = crested.tl.design.derive_intermediate_sequences(intermediate)
    scores, seqs = crested.tl.contribution_scores(
        intermediate_seqs[0],
        model=model,
        target_idx=1,
        method="integrated_grad"
    )
    crested.pl.design.step_contribution_scores(
        intermediate, scores, seqs, show=False
    )

    # test continue training
    trainer_2 = crested.tl.Crested(
        data=datamodule,
        model=model,
        config=config,
        project_name="tests/data/test_pipeline",
        run_name="test_peak_regression",
        logger=None,
    )
    trainer_2.fit(epochs=2)
    assert os.path.exists(
        "tests/data/test_pipeline/test_peak_regression/checkpoints/02.keras"
    )
