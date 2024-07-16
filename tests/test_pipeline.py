"""Full analysis pipeline tests."""

import os

import genomepy
import keras

import crested

from ._utils import create_anndata_with_regions

REGIONS = [
    "chr1:194208032-194208532",
    "chr1:92202766-92203266",
    "chr1:92298990-92299490",
    "chr1:3406052-3406552",
    "chr1:183669567-183670067",
    "chr1:109912183-109912683",
    "chr1:92210697-92211197",
    "chr1:59100954-59101454",
    "chr1:84634055-84634555",
    "chr1:48792527-48793027",
]


def test_peak_regression():
    adata = create_anndata_with_regions(REGIONS)

    crested.pp.change_regions_width(adata, width=600)
    crested.pp.train_val_test_split(
        adata, strategy="region", val_size=0.1, test_size=0.1
    )
    if not os.path.exists("tests/data/genomes/hg38.fa"):
        genomepy.install_genome(
            "hg38", annotation=False, provider="UCSC", genomes_dir="tests/data/genomes"
        )
    print(adata)
    print(adata.var)
    print(adata.var_names)

    datamodule = crested.tl.data.AnnDataModule(
        adata,
        genome_file="tests/data/genomes/hg38/hg38.fa",
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
    trainer.test()
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
    trainer.enhancer_design_in_silico_evolution(
        target_class="cell_1", n_sequences=1, n_mutations=1
    )
