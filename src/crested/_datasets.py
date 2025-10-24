from __future__ import annotations

import os

import pooch

_datasets = None


def _get_dataset_index():
    """
    Set up the pooch dataset registry.

    To add a self-hosted dataset:
    - Upload a tarball of topics/bigwigs (and a consensus peak file if not yet there)
    - Get sha256 file hashes of both files.
    - At `pooch.create()` argument `registry`: Add the file's relative path as a key (e.g. 'data/mouse_biccn/beds.tar.gz'), with the file hash as value.
    - At function `get_dataset()` below: add the user-facing name (e.g. 'mouse_cortex_topics') to the documentation
        and to dataset_mapping, mapped to the data and consensus peaks files
        (e.g. `'mouse_cortex_topics': ("data/mouse_biccn/beds.tar.gz", "data/mouse_biccn/consensus_peaks_biccn.bed")` )

    To add external datasets:
    - Get sha256 file hashes of files.
    - At `pooch.create()` argument `registry`: Add the file's name as key (e.g. '10x_pbmc_topics_beds.tar.gz'), with the file hash as value.
    - at `pooch.create()` argument `url`: Add the file's name as key again, with the URL as value.
    - At function `get_dataset()` below: add the user-facing name (e.g. '10x_pbmc_topics') to the documentation
        and to dataset_mapping, mapped to the data and consensus peaks files
        (e.g. `'10x_pbmc_topics': ("10x_pbmc_topics_beds.tar.gz", "10x_pbmc_topics_beds.tar.gz")` )
    """
    # Set datasets variable as global
    global _datasets
    # If it doesn't exist yet, load in
    if _datasets is None:
        _datasets = pooch.create(
            path=pooch.os_cache("crested"),
            base_url="https://resources.aertslab.org/CREsted/",
            env="CRESTED_DATA_DIR",
            # Datasets that can be downloaded.
            ## For datasets at the CREsted resources, a relative URL as name is sufficient. For an external dataset, add the absolute URL to `urls = {'name': 'https://thefullurl.eu/yourfile.txt'}`.
            registry={
                # BICCN datasets
                "data/mouse_biccn/bigwigs_coverage.tar.gz": "sha256:738504d26b864de10804978b4b47196094996174349c1140e44824ce6d0349ba",
                "data/mouse_biccn/bigwigs_cut_sites.tar.gz": "sha256:0a9871d75a1c03fdfe353171d6cebec307cd96d054594d28b95ee451f948e7f0",
                "data/mouse_biccn/beds.tar.gz": "sha256:0a2c42505eaced286a731c15a10ec63b680a90333a47de4a3c86a112e4e0f8df",
                "data/mouse_biccn/consensus_peaks_biccn.bed": "sha256:83ce5a58aee11c40d7a1e11b603ceb34012e0e4b91eea0953eb37a943707a1e5",
                # Melanoma datasets
                # Fly datasets
                # Motif databases
                "motif_db/motif_db.meme": "sha256:1667eaf9ca2abb37fa21b541faa9e1676690b58a1206078255a9d7e389731dbc",
                "motif_db/motif_tf_collection.tsv": "sha256:b91c4d4483ff92cef394d109899fa6f6bc5f3c474629e65d83b042b58d212f91",
                # Models
                "models/deepflybrain.tar.gz": "sha256:65ed3f8c2d15216b9b47a06165d330d2f7b9dbae1b8ba9c06c095c9a652e4a23",
                "models/biccn.tar.gz": "sha256:e59887c3010c740cd34fbb5d0b5853cd254d7e53ea66783c1fcd42405ed6ea0f",
                "models/deepbiccn2.tar.gz": "sha256:277f8321955e55f4c4d80e2ec202194af1873f4b782771324b3411017755cc18",
                "models/deepccl.tar.gz": "sha256:2bf0b379572b58b5a6512801ac133cb728621204ce19ad3b361b2e5e705a4ade",
                "models/deepchickenbrain1.tar.gz": "sha256:f87c58b7524cff8543bc817678408d01ca9c0349497f4b1e391c1b172046352f",
                "models/deepchickenbrain2.tar.gz": "sha256:e61310256f96a28dc450d2440edd708e7eaf6b3ee58c53a76aa3e708e8cd39fa",
                "models/deepglioma.tar.gz": "sha256:925931aadaf723a590843e37809afdf2757d2abe84448fd1bc674aedc609f18c",
                "models/deephumanbrain.tar.gz": "sha256:8f1e3d71208a587f64b8d20d24851bda9b0779dc6dca623e9287d1506a357c06",
                "models/deephumancortex1.tar.gz": "sha256:f69b75ea5839c666570e634ffa09646239be3fe2ab593b6f076589a36e5abea5",
                "models/deephumancortex2.tar.gz": "sha256:dfd93febfdf4ba44ae6f65a13ed674e7b8d6d6f124ce4abd34c102bc04de4530",
                "models/deepliver_accessibility.tar.gz": "sha256:8a3ddcfa29effa9e979eae769fbfc6a362de90197434a630015dccca9fecb34c",
                "models/deepliver_activity.tar.gz": "sha256:e5e12200ce90b9f0b56653a35294c37f0f97e4d2e951e79c701355ef62b1ea6c",
                "models/deepliver_zonation.tar.gz": "sha256:d7227b7ecf0bf7703e775713cef38466df1ee5e7a63a8ccf22a5da545040708c",
                "models/deeppbmc.tar.gz": "sha256:aac4f08d55b2bab595f95f9ae59ee90cadf383e190968084a4ac4cdb0e9f5589",
                "models/deepmel1.tar.gz": "sha256:f8ffd3362fb7ac3d946b30f4f1bc1eaa7da4c8ba826d1e206d69914e2a969c23",
                "models/deepmel2.tar.gz": "sha256:df6ad28fe9bf892810afd34ae5c0318022bef6eec2f41b4fea66654ac51f9e47",
                "models/deepmel2_gabpa.tar.gz": "sha256:856176c7755e7e93aadb532504f5841422ee00f3b431e7bc136ea498b586b303",
                "models/deepmousebrain1.tar.gz": "sha256:af2c2487aa7b49b0d487d199b5e4efddb147578dc075682dd2b4bd48ad6ffaa5",
                "models/deepmousebrain2.tar.gz": "sha256:4c7dd4649f642b6ec120fb324f479093644d622826d4a3a7d3ee9e3c2bb8bde6",
                "models/deepmousebrain3.tar.gz": "sha256:a6160a87bb38eb64e7c39b0bda7f147b5c23bd41c82f72909f3372967aae7bdc",
                "models/deepzebrafish.tar.gz": "sha256:a4aa540afc8fc372a9ed6a614ac2dc3f47f4675ece44eeb9c127800be20d6c26",
                "models/enformer_human.tar.gz": "628f67f540304d4d0e143176dc824ed72b3413f78f2fa2efe5d4f0ab51ea1bcc",
                "models/enformer_mouse.tar.gz": "77296be9f16bcf81b9c9d3ae2bba61d7ae99e01e77a05eefc1fa316ef5eb6e31",
                "models/borzoi_biccn.tar.gz": "sha256:8062c50622e297053796af5359757c56e59f5188c5bcca7d4c3a002b937b122c",
                "models/borzoi_human_rep0.tar.gz": "c78f0e15a4962a4ccb699c29eab71114fd92c6d08d435f28ea15c2f116ec8ff1",
                "models/borzoi_human_rep1.tar.gz": "f5eefd9bddcdee02f00a3a0cd758174b58ea0b269f4d04b8d9e1e0e67ec6b9bd",
                "models/borzoi_human_rep2.tar.gz": "48bc4dffc8e271eae7572fd63e8ce708744d8fe1f67a50c5422c1206e54d25db",
                "models/borzoi_human_rep3.tar.gz": "a3241116ad78dd91e11d8093a08e9146242f0034d5bfc019bea7c16d633a7ba8",
                "models/borzoi_mouse_rep0.tar.gz": "9ef37a00a5aaaab549c70e7d299a5a908f68121191fdec25893ee8416e40889a",
                "models/borzoi_mouse_rep1.tar.gz": "b931b14ee0d5a340f7c39b317ab0be099b45e5f4075511dc5e56cd0e9af7f857",
                "models/borzoi_mouse_rep2.tar.gz": "0b4183a2751975de8ee2070fbf9bad40cbbfb17fe6784299920f5a2723cc07b5",
                "models/borzoi_mouse_rep3.tar.gz": "a18fd295ec356ada86c6681b173ffe25f320447884129f5d3c15be80755423e4",
                "models/embryo_10x.tar.gz": "sha256:3de62e6913ea491ebdad4bc2ef4e9f404250531093b29e67b5a1fa907f5cf41b",
                "models/embryo_hydrop.tar.gz": "sha256:9036ef2f18ab894016c8adcd3bba5f1985b74bd2432f4e9b07e6128e43833636",
                "models/mousecortex_hydrop.tar.gz": "sha256:a850edcd6f9cabd0efb7ea66f48287c1c05ff7c3466363259dfb9229a076c53c",
            },
        )
    return _datasets


def get_dataset(dataset: str):
    """
    Fetch an example dataset. This function retrieves the dataset of bigwig or bed files and associated region file, downloading if not already cached, and returns the paths to the dataset.

    Provided examples:
      - 'mouse_cortex_bed': the BICCN mouse cortex snATAC-seq dataset, processed as BED files per topic. For use in topic classification.
      - 'mouse_cortex_bigwig_coverage': the BICCN mouse cortex snATAC-seq dataset, processed as pseudobulked bigWig coverage tracks per cell type. For use in peak regression.
      - 'mouse_cortex_bigwig_cut_sites': the BICCN mouse cortex snATAC-seq dataset, processed as pseudobulked bigWig cut site tracks per cell type. For use in peak regression.

    These two paths can be passed to :func:`crested.import_bigwigs()` / :func:`crested.import_beds()`.

    Note
    ----
    The cache location can be changed by setting environment variable $CRESTED_DATA_DIR.

    Parameters
    ----------
    dataset
        The name of the dataset to fetch. Available options:
          - 'mouse_cortex_bed'
          - 'mouse_cortex_bigwig_cut_sites'
          - 'mouse_cortex_bigwig_coverage'
          - 'mouse_cortex_bigwig' (deprecated, same as 'mouse_cortex_bigwig_coverage')

    Returns
    -------
    A tuple consisting of the BED/bigWig-containing directory and the consensus regions file.

    Example
    -------
    >>> beds_folder, regions_file = crested.get_dataset("mouse_cortex_bed")
    >>> adata = crested.import_beds(beds_folder=beds_folder, regions_file=regions_file)
    """
    # Mapping: "user_facing_name": ("tarball_name_in_registry.tar.gz", "cpeaks_name_in_registry.bed")
    dataset_mapping = {
        "mouse_cortex_bed": (
            "data/mouse_biccn/beds.tar.gz",
            "data/mouse_biccn/consensus_peaks_biccn.bed",
        ),
        "mouse_cortex_bigwig_coverage": (
            "data/mouse_biccn/bigwigs_coverage.tar.gz",
            "data/mouse_biccn/consensus_peaks_biccn.bed",
        ),
        "mouse_cortex_bigwig_cut_sites": (
            "data/mouse_biccn/bigwigs_cut_sites.tar.gz",
            "data/mouse_biccn/consensus_peaks_biccn.bed",
        ),
        "mouse_cortex_bigwig": (
            "data/mouse_biccn/bigwigs_coverage.tar.gz",
            "data/mouse_biccn/consensus_peaks_biccn.bed",
        ),  # Deprecated
    }
    assert (
        dataset in dataset_mapping
    ), f"Dataset {dataset} is not recognised. Available datasets: {tuple(dataset_mapping.keys())}"

    targets_url, cregions_url = dataset_mapping[dataset]
    targets_paths = _get_dataset_index().fetch(
        targets_url, processor=pooch.Untar(), progressbar=True
    )
    cregions_path = _get_dataset_index().fetch(cregions_url, progressbar=True)
    targets_dir = os.path.dirname(targets_paths[0])
    return targets_dir, cregions_path


def get_motif_db():
    """
    Fetch the motif database. This function retrieves the Aerts lab motif database for use in motif analysis, downloading if not already cached, and returns the paths to the dataset.

    These two paths can be passed to :func:`crested.import_bigwigs()` / :func:`crested.import_beds()`.

    Note
    ----
    The cache location can be changed by setting environment variable $CRESTED_DATA_DIR.

    Returns
    -------
    A tuple consisting of the motif db .meme file path and the transcription factor info .tsv file path.

    Example
    -------
    >>> motif_db_path, motif_tf_collection_path = crested.get_motif_db()
    """
    motif_db_path = _get_dataset_index().fetch(
        "motif_db/motif_db.meme", progressbar=True
    )
    motif_collection_path = _get_dataset_index().fetch(
        "motif_db/motif_tf_collection.tsv", progressbar=True
    )
    return motif_db_path, motif_collection_path


def get_model(model: str) -> tuple[str, list[str]]:
    """
    Fetch a model.

    This function retrieves the model files, downloading if not already cached, and returns the paths to the model and a list of output classnames.
    The model folder contains the model.keras file and the output classnames file (.tsv).

    Note
    ----
    The cache location can be changed by setting environment variable $CRESTED_DATA_DIR.

    Parameters
    ----------
    model
        The name of the model to fetch. Available options:
          - 'DeepBICCN'
          - 'DeepBICCN2'
          - 'DeepCCL'
          - 'DeepChickenBrain1'
          - 'DeepChickenBrain2'
          - 'DeepFlyBrain'
          - 'DeepGlioma'
          - 'DeepHumanBrain'
          - 'DeepHumanCortex1'
          - 'DeepHumanCortex2'
          - 'DeepLiver_accessibility'
          - 'DeepLiver_activity'
          - 'DeepLiver_zonation'
          - 'DeepMEL1'
          - 'DeepMEL2'
          - 'DeepMEL2_gabpa'
          - 'DeepMouseBrain1'
          - 'DeepMouseBrain2'
          - 'DeepMouseBrain3'
          - 'DeepPBMC'
          - 'DeepZebraFish'
          - 'Enformer_human'
          - 'Enformer_mouse'
          - 'BorzoiBICCN'
          - 'Borzoi_human_rep[0-3]'
          - 'Borzoi_mouse_rep[0-3]'
          - 'Embryo10x"
          - 'EmbryoHydrop'
          - 'MouseCortexHydrop'

    Returns
    -------
    A tuple consisting of the .keras model file path and a list of output classnames.

    Example
    -------
    >>> model_file, output_names = crested.get_model("DeepFlyBrain")
    """
    # Mapping: "user_facing_name": ("model_folder_in_registry.tar.gz")
    model_mapping = {
        "DeepBICCN": ("models/biccn.tar.gz"),
        "DeepBICCN2": ("models/deepbiccn2.tar.gz"),
        "DeepCCL": ("models/deepccl.tar.gz"),
        "DeepChickenBrain1": ("models/deepchickenbrain1.tar.gz"),
        "DeepChickenBrain2": ("models/deepchickenbrain2.tar.gz"),
        "DeepFlyBrain": ("models/deepflybrain.tar.gz"),
        "DeepGlioma": ("models/deepglioma.tar.gz"),
        "DeepHumanBrain": ("models/deephumanbrain.tar.gz"),
        "DeepHumanCortex1": ("models/deephumancortex1.tar.gz"),
        "DeepHumanCortex2": ("models/deephumancortex2.tar.gz"),
        "DeepLiver_accessibility": ("models/deepliver_accessibility.tar.gz"),
        "DeepLiver_activity": ("models/deepliver_activity.tar.gz"),
        "DeepLiver_zonation": ("models/deepliver_zonation.tar.gz"),
        "DeepPBMC": ("models/deeppbmc.tar.gz"),
        "DeepMEL1": ("models/deepmel1.tar.gz"),
        "DeepMEL2": ("models/deepmel2.tar.gz"),
        "DeepMEL2_gabpa": ("models/deepmel2_gabpa.tar.gz"),
        "DeepMouseBrain1": ("models/deepmousebrain1.tar.gz"),
        "DeepMouseBrain2": ("models/deepmousebrain2.tar.gz"),
        "DeepMouseBrain3": ("models/deepmousebrain3.tar.gz"),
        "DeepZebraFish": ("models/deepzebrafish.tar.gz"),
        "Enformer_human": ("models/enformer_human.tar.gz"),
        "Enformer_mouse": ("models/enformer_mouse.tar.gz"),
        "BorzoiBICCN": ("models/borzoi_biccn.tar.gz"),
        "Borzoi_human_rep0": ("models/borzoi_human_rep0.tar.gz"),
        "Borzoi_human_rep1": ("models/borzoi_human_rep1.tar.gz"),
        "Borzoi_human_rep2": ("models/borzoi_human_rep2.tar.gz"),
        "Borzoi_human_rep3": ("models/borzoi_human_rep3.tar.gz"),
        "Borzoi_mouse_rep0": ("models/borzoi_mouse_rep0.tar.gz"),
        "Borzoi_mouse_rep1": ("models/borzoi_mouse_rep1.tar.gz"),
        "Borzoi_mouse_rep2": ("models/borzoi_mouse_rep2.tar.gz"),
        "Borzoi_mouse_rep3": ("models/borzoi_mouse_rep3.tar.gz"),
        "Embryo10x": ("models/embryo_10x.tar.gz"),
        "EmbryoHydrop": ("models/embryo_hydrop.tar.gz"),
        "MouseCortexHydrop": ("models/mousecortex_hydrop.tar.gz"),
    }
    assert (
        model in model_mapping
    ), f"Model {model} is not recognised. Available models: {tuple(model_mapping.keys())}"

    model_folder = model_mapping[model]
    model_folder_paths = _get_dataset_index().fetch(
        model_folder, processor=pooch.Untar(), progressbar=True
    )
    for path in model_folder_paths:
        if path.endswith(".keras"):
            model_file = path
        elif path.endswith("_output_classes.tsv"):
            model_output_names = path
            with open(model_output_names) as f:
                model_output_classes = f.read().splitlines()
        else:
            raise ValueError(f"Unexpected file found in model folder: {path}")
    return model_file, model_output_classes
