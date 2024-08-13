import os

import pooch

_datasets = None


def _get_dataset_index():
    """Function that sets up the pooch dataset registry.

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
                "data/mouse_biccn/bigwigs.tar.gz": "sha256:738504d26b864de10804978b4b47196094996174349c1140e44824ce6d0349ba",
                "data/mouse_biccn/beds.tar.gz": "sha256:0a2c42505eaced286a731c15a10ec63b680a90333a47de4a3c86a112e4e0f8df",
                "data/mouse_biccn/consensus_peaks_biccn.bed": "sha256:83ce5a58aee11c40d7a1e11b603ceb34012e0e4b91eea0953eb37a943707a1e5",
                # Melanoma datasets
                # Fly datasets
                # Models
                # Motif databases
                "motif_db/motif_db.meme": "sha256:31d3fa1117e752b0d3076a73b278b59bb4a056d744401e9d5861310d03186cfd",
                "motif_db/motif_tf_collection.tsv": "sha256:438933d41033b274035ec0bcf66bdafb1de2f22a1eb142800d1e76b6729e3438",
            },
        )
    return _datasets


def get_dataset(dataset: str):
    """
    Fetch an example dataset. This function retrieves the dataset of bigwig or bed files and associated region file, downloading if not already cached, and returns the paths to the dataset.

    Provided examples:
    - 'mouse_cortex_bed': the BICCN mouse cortex snATAC-seq dataset, processed as BED files per topic. For use in topic classification.
    - 'mouse_cortex_bigwig': the BICCN mouse cortex snATAC-seq dataset, processed as pseudobulked bigWig tracks per cell type. For use in peak regression.

    These two paths can be passed to :func:`crested.import_bigwigs()` / :func:`crested.import_beds()`.

    Note
    ----
    The cache location can be changed by setting environment variable $CRESTED_DATA_DIR.

    Parameters
    ----------
    dataset
        The name of the dataset to fetch.
        Options: 'mouse_cortex_bed', 'mouse_cortex_bigwig'

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
        "mouse_cortex_bigwig": (
            "data/mouse_biccn/bigwigs.tar.gz",
            "data/mouse_biccn/consensus_peaks_biccn.bed",
        ),
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
