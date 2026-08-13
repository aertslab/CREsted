import sys

import numpy as np
import pybigtools
import pytest
from anndata import AnnData

import crested


def test_package_has_version():
    assert crested.__version__ is not None


def test_import_beds_shape():
    ann_data = crested.import_beds(
        beds_folder="tests/data/test_topics",
        regions_file="tests/data/test.regions.bed",
    )
    # Test type
    assert isinstance(ann_data, AnnData)

    # Test shape
    expected_number_of_topics = 3
    expected_number_of_peaks = 23186

    assert ann_data.shape == (expected_number_of_topics, expected_number_of_peaks)

    # Test columns
    assert "file_path" in ann_data.obs.columns
    assert "n_open_regions" in ann_data.obs.columns
    assert "n_classes" in ann_data.var.columns


def test_import_beds_classes_subset():
    ann_data = crested.import_beds(
        beds_folder="tests/data/test_topics",
        regions_file="tests/data/test.regions.bed",
        classes_subset=["Topic_1", "Topic_2"],
    )
    assert ann_data.shape[0] == 2


def test_import_beds_invalid_files():
    with pytest.raises(FileNotFoundError):
        crested.import_beds(beds_folder="invalid_folder", regions_file="invalid_file")


def test_import_beds_compression():
    ann_data_c = crested.import_beds(
        beds_folder="tests/data/test_topics",
        regions_file="tests/data/test.regions.bed",
        compress=True,
    )
    assert ann_data_c.X.getformat() == "csr"
    assert ann_data_c.X.shape == (3, 23186)

    ann_data = crested.import_beds(
        beds_folder="tests/data/test_topics",
        regions_file="tests/data/test.regions.bed",
        compress=False,
    )
    assert isinstance(ann_data.X, np.ndarray)
    assert ann_data.X.shape == (3, 23186)

    assert sys.getsizeof(ann_data_c.X) < sys.getsizeof(ann_data.X)


def test_import_beds_chromsizes():
    ann_data = crested.import_beds(
        beds_folder="tests/data/test_topics",
        regions_file="tests/data/test.regions.bed",
        chromsizes_file="tests/data/test.chrom.sizes",
        compress=True,
    )
    expected_removed_regions = ["chr19:60789836-60790336"]
    for region in expected_removed_regions:
        assert region not in list(ann_data.var.index)


def test_import_bigwigs_type():
    ann_data = crested.import_bigwigs(
        bigwigs_folder="tests/data/test_bigwigs",
        regions_file="tests/data/test_bigwigs/consensus_peaks_subset.bed",
    )
    # Test type
    assert isinstance(ann_data, AnnData)


def test_import_bigwigs_invalid_files():
    with pytest.raises(FileNotFoundError):
        crested.import_bigwigs(
            bigwigs_folder="invalid_folder", regions_file="invalid_file"
        )


def test_import_bigwigs_shape():
    ann_data = crested.import_bigwigs(
        bigwigs_folder="tests/data/test_bigwigs",
        regions_file="tests/data/test_bigwigs/consensus_peaks_subset.bed",
    )
    # Test shape
    expected_number_of_bigwigs = 2
    expected_number_of_peaks = 5000

    assert ann_data.shape == (expected_number_of_bigwigs, expected_number_of_peaks)


def test_import_bigwigs_columns():
    ann_data = crested.import_bigwigs(
        bigwigs_folder="tests/data/test_bigwigs",
        regions_file="tests/data/test_bigwigs/consensus_peaks_subset.bed",
    )
    # Test columns in .obs
    assert "file_path" in ann_data.obs.columns

    # Test columns in .var (chromosome regions)
    assert "chr" in ann_data.var.columns
    assert "start" in ann_data.var.columns
    assert "end" in ann_data.var.columns


def test_import_bigwigs_full_chrom_mismatch_error(tmp_path):
    # All regions on a chromosome absent from the bigwig (only has chr1) should raise
    bad_bed = tmp_path / "bad.bed"
    with open(bad_bed, "w") as f:
        for i in range(5):
            f.write(f"chr2\t{1000 + i * 600}\t{1000 + i * 600 + 500}\tregion_{i}\n")

    with pytest.raises(ValueError, match="All read-in values are NaNs"):
        crested.import_bigwigs(
            bigwigs_folder=["tests/data/test_bigwigs/lamp5_sample.bw"],
            regions_file=str(bad_bed),
        )


def test_import_bigwigs_partial_chrom_mismatch_warning(tmp_path, capfd):
    # A minority of regions on a chromosome absent from the bigwig should warn but still return values
    mixed_bed = tmp_path / "mixed.bed"
    with open(mixed_bed, "w") as f:
        f.write("chr1\t3094805\t3095305\tr0\n")
        for i in range(5):
            f.write(f"chr2\t{1000 + i * 600}\t{1000 + i * 600 + 500}\tregion_{i}\n")

    ann_data = crested.import_bigwigs(
        bigwigs_folder=["tests/data/test_bigwigs/lamp5_sample.bw"],
        regions_file=str(mixed_bed),
    )
    captured = capfd.readouterr()
    assert "did not match chromosomes" in captured.err + captured.out
    # All 6 regions are kept, but the 5 mismatched (chr2) ones are NaN-filled
    assert ann_data.shape == (1, 6)
    assert np.isnan(ann_data.X).sum() == 5


def test_import_beds_chromsizes_mismatch_error(tmp_path):
    # A chromsizes file with no chromosomes in common with the regions file should raise
    bad_chromsizes = tmp_path / "bad.chrom.sizes"
    with open(bad_chromsizes, "w") as f:
        f.write("chrZZZ\t1000000\n")

    with pytest.raises(ValueError, match="fell within known chromosomes"):
        crested.import_beds(
            beds_folder="tests/data/test_topics",
            regions_file="tests/data/test.regions.bed",
            chromsizes_file=str(bad_chromsizes),
        )


def test_import_bigwigs_negative_values_warning(tmp_path, capfd):
    neg_bw_path = tmp_path / "neg.bw"
    writer = pybigtools.open(str(neg_bw_path), "w")
    writer.write(
        {"chr1": 195471971},
        [("chr1", 3094805, 3095305, -1.5), ("chr1", 3095470, 3095970, 2.0)],
    )

    ann_data = crested.import_bigwigs(
        bigwigs_folder=[str(neg_bw_path)],
        regions_file="tests/data/test_bigwigs/consensus_peaks_subset.bed",
    )
    captured = capfd.readouterr()
    assert "contain negative values" in captured.err + captured.out
    assert isinstance(ann_data, AnnData)


def test_import_bigwigs_list_input():
    ann_data = crested.import_bigwigs(
        bigwigs_folder=[
            "tests/data/test_bigwigs/lamp5_sample.bw",
            "tests/data/test_bigwigs/vip_sample.bigwig",
        ],
        regions_file="tests/data/test_bigwigs/consensus_peaks_subset.bed",
    )
    assert list(ann_data.obs.index) == ["lamp5_sample", "vip_sample"]
    assert ann_data.shape == (2, 5000)


def test_import_bigwigs_dict_input():
    ann_data = crested.import_bigwigs(
        bigwigs_folder={
            "MyLamp5": "tests/data/test_bigwigs/lamp5_sample.bw",
            "MyVip": "tests/data/test_bigwigs/vip_sample.bigwig",
        },
        regions_file="tests/data/test_bigwigs/consensus_peaks_subset.bed",
    )
    assert list(ann_data.obs.index) == ["MyLamp5", "MyVip"]
    assert ann_data.shape == (2, 5000)


def test_import_beds_list_input():
    ann_data = crested.import_beds(
        beds_folder=[
            "tests/data/test_topics/Topic_1.bed",
            "tests/data/test_topics/Topic_2.bed",
        ],
        regions_file="tests/data/test.regions.bed",
    )
    assert list(ann_data.obs.index) == ["Topic_1", "Topic_2"]
    assert ann_data.shape[0] == 2


def test_import_beds_dict_input():
    ann_data = crested.import_beds(
        beds_folder={
            "MyTopic1": "tests/data/test_topics/Topic_1.bed",
            "MyTopic2": "tests/data/test_topics/Topic_2.bed",
        },
        regions_file="tests/data/test.regions.bed",
    )
    assert list(ann_data.obs.index) == ["MyTopic1", "MyTopic2"]
    assert ann_data.shape[0] == 2


def test_import_beds_list_input_classes_subset_rejected():
    with pytest.raises(ValueError, match="classes_subset only works"):
        crested.import_beds(
            beds_folder=["tests/data/test_topics/Topic_1.bed"],
            regions_file="tests/data/test.regions.bed",
            classes_subset=["Topic_1"],
        )
