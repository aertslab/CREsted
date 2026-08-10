"""Data loading objects that allow you to train on batches of entries from your AnnData."""

from . import utils
from ._anndatawrapper import AnnDataWrapper, MultiAnnDataWrapper
from ._old._anndatamodule import AnnDataModule
