"""Defines the `Dataset` class and its configuration `DatasetConfig`."""

import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional, TypedDict

import h5py
import torch
import scipy.signal
import numpy as np
import numpy.typing as npt

from .util.path import resolve_path


_logger = logging.getLogger(__name__)


class DatasetConfig(TypedDict):
    """Configuration for creating a `Dataset`."""

    hdf5_filename: str | os.PathLike
    """Path to the HDF5 file."""

    predicate: Optional[str | Callable[[h5py.File], np.ndarray]]
    """If given, a lambda function that takes an HDF5 file and returns a boolean mask
    indicating which items should be included in the dataset."""

    signal_dtype: npt.DTypeLike
    """The NumPy dtype for ECG signals."""

    in_leads: Sequence[int]
    """Indices of the leads used as `input`."""

    out_leads: Sequence[int]
    """Indices of the leads used as `target`."""

    filter_type: str | Callable
    """A function in `scipy.signal` used to create an SOS for an IIR filter."""

    filter_args: Mapping[str, Any]
    """Arguments passed to `filter_type`."""

    mean_normalization: bool
    """Whether to apply mean normalization to the signals."""

    feature_scaling: bool
    """Whether to apply feature scaling to the signals."""

    include_original_signal: bool
    """Whether to include the original signals as `original_signal`."""

    include_filtered_signal: bool
    """Whether to include the signals after filtering (and other preprocessing if
    specified) as `filtered_signal`."""

    include_labels: Mapping[str, str | Sequence[str]]
    """The labels to include. Each key is a label to be used in output dictionaries, and
    each value is a label in the HDF5 file or a sequence of such labels."""


class Dataset(torch.utils.data.Dataset):
    """A dataset of ECG signals and their labels."""

    def __init__(self, config: DatasetConfig) -> None:
        """Creates a dataset from a configuration dictionary."""
        _logger.debug("Create dataset from config: %s", config)
        self._config = config
        self._hdf5_file = h5py.File(
            resolve_path(config["hdf5_filename"], relative_to="src/datasets")
        )
        self._filter_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None

        total_num_items = len(self._hdf5_file["signal"])
        if config["predicate"] is None:
            self._lookup_table = np.arange(total_num_items)
            _logger.debug(
                "No predicate applied to dataset, %d items in total", total_num_items
            )
        else:
            predicate = config["predicate"]
            if isinstance(predicate, str):
                predicate = eval(predicate, {"np": np})  # pylint: disable=eval-used
            (self._lookup_table,) = np.nonzero(predicate(self._hdf5_file))
            _logger.debug(
                "Predicate applied to dataset, %d items out of %d remained",
                len(self._lookup_table),
                total_num_items,
            )

        filter_type = config["filter_type"]
        if isinstance(filter_type, str):
            filter_type = getattr(scipy.signal, filter_type)
        self._filter_sos = filter_type(
            **{"output": "sos", "fs": self.sampling_rate, **config["filter_args"]}
        ).astype(config["signal_dtype"], copy=False)

    @property
    def in_leads(self) -> list[int]:
        """Indices of the leads used as `input`."""
        return list(self._config["in_leads"])

    @property
    def out_leads(self) -> list[int]:
        """Indices of the leads used as `target`."""
        return list(self._config["out_leads"])

    @property
    def signal_length(self) -> int:
        """The number of samples in each signal."""
        return self._hdf5_file["signal"].shape[-1]  # pylint: disable=no-member

    @property
    def sampling_rate(self) -> float:
        """The sampling rate of the signals in Hertz."""
        return self._hdf5_file["signal"].attrs["sampling_rate"]

    def __len__(self) -> int:
        """The number of items in the dataset."""
        return len(self._lookup_table)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Returns the dataset item at `index`."""
        raw_index = self._lookup_table[index]
        original_signal = self._hdf5_file["signal"][raw_index].astype(
            self._config["signal_dtype"], copy=False
        )
        filtered_signal = scipy.signal.sosfiltfilt(self._filter_sos, original_signal)
        if self._config["mean_normalization"]:
            # Use channel-wise means.
            filtered_signal -= filtered_signal.mean(axis=-1, keepdims=True)
        if self._config["feature_scaling"]:
            # Use global standard deviation.
            filtered_signal /= filtered_signal.std()
        item = {
            "input": filtered_signal.take(self._config["in_leads"], axis=0),
            "target": filtered_signal.take(self._config["out_leads"], axis=0),
        }
        if self._config["include_original_signal"]:
            item["original_signal"] = original_signal.copy()
        if self._config["include_filtered_signal"]:
            item["filtered_signal"] = filtered_signal.copy()
        for dst_label_name, src_label_name in self._config["include_labels"].items():
            if isinstance(src_label_name, str):
                item[dst_label_name] = self._hdf5_file[src_label_name][raw_index]
            else:
                item[dst_label_name] = np.array(
                    [self._hdf5_file[name][raw_index] for name in src_label_name]
                )
        return item

    def __getstate__(self) -> dict[str, Any]:
        """Pickles the dataset."""
        _logger.debug("Pickle dataset")
        state = self.__dict__.copy()
        del state["_hdf5_file"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Unpickles the dataset."""
        worker_info = torch.utils.data.get_worker_info()
        _logger.debug(
            "Unpickle dataset in worker %s",
            None if worker_info is None else worker_info.id,
        )
        self.__dict__.update(state)
        self._hdf5_file = h5py.File(
            resolve_path(self._config["hdf5_filename"], relative_to="src/datasets")
        )

    def __del__(self) -> None:
        """Closes the dataset."""
        try:
            # If this function is called on interpreter exit, calling a PyTorch function
            # is very unlikely to succeed.
            worker_info = torch.utils.data.get_worker_info()
            _logger.debug(
                "Close dataset in worker %s",
                None if worker_info is None else worker_info.id,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        try:
            self._hdf5_file.close()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
