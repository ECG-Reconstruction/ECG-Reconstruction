"""Defines functions for measuring and comparing some important parameters of ECG
signals."""

import logging
import sys
from typing import Optional

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import torch
from torch import nn

from .dataset import Dataset

_logger = logging.getLogger(__name__)


_dwt_ecg_delineator = sys.modules[  # # pylint: disable=protected-access
    "neurokit2.ecg.ecg_delineate"
]._dwt_ecg_delineator


def evaluate_synthesized_ecg_parameters(
    model: nn.Module,
    dataset: Dataset,
    dataset_index: int,
    lead_index: int = 7,  # V2
    clean_real: bool = True,
    clean_fake: bool = True,
    debug: bool = False,
) -> Optional[dict[str, np.ndarray]]:
    """
    Evaluates the parameters of real and synthesized ECG signals, and compares the
    differences between them.

    Args:
        model: The reconstruction model to evaluate.
        dataset: The dataset to sample an ECG signal from.
        dataset_index: The index of the ECG signal in the dataset to sample.
        lead_index: The index of the output lead to evaluate.
        clean_real: Whether to clean the real ECG signal before detection.
        clean_fake: Whether to clean the synthesized ECG signal before detection.
        debug: If true, turns on some debugging features for internal use.

    Returns:
        A name-number dictionary of parameters, or `None` if the detection fails.
    """
    dataset_item = dataset[dataset_index]
    input_tensor = torch.from_numpy(np.expand_dims(dataset_item["input"], axis=0)).to(
        next(model.parameters()).device
    )
    with torch.no_grad():
        model.eval()
        output_tensor = model(input_tensor)
    output = np.squeeze(output_tensor.cpu().numpy(), axis=0)
    lead_relative_index = dataset.out_leads.index(lead_index)
    lead_target = dataset_item["target"][lead_relative_index]
    lead_output = output[lead_relative_index]
    return compare_ecg_parameters(
        lead_target, lead_output, dataset.sampling_rate, clean_real, clean_fake, debug
    )


def compare_ecg_parameters(
    real_signal: np.ndarray,
    fake_signal: np.ndarray,
    sampling_rate: int = 500,
    clean_real: bool = True,
    clean_fake: bool = True,
    debug: bool = False,
) -> Optional[dict[str, np.ndarray]]:
    """
    Evaluates the parameters of real and synthesized ECG signals, and compares the
    differences between them.

    Args:
        real_signal: A 1-D array of the real ECG signal.
        fake_signal: A 1-D array of the synthesized ECG signal.
        sampling_rate: The sampling rate of the ECG signal in Hertz.
        clean_real: Whether to clean the real ECG signal before detection.
        clean_fake: Whether to clean the synthesized ECG signal before detection.
        debug: If true, turns on some debugging features for internal use.

    Returns:
        A name-number dictionary of parameters, or `None` if the detection fails.
    """
    assert (
        real_signal.ndim == 1
        and fake_signal.ndim == 1
        and real_signal.shape == fake_signal.shape
    )
    real_points = _delineate_ecg(real_signal, sampling_rate, clean_real)
    fake_points = _delineate_ecg(fake_signal, sampling_rate, clean_fake)
    if real_points is None or fake_points is None:
        return None
    real_indices, fake_indices = _match_r_peaks(
        real_points["ECG_R_Peaks"],
        fake_points["ECG_R_Peaks"],
        round(0.2 * sampling_rate),  # 0.2-second tolerance
    )
    if len(real_indices) == 0 or len(fake_indices) == 0:
        _logger.warning("No matching R peak is found")
        return None

    if debug:
        if (
            should_display := abs(
                len(real_points["ECG_R_Peaks"]) - len(fake_points["ECG_R_Peaks"])
            )
            >= 3
        ):
            _logger.warning("Real R peaks: %s", real_points["ECG_R_Peaks"])
            _logger.warning("Fake R peaks: %s", fake_points["ECG_R_Peaks"])
        raw_real_points = real_points
        raw_fake_points = fake_points

    real_points, fake_points = _align_ecg_points(
        real_points, fake_points, real_indices, fake_indices
    )
    real_points = {
        name: np.array(points, float) for name, points in real_points.items()
    }
    fake_points = {
        name: np.array(points, float) for name, points in fake_points.items()
    }

    if debug and should_display:
        plt.figure(figsize=(10, 6))
        plt.plot(real_signal, color="k")
        plt.vlines(real_points["ECG_P_Onsets"], ymin=-2, ymax=-1, colors="r")
        plt.vlines(real_points["ECG_R_Onsets"], ymin=-1, ymax=0, colors="g")
        plt.vlines(real_points["ECG_R_Peaks"], ymin=0, ymax=1, colors="b")
        plt.vlines(real_points["ECG_R_Offsets"], ymin=1, ymax=2, colors="m")
        plt.vlines(real_points["ECG_T_Offsets"], ymin=2, ymax=3, colors="y")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(fake_signal, color="k")
        plt.vlines(fake_points["ECG_P_Onsets"], ymin=-2, ymax=-1, colors="r")
        plt.vlines(fake_points["ECG_R_Onsets"], ymin=-1, ymax=0, colors="g")
        plt.vlines(fake_points["ECG_R_Peaks"], ymin=0, ymax=1, colors="b")
        plt.vlines(fake_points["ECG_R_Offsets"], ymin=1, ymax=2, colors="m")
        plt.vlines(fake_points["ECG_T_Offsets"], ymin=2, ymax=3, colors="y")
        plt.show()

        raise RuntimeError()

    if debug and np.any(
        np.abs(real_points["ECG_R_Peaks"] - fake_points["ECG_R_Peaks"]) > 80
    ):
        _logger.warning("Real R peaks (aligned): %s", real_points["ECG_R_Peaks"])
        _logger.warning("Real R peaks (raw): %s", raw_real_points["ECG_R_Peaks"])
        _logger.warning("Fake R peaks (aligned): %s", fake_points["ECG_R_Peaks"])
        _logger.warning("Fake R peaks (raw): %s", raw_fake_points["ECG_R_Peaks"])

        plt.figure(figsize=(10, 6))
        plt.plot(real_signal, color="k")
        plt.vlines(real_points["ECG_R_Peaks"], ymin=0, ymax=1, colors="r")
        plt.vlines(raw_real_points["ECG_R_Peaks"], ymin=-1, ymax=0, colors="g")
        plt.title("Real")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(fake_signal, color="k")
        plt.vlines(fake_points["ECG_R_Peaks"], ymin=0, ymax=1, colors="r")
        plt.vlines(raw_fake_points["ECG_R_Peaks"], ymin=-1, ymax=0, colors="g")
        plt.title("Fake")
        plt.show()

        raise RuntimeError()

    diffs: dict[str, np.ndarray] = {}

    real_rr = real_points["ECG_R_Onsets"][1:] - real_points["ECG_R_Onsets"][:-1]
    fake_rr = fake_points["ECG_R_Onsets"][1:] - fake_points["ECG_R_Onsets"][:-1]
    diffs["RR"] = np.absolute(fake_rr - real_rr) / real_rr

    real_pr = real_points["ECG_R_Onsets"] - real_points["ECG_P_Onsets"]
    fake_pr = fake_points["ECG_R_Onsets"] - fake_points["ECG_P_Onsets"]
    diffs["PR"] = np.absolute(fake_pr - real_pr) / real_pr

    real_qrs = real_points["ECG_R_Offsets"] - real_points["ECG_R_Onsets"]
    fake_qrs = fake_points["ECG_R_Offsets"] - fake_points["ECG_R_Onsets"]
    diffs["QRS"] = np.absolute(fake_qrs - real_qrs) / real_qrs

    real_qt = real_points["ECG_T_Offsets"] - real_points["ECG_R_Onsets"]
    fake_qt = fake_points["ECG_T_Offsets"] - fake_points["ECG_R_Onsets"]
    diffs["QT"] = np.absolute(fake_qt - real_qt) / real_qt

    real_qtcb = real_qt[1:] / np.sqrt(real_rr * sampling_rate)
    fake_qtcb = fake_qt[1:] / np.sqrt(fake_rr * sampling_rate)
    diffs["QTcB"] = np.absolute(fake_qtcb - real_qtcb) / real_qtcb

    return diffs


def _delineate_ecg(
    signal: np.ndarray,
    sampling_rate: int = 500,
    clean: bool = True,
) -> Optional[dict[str, list[int | float]]]:
    if not clean:
        signal = nk.ecg_clean(signal)
    r_peaks = nk.ecg_peaks(signal, sampling_rate, method="neurokit")[1]["ECG_R_Peaks"]
    assert all(map(np.isfinite, r_peaks))
    if len(r_peaks) <= 3:
        _logger.warning("Number of detected R peaks %d is too small", len(r_peaks))
        return None
    waves: dict[str, list[int | float]] = _dwt_ecg_delineator(
        signal, r_peaks, sampling_rate
    )
    assert "ECG_R_Peaks" not in waves
    if any(len(points) != len(r_peaks) for points in waves.values()):
        _logger.warning(
            "Inconsistent numbers of points: %s",
            {
                "ECG_R_Peaks": len(r_peaks),
                **{name: len(points) for name, points in waves.items()},
            },
        )
        return None
    waves["ECG_R_Peaks"] = r_peaks
    return waves


def _match_r_peaks(
    a_peaks: list[int], b_peaks: list[int], tolerance: int = -1
) -> tuple[list[int], list[int]]:
    abs_diffs = np.absolute(
        np.expand_dims(np.array(a_peaks, int), axis=1) - np.array(b_peaks, int)
    )
    closest_a_to_b = np.argmin(abs_diffs, axis=1)
    closest_b_to_a = np.argmin(abs_diffs, axis=0)
    a_indices: list[int] = []
    b_indices: list[int] = []
    for a_index, a_peak in enumerate(a_peaks):
        b_index = closest_a_to_b[a_index]
        a_index_cycle = closest_b_to_a[b_index]
        if a_index_cycle == a_index and (
            tolerance < 0 or abs(a_peak - b_peaks[b_index]) <= tolerance
        ):
            a_indices.append(a_index)
            b_indices.append(b_index)
    return a_indices, b_indices


def _align_ecg_points(
    a_points: dict[str, list[int | float]],
    b_points: dict[str, list[int | float]],
    a_indices: list[int],
    b_indices: list[int],
) -> tuple[dict[str, list[int | float]], dict[str, list[int | float]]]:
    assert len(a_indices) == len(b_indices)
    np_a_indices = np.array([-1, *a_indices, len(next(iter(a_points.values())))], int)
    np_b_indices = np.array([-1, *b_indices, len(next(iter(b_points.values())))], int)
    gaps = (np_a_indices[1:] - np_a_indices[:-1] > 1) | (
        np_b_indices[1:] - np_b_indices[:-1] > 1
    )
    a_points_aligned = {name: [] for name in a_points}
    b_points_aligned = {name: [] for name in b_points}
    for i, (a_index, b_index) in enumerate(zip(a_indices, b_indices)):
        gap_before = gaps[i]
        gap_after = gaps[i + 1]
        for name in (
            "ECG_P_Onsets",
            "ECG_P_Peaks",
            "ECG_P_Offsets",
            "ECG_Q_Peaks",
            "ECG_R_Onsets",
        ):
            a_points_aligned[name].append(
                np.nan if gap_before else a_points[name][a_index]
            )
            b_points_aligned[name].append(
                np.nan if gap_before else b_points[name][b_index]
            )
        a_points_aligned["ECG_R_Peaks"].append(a_points["ECG_R_Peaks"][a_index])
        b_points_aligned["ECG_R_Peaks"].append(b_points["ECG_R_Peaks"][b_index])
        for name in (
            "ECG_R_Offsets",
            "ECG_S_Peaks",
            "ECG_T_Onsets",
            "ECG_T_Peaks",
            "ECG_T_Offsets",
        ):
            a_points_aligned[name].append(
                np.nan if gap_after else a_points[name][a_index]
            )
            b_points_aligned[name].append(
                np.nan if gap_after else b_points[name][b_index]
            )
        if gap_after:
            for points_dict in a_points_aligned, b_points_aligned:
                for points in points_dict.values():
                    points.append(np.nan)
    return a_points_aligned, b_points_aligned
