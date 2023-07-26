"""
This `generate.py` script should be run from the command line to generate the HDF5 files
for the PTB-XL dataset.

Basic example:
```sh
python3 generate.py                                                                 \
    --download-dir /path/to/ecg-reconstruction/src/ecg/dataset-gen/ptb-xl/downloads \
    --output-dir /path/to/ecg-reconstruction/src/datasets/ptb-xl
```

Advanced example:
```sh
python3 generate.py                                                                 \
    --download-dir /path/to/ecg-reconstruction/src/ecg/dataset-gen/ptb-xl/downloads \
    --output-dir /path/to/ecg-reconstruction/src/datasets/ptb-xl                    \
    --base-filename statement_stratified                                            \
    --split "(('train', 0.7), ('validation', 0.15), ('test', 0.15))"                \
    --label-kind statement                                                          \
    --label-names NORM,NDT,IRBBB,LAFB,NST_,ASMI,CLBBB,IMI                           \
    --resolution high
```

Run `python3 generate.py -h` for more information.
"""

import ast
import itertools
import os
from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Optional

import h5py
import numpy as np
import pandas as pd
import wfdb

_LEAD_NAMES = (
    "I",
    "II",
    "III",
    "AVR",
    "AVL",
    "AVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)


class HDF5Generator:
    """The class that generates HDF5 files for the PTB-XL dataset."""

    def __init__(
        self, download_dir: str | os.PathLike, output_dir: str | os.PathLike
    ) -> None:
        """
        Initializes the generator.

        Args:
          download_dir: The directory where the original dataset is downloaded.
          output_dir: The directory where the generated HDF5 files will be saved.
        """
        self._download_dir = Path(download_dir)
        self._output_dir = Path(output_dir)
        if not self._output_dir.exists():
            self._output_dir.mkdir(parents=True)
        self._database_df: Optional[pd.DataFrame] = None
        self._scp_code_to_classes: Optional[dict[str, tuple[str, str]]] = None

    def generate_hdf5_files(
        self,
        base_filename: str,
        splits: Sequence[tuple[str, float]],
        label_kind: Optional[Literal["statement", "class", "subclass"]],
        label_names: Optional[Sequence[str]] = None,
        resolution: Literal["hr", "lr"] = "hr",
    ) -> None:
        """
        Generates the HDF5 files.

        Args:
          base_filename: The prefix of HDF5 filenames.
          splits: The name and weight of each split.
          label_kind: The kind of labels to use for stratified sampling, if any.
          label_names: The names of labels to use for stratified sampling, if any.
          resolution: Whether to use the high-resolution (500 Hz) or low-resolution
            (100 Hz) data.
        """
        self._read_ptbxl_database_csv()
        if label_kind is None:
            assert label_names is None
            split_data_frames = self._split_dataset([weight for _, weight in splits])
        else:
            assert label_names is not None
            self._read_scp_statements_csv()
            split_data_frames = self._split_dataset_stratified(
                [weight for _, weight in splits], label_kind, label_names
            )
        for (split_name, _), split_df in zip(splits, split_data_frames):
            self._save_hdf5_file(
                f"{base_filename}{split_name}.hdf5", split_df, label_names, resolution
            )

    def _split_dataset(self, split_weights: Sequence[float]) -> list[pd.DataFrame]:
        split_weights_cum = np.array(split_weights)
        split_weights_cum /= split_weights_cum.sum()
        split_weights_cum = np.cumsum(split_weights)
        split_counts_cum = [0] + [
            round(len(self._database_df) * weight_cum)
            for weight_cum in split_weights_cum
        ]
        return [
            self._database_df.iloc[split_counts_cum[i] : split_counts_cum[i + 1]]
            for i in range(len(split_weights))
        ]

    def _split_dataset_stratified(
        self,
        split_weights: Sequence[float],
        label_kind: Literal["statement", "class", "subclass"],
        label_names: Sequence[str],
    ) -> list[pd.DataFrame]:
        split_weights_cum = np.array(split_weights)
        split_weights_cum /= split_weights_cum.sum()
        split_weights_cum = np.cumsum(split_weights_cum)
        split_count_sum = 0
        split_counts = [0] * len(split_weights)
        split_data_frames: list[list[pd.DataFrame]] = [[] for _ in split_weights]

        label_indices = self._database_df["scp_codes"].map(
            lambda s: self._get_label_index(
                ast.literal_eval(s), label_names, label_kind
            ),
        )
        for label_index in range(len(label_names)):
            mask = label_indices == label_index
            stratum_df = self._database_df[mask]
            split_count_sum += len(stratum_df)
            split_counts_old = split_counts
            split_counts_cum = [0] + [
                round(split_count_sum * weight_cum) for weight_cum in split_weights_cum
            ]
            split_counts = [
                split_counts_cum[i + 1] - split_counts_cum[i]
                for i in range(len(split_weights))
            ]
            split_indices = list(
                itertools.accumulate(
                    (
                        count - count_old
                        for count, count_old in zip(split_counts, split_counts_old)
                    ),
                    initial=0,
                ),
            )
            for i in range(len(split_weights)):
                split_data_frames[i].append(
                    stratum_df.iloc[split_indices[i] : split_indices[i + 1]]
                )

        result: list[pd.DataFrame] = []
        for data_frames in split_data_frames:
            split_df = pd.concat(data_frames)
            split_df.sort_index(inplace=True)
            split_df["label_index"] = split_df["scp_codes"].map(
                lambda s: self._get_label_index(
                    ast.literal_eval(s), label_names, label_kind
                ),
            )
            result.append(split_df)
        return result

    def _save_hdf5_file(
        self,
        filename: str | os.PathLike,
        df: pd.DataFrame,
        label_names: Sequence[str] | None,
        resolution: Literal["hr", "lr"],
    ) -> None:
        count = len(df)
        match resolution:
            case "hr":
                signal_length = 5000
                sampling_rate = 500
            case "lr":
                signal_length = 1000
                sampling_rate = 100
            case _:
                raise ValueError(f"invalid resolution kind {resolution}")

        f = h5py.File(self._output_dir / filename, "w")
        id_dataset = f.create_dataset("id", shape=(count,), dtype="int16")
        signal_dataset = f.create_dataset(
            "signal", shape=(count, 12, signal_length), dtype="float16"
        )
        signal_dataset.attrs.create("sampling_rate", sampling_rate, dtype="int16")
        signal_dataset.attrs.create(
            "unit", "mV", dtype=h5py.string_dtype("ascii", length=2)
        )
        signal_dataset.attrs.create(
            "lead_names",
            _LEAD_NAMES,
            dtype=h5py.string_dtype("ascii", length=max(map(len, _LEAD_NAMES))),
        )
        if label_names is None:
            label_dataset = None
        else:
            label_dataset = f.create_dataset("label", shape=(count,), dtype="uint8")
            label_dataset.attrs.create(
                "label_names",
                label_names,
                dtype=h5py.string_dtype("ascii", length=max(map(len, label_names))),
            )
            print(f"Label distribution of {filename}:")
            for label_index, label_name in enumerate(label_names):
                print(f"  {label_name}:", len(df[df["label_index"] == label_index]))
        print(f"Number of elements in {filename}:", len(df))

        for i, (ecg_id, series) in enumerate(df.iterrows()):
            id_dataset[i] = ecg_id
            print(self._download_dir / series["filename_" + resolution])
            signal, metadata = wfdb.rdsamp(
                self._download_dir / series["filename_" + resolution], return_res=16
            )
            assert all(unit == "mV" for unit in metadata["units"])
            assert tuple(metadata["sig_name"]) == _LEAD_NAMES
            signal_dataset[i] = signal.transpose()
            if label_dataset is not None:
                label_dataset[i] = series["label_index"]
        f.close()

    def _get_label_index(
        self,
        scp_codes: Mapping[str, float],
        label_names: Sequence[str],
        label_kind: Literal["statement", "class", "subclass"],
    ) -> int:
        label_index = -1
        for scp_code, likelihood in scp_codes.items():
            if likelihood < 100:
                continue
            match label_kind:
                case "statement":
                    target_label = scp_code
                case "class":
                    if scp_code not in self._scp_code_to_classes:
                        continue
                    target_label = self._scp_code_to_classes[scp_code][0]
                case "subclass":
                    if scp_code not in self._scp_code_to_classes:
                        continue
                    target_label = self._scp_code_to_classes[scp_code][1]
                case _:
                    raise ValueError(f"invalid label kind {label_kind}")
            if target_label in label_names:
                if label_index < 0:
                    label_index = label_names.index(target_label)
                else:
                    return -1
        return label_index

    def _read_ptbxl_database_csv(self) -> None:
        if self._database_df is not None:
            return
        df = pd.read_csv(
            self._download_dir / "ptbxl_database.csv",
            index_col="ecg_id",
            usecols=[
                "ecg_id",
                "scp_codes",
                "electrodes_problems",
                "strat_fold",
                "filename_lr",
                "filename_hr",
            ],
        )
        # Remove signals that are annotated to have electrode problems.
        df.drop(df[~df["electrodes_problems"].isna()].index, inplace=True)
        df.drop("electrodes_problems", axis=1, inplace=True)
        # Sort signals by ID's.
        df.sort_index(inplace=True)
        # Because the original dataset is already stratified, and the last few folds are
        # said to have higher quality, we sort the signals by fold numbers and use the
        # last signals for evaluation and testing.
        df.sort_values("strat_fold", inplace=True, kind="stable")
        self._database_df = df

    def _read_scp_statements_csv(self) -> None:
        if self._scp_code_to_classes is not None:
            return
        df = pd.read_csv(self._download_dir / "scp_statements.csv", index_col=0)
        self._scp_code_to_classes = {
            scp_code: (series["diagnostic_class"], series["diagnostic_subclass"])
            for scp_code, series in df[df["diagnostic"] == 1].iterrows()
        }


def _main() -> None:
    parser = ArgumentParser("generate")
    parser.add_argument(
        "-i",
        "--download-dir",
        type=Path,
        required=True,
        help="The directory where the original dataset is downloaded",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="The directory where the generated HDF5 files will be saved",
    )
    parser.add_argument(
        "-p",
        "--base-filename",
        type=str,
        default="",
        help="The prefix of HDF5 filenames",
    )
    parser.add_argument(
        "-s",
        "--splits",
        type=str,
        default="(('train', 0.7), ('validation', 0.15), ('test', 0.15))",
        help="The name and weight of each split",
    )
    parser.add_argument(
        "-l",
        "--label-kind",
        choices=["statement", "class", "subclass", None],
        default=None,
        help="The kind of labels to use for stratified sampling, if any",
    )
    parser.add_argument(
        "-n",
        "--label-names",
        type=str,
        default=None,
        help="The names of labels to use for stratified sampling, if any",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        choices=["low", "high"],
        default="high",
        help="Whether to use the high-resolution (500 Hz) or low-resolution (100 Hz) "
        "data",
    )
    args = parser.parse_args()
    hdf5_generator = HDF5Generator(args.download_dir, args.output_dir)
    hdf5_generator.generate_hdf5_files(
        args.base_filename,
        ast.literal_eval(args.splits),
        args.label_kind,
        args.label_names.split(",") if args.label_names else None,
        "lr" if args.resolution == "low" else "hr",
    )


if __name__ == "__main__":
    _main()
