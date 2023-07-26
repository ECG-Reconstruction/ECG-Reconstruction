import json
import logging
import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys
from functools import partial
from pathlib import Path
from typing import Optional
from io import StringIO
import traceback

import optuna
import torch
from optuna.storages import JournalStorage, JournalFileStorage
import yaml

logging.basicConfig(level=logging.INFO)

try:
    from google.colab import drive
except ImportError:
    logging.info("Local machine detected")
    sys.path.append(os.path.realpath(".."))
else:
    logging.info("Colab detected")
    drive.mount("/content/drive")
    sys.path.append("/content/drive/MyDrive/ecg-reconstruction/src")

from ecg.trainer import Trainer, TrainerConfig
from ecg.reconstructor.cnn.cnn import StackedCNN
from ecg.reconstructor.transformer.transformer import UFormer, NaiveTransformerEncoder
from ecg.reconstructor.transformer.fastformer import (
    Fastformer,
    UFastformer,
    FastformerPlus,
)
from ecg.reconstructor.lstm.lstm import LSTM, CNNLSTM
from ecg.util.device import get_device
from ecg.util.tree import deep_merge
from ecg.util.path import get_project_root_dir


def train_experiment(
    trial: optuna.Trial,
    base_config: TrainerConfig,
    tuning_dir: Path,
) -> Optional[float]:
    """
    This is the main function for optuna to tune a model.
    """
    reconstructor_type = base_config["reconstructor"]["type"]
    config = deep_merge(base_config, reconstructor_type.suggest_config(trial))

    # The followings are the configs after tuning.
    config["optimizer"]["args"] = {
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 5e-6, 5e-2, log=True),
    }
    if config["optimizer"]["type"] == "AdamW":
        config["optimizer"]["args"]["betas"] = [
            trial.suggest_float("b1", 0.85, 0.95, log=True),
            trial.suggest_float("b2", 0.950, 0.9999, log=True),
        ]

    config["lr_scheduler"] = {}
    scheduler_type = config["lr_scheduler"]["type"] = trial.suggest_categorical(
        "type", ["CosineAnnealingWarmRestarts", "ReduceLROnPlateau"]
    )
    if scheduler_type == "ReduceLROnPlateau":
        config["lr_scheduler"]["args"] = {
            "factor": trial.suggest_float("factor", 0.2, 0.8),
            "patience": trial.suggest_int("patience", 2, 5),
        }
    elif scheduler_type == "CosineAnnealingWarmRestarts":
        config["lr_scheduler"]["args"] = {
            "T_0": trial.suggest_int("T_0", 1, 4),
            "T_mult": 1,
        }

    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    config["dataloader"]["common"]["batch_size"] = batch_size
    # to avoide OOM
    config["accumulate_grad_batches"] = max(
        1, batch_size // reconstructor_type.max_batch_size
    )

    config_stream = StringIO()
    yaml.dump(config, config_stream, yaml.Dumper, indent=4)
    logging.info("Config:\n%s", config_stream.getvalue())

    config_to_log = config.copy()
    del config_to_log["reconstructor"]
    logging.info(json.dumps(config_to_log, indent=4))
    trainer = Trainer(config)
    tuning_config_dir = tuning_dir / f"trial_{trial.number}"
    with tuning_config_dir.open("w", encoding="utf-8") as config_file:
        yaml.dump(trainer.config, config_file, Dumper=yaml.Dumper)
    try:
        loss = trainer.fit(trial=trial)
    except RuntimeError:
        error_stream = StringIO()
        traceback.print_exc(file=error_stream)
        logging.error("Training failed\n%s", error_stream.getvalue())
        loss = None
    del trainer
    if get_device().type == "cuda":
        torch.cuda.empty_cache()
    return loss


# MODEL_TYPE = NaiveTransformerEncoder
# MODEL_TYPE = StackedCNN
MODEL_TYPE = FastformerPlus
# MODEL_TYPE = CNNLSTM
dataset_name = "ptb-xl"
trial_epochs = 8
n_trials = 50

base_config: TrainerConfig = {
    "in_leads": [0, 1, 8],
    "out_leads": [6, 7, 9, 10, 11],
    "max_epochs": trial_epochs,
    "accumulate_grad_batches": 1,
    "dataset": {
        "common": {
            # "predicate": "lambda f: f['SB'][:]",
            "predicate": None,
            "signal_dtype": "float32",
            "filter_type": "butter",
            "filter_args": {"N": 3, "Wn": (0.5, 60), "btype": "bandpass"},
            "mean_normalization": True,
            "feature_scaling": False,
            "include_original_signal": False,
            "include_filtered_signal": False,
            "include_labels": {},
        },
        "train": {"hdf5_filename": f"{dataset_name}/train.hdf5"},
        "eval": {"hdf5_filename": f"{dataset_name}/validation.hdf5"},
    },
    "dataloader": {
        "common": {"num_workers": 6},
    },
    "reconstructor": {"type": MODEL_TYPE},
}


if __name__ == "__main__":
    tuning_dir = get_project_root_dir() / "src" / "tuning" / MODEL_TYPE.__name__
    tuning_dir.mkdir(exist_ok=True, parents=True)
    experiment_name = f"tuning_logs_{MODEL_TYPE.__name__}"
    storage = JournalStorage(
        JournalFileStorage(str(tuning_dir / f"{experiment_name}.log"))
    )
    try:
        study = optuna.load_study(
            storage=storage,
            study_name=experiment_name,
        )
    except:
        study = optuna.create_study(
            direction="minimize",
            storage=storage,  # Specify the storage URL here.
            study_name=experiment_name,
            pruner=optuna.pruners.MedianPruner(),
            load_if_exists=True,
        )
    trial_function = partial(
        train_experiment, base_config=base_config, tuning_dir=tuning_dir
    )
    # study.optimize(trial_function, n_trials=n_trials)
    best_number = study.best_trial.number
    with open(tuning_dir / "best_trial", "w") as f:
        f.write(f"{best_number}")
    print(study.best_params)
