import logging
import os
import sys
from pathlib import Path

import yaml
from tqdm.auto import tqdm

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
from ecg.reconstructor.transformer.transformer import UFormer
from ecg.reconstructor.lstm.lstm import LSTM
from ecg.reconstructor.linear.linear import Linear
from ecg.util.tree import deep_merge




if __name__ == '__main__':
    MODEL_TYPE = Linear
    dataset = "code15%"

    # current_range = range(6,8)
    # current_range = range(8,10)
    current_range = range(10,12)
    for i in tqdm(current_range):
        config: TrainerConfig = {
            "in_leads": [0, 1, i],
            "out_leads": [oidx for oidx in range(6, 12) if oidx != i],
            "max_epochs": 32,
            "accumulate_grad_batches": 8,
            "dataset": {
                "common": {
                    "predicate": None,
                    "signal_dtype": "float32",
                    "filter_type": "butter",
                    "filter_args": {"N": 3, "Wn": (0.5, 60), "btype": "bandpass"},
                    # "filter_args": {"N": 3, "Wn": (0.05, 150), "btype": "bandpass"},
                    # "mean_normalization": True,
                    "mean_normalization": False,
                    "feature_scaling": False,
                    "include_original_signal": False,
                    "include_filtered_signal": False, # This will be set to True in visulization
                    "include_labels": {},
                },
                "train": {"hdf5_filename": f"{dataset}/train.hdf5"},
                "eval": {"hdf5_filename": f"{dataset}/validation.hdf5"},
            },
            "dataloader": {
                "common": {"num_workers": 6},
            },
            "reconstructor": {"type": MODEL_TYPE},
        }
        # with open(os.path.join(f"../best_configs/{MODEL_TYPE.__name__}/tuned_config.yaml"), 'r') as fp:
        #     best_config = yaml.safe_load(fp)

        config = deep_merge(config, MODEL_TYPE.default_config())
        config['reconstructor']["args"]['in_leads'] = config['in_leads']
        config['reconstructor']["args"]['out_leads'] = config['out_leads']
        config['dataloader']['common']["batch_size"] = 256
        config["accumulate_grad_batches"] = 1
        trainer = Trainer(config)
        trainer.fit()
        