import logging
import os
import sys
import yaml
import torch
import thop
from io import StringIO

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
from ecg.reconstructor.transformer.transformer import UFormer, NaiveTransformerEncoder
from ecg.reconstructor.lstm.lstm import LSTM, CNNLSTM
from ecg.reconstructor.cnn.cnn import StackedCNN
from ecg.reconstructor.transformer.fastformer import Fastformer, UFastformer, FastformerPlus, FastformerZero, FastformerStuff
from ecg.reconstructor.unet.unet import UNet
from ecg.util.path import resolve_path
from ecg.util.tree import deep_merge

dataset_name = "ptb-xl"  # "code15%"
# dataset_name = "code15%"

training_list = [
    # FastformerZero
    FastformerStuff

    # StackedCNN,
    # LSTM,
    # # NaiveTransformerEncoder,
    # FastformerPlus,
    # CNNLSTM,
    # UFormer,
    # UNet,
    # Fastformer,
    # UFastformer,
]

base_config: TrainerConfig = {
    "in_leads": [0, 1, 8],
    "out_leads": [6, 7, 9, 10, 11],
    "max_epochs": 32,
    "dataset": {
        "train": {"hdf5_filename": f"{dataset_name}/train.hdf5"},
        "eval": {"hdf5_filename": f"{dataset_name}/validation.hdf5"},
    },
}

if __name__ == "__main__":
    for MODEL_TYPE in training_list:
        with open(
            resolve_path("src/best_configs") / MODEL_TYPE.__name__ / "tuned_config.yaml",
            "r", encoding="utf-8",
        ) as fp:
            best_config = yaml.load(fp, Loader=yaml.Loader)

        config = deep_merge(best_config, base_config)
        config['dataloader']['common']['num_workers'] = 6
        config['reconstructor']['type'] = MODEL_TYPE

        config_stream = StringIO()
        yaml.dump(config, config_stream, yaml.Dumper, indent=4)
        logging.info("Config:\n%s", config_stream.getvalue())
        
        trainer = Trainer(config)
        total_params = sum(param.numel() for param in trainer.reconstructor.parameters())
        logging.info("Number of parameters: %d", total_params)
        device = next(iter(trainer.reconstructor.parameters())).device
        dummy_input = torch.from_numpy(trainer.eval_dataset[0]["input"][None, ...]).to(device)
        with torch.no_grad():
            macs, params = thop.profile(trainer.reconstructor, (dummy_input,))
        macs_g = macs / 1e9
        params_m = params / 1e6
        logging.info("MACs (G): %f", macs_g)
        logging.info("Params (M): %f", params_m)
        trainer.fit()
