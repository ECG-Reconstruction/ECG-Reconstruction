"""ECG reconstruction models with a transformer architecture and models with a mix of
transformer and U-Net architectures."""

import math
from collections.abc import Sequence
from typing import Optional

import torch
from optuna import Trial
from torch import nn, Tensor

from ..reconstructor import Reconstructor
from ..unet.unet import DoubleConv, DownSample, OutConv, UpSample


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, 1, d_model)
        encoding[:, 0, 0::2] = torch.sin(position * div_term)
        encoding[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs: A tensor of shape `(seq_len, batch_size, d_model)`.

        Returns:
            A tensor of the same shape.
        """
        return self.dropout(inputs + self.encoding[: inputs.size(dim=0)])


class NaiveTransformerEncoder(Reconstructor):
    """An ECG reconstruction model with an encoder, a transformer, and a decoder."""
    max_batch_size = 2

    def __init__(
        self,
        in_leads: Sequence[int],
        out_leads: Sequence[int],
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        """
        Initializes the model.

        Args:
            in_leads: Indices of the input leads.
            out_leads: Indices of the output leads.
            d_model: The number of input features in the transformer.
            num_heads: The number of heads in the transformer.
            dim_feedforward: The dimension of the feedforward network model in the
              transformer.
            num_layers: The number of layers in the transformer encoder.
            dropout: The dropout value of the transformer encoder.
        """
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = _PositionalEncoding(d_model, dropout)
        self.encoder = nn.Conv1d(len(in_leads), d_model, kernel_size=1)
        # self.encoder = nn.Sequential(
        #     nn.Conv1d(
        #         len(in_leads), d_model // 4, kernel_size=3, padding="same", bias=False
        #     ),
        #     # nn.ELU(),
        #     nn.MaxPool1d(2),
        #     nn.Conv1d(
        #         d_model // 4, d_model // 2, kernel_size=3, padding="same", bias=False
        #     ),
        #     # nn.ELU(),
        #     nn.MaxPool1d(2),
        #     nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding="same", bias=False),
        # )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout),
            num_layers,
        )
        self.decoder = nn.Conv1d(d_model, len(out_leads), kernel_size=1)
        # self.decoder = nn.Sequential(
        #     nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding="same", bias=False),
        #     # nn.ELU(),
        #     nn.Upsample(scale_factor=2, mode="linear", align_corners=True),
        #     nn.Conv1d(
        #         d_model // 2, d_model // 4, kernel_size=3, padding="same", bias=False
        #     ),
        #     # nn.ELU(),
        #     nn.Upsample(scale_factor=2, mode="linear", align_corners=True),
        #     nn.Conv1d(
        #         d_model // 4, len(out_leads), kernel_size=3, padding="same", bias=False
        #     ),
        # )

    def forward(self, inputs: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            inputs: A tensor of shape `(batch_size, in_channels, seq_len)`.
            src_mask: An optional tensor of shape `(seq_len, seq_len)`.

        Returns:
            A tensor of shape `(batch_size, out_channels, seq_len)`.
        """
        # (batch_size, d_model, seq_len)
        src = self.encoder(inputs) * math.sqrt(self.d_model)
        # src = inputs * math.sqrt(self.d_model)
        # (seq_len, batch_size, d_model)
        src = self.positional_encoding(src.permute(2, 0, 1))
        outputs = self.transformer_encoder(src, src_mask)
        # (batch_size, out_channels, seq_len)
        return self.decoder(outputs.permute(1, 2, 0))
        # return outputs.permute(1, 2, 0)


    @staticmethod
    def default_config() -> dict:
        return {
            "reconstructor": {
                "args": {
                    "d_model": 64,
                    "num_heads": 2,
                    "dim_feedforward": 64,
                    "num_layers": 2,
                    "dropout": 0.2,
                }
            },
            "optimizer": {
                "type": "AdamW",
                "args": {
                    "lr": 2e-3,
                    "weight_decay": 5e-2,
                },
            },
            "lr_scheduler": {
                "type": "ReduceLROnPlateau",
                "args": {
                    "factor": 0.5,
                    "patience": 5,
                },
            },
            "dataloader": {
                "common": {"batch_size": 128},
            },
        }

    @staticmethod
    def suggest_config(trial: Trial) -> dict:
        config = NaiveTransformerEncoder.default_config()
        config["reconstructor"]["args"].update(
            d_model=trial.suggest_int("d_model", 8, 128, step=32),
            num_heads=trial.suggest_categorical("num_heads", [1, 2, 4, 8]),
            dim_feedforward=trial.suggest_int("dim_feedforward", 32, 128, step=32),
            num_layers=trial.suggest_int("num_layers", 1, 4),
        )

        return config


class DSTransformerEncoder(Reconstructor):
    """An ECG reconstruction model based on a naive transformer encoder, but
    additionally inputs to the transformer are down-sampled and outputs of the
    transformer are up-sampled."""

    def __init__(
        self,
        in_leads: Sequence[int],
        out_leads: Sequence[int],
        down_rate: int,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        """
        Initializes the model.

        Args:
            in_leads: Indices of the input leads.
            out_leads: Indices of the output leads.
            down_rate: Inputs to the transformer are down-sampled by this rate, and
              outputs of the transformer are up-sampled by this rate.
            d_model: The number of input features in the transformer.
            num_heads: The number of heads in the transformer.
            dim_feedforward: The dimension of the feedforward network model in the
              transformer.
            num_layers: The number of layers in the transformer encoder.
            dropout: The dropout value of the transformer encoder.
        """
        super().__init__()
        self.down_rate = down_rate
        self.positional_encoding = _PositionalEncoding(d_model, dropout)
        self.input_embedding = nn.Sequential(
            nn.Conv1d(
                len(in_leads),
                d_model // 4,
                kernel_size=3,
                dilation=2,
                padding="same",
                bias=False,
            ),
            nn.ELU(),
            nn.Conv1d(
                d_model // 4,
                d_model // 2,
                kernel_size=3,
                dilation=2,
                padding="same",
                bias=False,
            ),
            nn.ELU(),
            nn.Conv1d(
                d_model // 2,
                d_model,
                kernel_size=3,
                dilation=2,
                padding="same",
                bias=False,
            ),
        )
        self.encoder = nn.Conv1d(d_model * down_rate, d_model, kernel_size=1)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout),
            num_layers,
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding="same", bias=False),
            nn.ELU(),
            nn.Conv1d(
                d_model // 2, d_model // 4, kernel_size=3, padding="same", bias=False
            ),
            nn.ELU(),
            nn.Conv1d(
                d_model // 4,
                len(out_leads) * down_rate,
                kernel_size=3,
                padding="same",
                bias=False,
            ),
        )

    def forward(self, inputs: Tensor, src_mask=None) -> Tensor:
        """
        Args:
            inputs: A tensor of shape `(batch_size, in_channels, seq_len)`.
            src_mask: A tensor of shape `(seq_len, seq_len)`.

        Returns:
            A tensor of shape `(batch_size, out_channels, seq_len)`.
        """
        inputs = self.input_embedding(inputs)
        batch_size, d_model, seq_len = inputs.size()
        if pad_size := seq_len % self.down_rate:
            # Pad zeros on the back.
            inputs = nn.functional.pad(inputs, (0, pad_size), mode="constant", value=0)
            seq_len = inputs.size(dim=-1)
        # (batch_size, d_model * down_rate, (seq_len + pad_size) / down_rate)
        inputs = inputs.reshape(
            batch_size, d_model * self.down_rate, seq_len // self.down_rate
        )
        src = self.encoder(inputs) * math.sqrt(d_model)
        # ((seq_len + pad_size) / down_rate, batch_size, d_model * down_rate)
        src = self.positional_encoding(src.permute(2, 0, 1))
        outputs = self.transformer_encoder(src, src_mask)
        # (batch_size, out_channels * down_rate, (seq_len + pad_size) / down_rate)
        outputs = self.decoder(outputs.permute(1, 2, 0))
        # (batch_size, out_channels, seq_len + pad_size)
        outputs = outputs.reshape(batch_size, -1, seq_len)
        # (batch_size, out_channels, seq_len)
        return outputs[..., : seq_len - pad_size]

    @staticmethod
    def default_config() -> dict:
        config = NaiveTransformerEncoder.default_config()
        config["reconstructor"]["args"]["down_rate"] = 10
        return config

    @staticmethod
    def suggest_config(trial: Trial) -> dict:
        config = NaiveTransformerEncoder.suggest_config(trial)
        config["reconstructor"]["args"]["down_rate"] = trial.suggest_categorical(
            "down_rate", [5, 8, 10, 20]
        )
        return config


class UFormer(Reconstructor):
    """An ECG reconstruction model based on the U-Net, but with an additional
    transformer between the encoder and decoder."""
    max_batch_size = 16
    def __init__(
        self,
        in_leads: Sequence[int],
        out_leads: Sequence[int],
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        min_channels: int,
        min_dilation: int,
        dilation_rate: int,
        bilinear: bool,
        transformer_num_layers: int,
        unet_num_layers: int,
        dropout: float,
        **kwargs,
    ) -> None:
        """
        Initializes the model.

        Args:
            in_leads: Indices of the input leads.
            out_leads: Indices of the output leads.
            d_model: The number of input features in the transformer.
            num_heads: The number of heads in the transformer.
            dim_feedforward: The dimension of the feedforward network model in the
              transformer.
            min_channels: The number of input channels in the first convolutional layer
              of the U-Net encoder.
            min_dilation: The base dilation of the convolutional layers in the U-Net
              encoder.
            dilation_rate: The rate at which the dilation increases in the U-Net
              encoder.
            bilinear: Whether to use bilinear interpolation in the U-Net decoder.
            transformer_num_layers: The number of layers in the transformer encoder.
            unet_num_layers: The number of convolutional layers in the U-Net encoder or
              decoder.
            dropout: The dropout value of the positional encoding and the transformer
              encoder.
            kwargs: Additional arguments to `DoubleConv`.
        """
        super().__init__()
        self.d_model = d_model
        self.in_layer = DoubleConv(len(in_leads), min_channels, **kwargs)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(unet_num_layers):
            in_channels = min_channels << i
            out_channels = in_channels * 2
            dilation = min_dilation * dilation_rate ** (i + 1)
            self.encoder.append(
                DownSample(in_channels, out_channels, dilation=dilation, **kwargs)
            )
            self.decoder.append(
                UpSample(out_channels, in_channels, bilinear, dilation=1, **kwargs)
            )
        self.decoder = self.decoder[::-1]
        self.out_layer = OutConv(min_channels, len(out_leads))

        center_in_channels = out_channels
        self.positional_encoding = _PositionalEncoding(d_model, dropout)
        self.input_embedding = nn.Conv1d(center_in_channels, d_model, kernel_size=1)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout),
            transformer_num_layers,
        )
        self.output_transform = nn.Conv1d(d_model, center_in_channels, kernel_size=1)

    def forward(self, inputs: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            inputs: A tensor of shape `(batch_size, in_channels, seq_len)`.
            src_mask: An optional tensor of shape `(seq_len, seq_len)`.

        Returns:
            A tensor of shape `(batch_size, out_channels, seq_len)`.
        """
        encoder_outputs = [self.in_layer(inputs)]
        for layer in self.encoder:
            encoder_outputs.append(layer(encoder_outputs[-1]))
        # (batch_size, min_channels << unet_num_layers, seq_len)
        transformer_inputs = encoder_outputs.pop()
        # (batch_size, d_model, seq_len)
        src = self.input_embedding(transformer_inputs) * math.sqrt(self.d_model)
        # (seq_len, batch_size, d_model)
        src = self.positional_encoding(src.permute(2, 0, 1))
        outputs = self.transformer_encoder(src, src_mask)
        # (batch_size, min_channels << unet_num_layers, seq_len)
        direct_inputs = self.output_transform(outputs.permute(1, 2, 0))
        for layer in self.decoder:
            skip_inputs = encoder_outputs.pop()
            direct_inputs = layer(direct_inputs, skip_inputs)
        return self.out_layer(direct_inputs)

    @staticmethod
    def default_config() -> dict:
        config = NaiveTransformerEncoder.default_config()
        transformer_num_layers = config["reconstructor"]["args"].pop("num_layers")
        config["reconstructor"]["args"].update(
            min_channels=16,
            min_dilation=1,
            dilation_rate=2,
            bilinear=True,
            transformer_num_layers=transformer_num_layers,
            unet_num_layers=2,
        )
        return config

    @staticmethod
    def suggest_config(trial: Trial) -> dict:
        config = UFormer.default_config()
        config["reconstructor"]["args"].update(
            # Transformer part.
            d_model=trial.suggest_int("d_model", 32, 256, step=32),
            num_heads=trial.suggest_categorical("num_heads", [1, 2, 4, 8]),
            dim_feedforward=trial.suggest_int("dim_feedforward", 32, 256, step=32),
            transformer_num_layers=trial.suggest_int("transformer_num_layers", 1, 3),
            # U-Net part.
            unet_num_layers=trial.suggest_int("unet_num_layers", 3, 5),
            min_channels=trial.suggest_int("min_channels", 16, 32, step=4),
            kernel_size=trial.suggest_int("kernel_size", 3, 33, step=2),
            min_dilation=trial.suggest_int("min_dilation", 1, 4),
            dilation_rate=trial.suggest_int("dilation_rate", 1, 8),
        )
        batch_size = trial.suggest_categorical("batch_size", [64, 128])
        config["dataloader"]["common"]["batch_size"] = batch_size
        # to avoide OOM
        config["accumulate_grad_batches"] = batch_size // 64
        return config
