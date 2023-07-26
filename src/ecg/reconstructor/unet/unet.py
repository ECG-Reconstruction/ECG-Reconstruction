"""ECG reconstruction models with a U-Net architecture."""

from collections.abc import Sequence
from typing import Optional

import torch
from optuna import Trial
from torch import nn, Tensor

from ..reconstructor import Reconstructor


class DoubleConv(nn.Module):
    """The double-convolution module of a U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        kernel_size: int = 3,
        padding: str | int = "same",
        norm_type: type[nn.Module] = nn.BatchNorm1d,
        dilation: int = 1,
        dilation_scale: int = 2,
        bias: bool = False,
        **kwargs,
    ) -> None:
        """
        Initializes the module.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            mid_channels: The number of channels between the two convolutional layers.
            kernel_size: The kernel size of convolutional layers.
            padding: The padding of convolutional layers.
            norm_type: The normalization applied after each convolutional layer.
            dilation: The dilation of the first convolutional layer.
            dilation_scale: The dilation scale of the second convolutional layer.
            bias: Whether to add a bias term to convolutional layers.
            kwargs: Additional arguments to convolutional layers.
        """
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels * 3 
        # if not mid_channels:
        #     mid_channels = out_channels
        self.stage1 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                mid_channels,
                kernel_size,
                padding=padding,
                bias=bias,
                dilation=dilation,
                **kwargs,
            ),
            norm_type(mid_channels),
            nn.LeakyReLU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv1d(
                mid_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
                dilation=dilation * dilation_scale,
                **kwargs,
            ),
            norm_type(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Performs convolution on the inputs."""
        return self.stage2((self.stage1(inputs)))


class DownSample(nn.Module):
    """The down-sampling module of a U-Net."""

    def __init__(
        self, in_channels: int, out_channels: int, pool_size: int = 2, **kwargs
    ) -> None:
        """
        Initializes the module.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            pool_size: The kernel size of the pooling layer.
            kwargs: Additional arguments to `DoubleConv`.
        """
        super().__init__()
        self.down_layer = nn.Sequential(
            nn.MaxPool1d(pool_size),
            DoubleConv(in_channels, out_channels, **kwargs),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Down-samples the inputs."""
        return self.down_layer(inputs)


class UpSample(nn.Module):
    """The up-sampling module of a U-Net."""

    def __init__(
        self, in_channels: int, out_channels: int, bilinear: bool = True, **kwargs
    ) -> None:
        """
        Initializes the module.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            bilinear: Whether to use bilinear interpolation.
            kwargs: Additional arguments to `DoubleConv`.
        """
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="linear", align_corners=True),
                nn.Conv1d(in_channels, in_channels // 2, kernel_size=3, stride=2),
            )
        else:
            self.up = nn.ConvTranspose1d(
                in_channels, in_channels // 2, kernel_size=3, stride=2
            )
        self.conv = DoubleConv(in_channels, out_channels, **kwargs)
        self.bridge_conv = nn.Conv1d(
            in_channels // 2,
            in_channels // 2,
            kernel_size=1,
            padding="same",
            bias=False,
        )

    def forward(self, direct_inputs: Tensor, skip_inputs: Tensor) -> Tensor:
        """Up-samples `direct_inputs` and concatenates it with `skip_inputs`."""
        direct_inputs = self.up(direct_inputs)
        skip_inputs = self.bridge_conv(skip_inputs)
        diff_y = skip_inputs.size(dim=2) - direct_inputs.size(dim=2)
        direct_inputs = nn.functional.pad(
            direct_inputs, (diff_y // 2, diff_y - diff_y // 2)
        )
        outputs = torch.cat((skip_inputs, direct_inputs), dim=1)
        return self.conv(outputs)


class OutConv(nn.Module):
    """The final convolutional module of a U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initializes the module.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
        """
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, padding="same", kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        """Performs convolution on the inputs."""
        return self.conv(inputs)


class OutConvExt(nn.Module):
    """The final convolutional module of a U-Net, extended version."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initializes the module.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels * 2, padding="same", kernel_size=3),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(in_channels * 2, in_channels * 2, padding="same", kernel_size=3),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(in_channels * 2, out_channels, padding="same", kernel_size=1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Performs convolution on the inputs."""
        return self.conv(inputs)


class UNet(Reconstructor):
    """An ECG reconstruction model with a U-Net architecture."""
    max_batch_size = 128
    def __init__(
        self,
        in_leads: Sequence[int],
        out_leads: Sequence[int],
        num_layers: int,
        min_channels: int,
        min_dilation: int,
        dilation_rate: int,
        bilinear: bool,
        out_conv_ext: bool,
        **kwargs,
    ) -> None:
        """
        Initializes the model.

        Args:
            in_leads: Indices of the input leads.
            out_leads: Indices of the output leads.
            num_layers: The number of convolutional layers in the encoder or decoder.
            min_channels: The number of input channels in the first convolutional layer
              of the encoder.
            min_dilation: The base dilation of the convolutional layers in the encoder.
            dilation_rate: The rate at which the dilation increases in the encoder.
            bilinear: Whether to use bilinear interpolation in the decoder.
            out_conv_ext: Whether to use the extended output layer (i.e., `OutConvExt`
              v.s. `OutConv`).
        """
        super().__init__()
        self.in_layer = DoubleConv(len(in_leads), min_channels, **kwargs)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(num_layers):
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
        out_conv_type = OutConvExt if out_conv_ext else OutConv
        self.out_layer = out_conv_type(min_channels, len(out_leads))

    def forward(self, inputs: Tensor) -> Tensor:
        encoder_outputs = [self.in_layer(inputs)]
        for layer in self.encoder:
            encoder_outputs.append(layer(encoder_outputs[-1]))
        direct_inputs = encoder_outputs.pop()
        for layer in self.decoder:
            skip_inputs = encoder_outputs.pop()
            direct_inputs = layer(direct_inputs, skip_inputs)
        return self.out_layer(direct_inputs)

    @staticmethod
    def default_config() -> dict:
        return {
            "reconstructor": {
                "args": {
                    "num_layers": 3,
                    "min_channels": 16,
                    "min_dilation": 1,
                    "dilation_rate": 2,
                    "bilinear": True,
                    "out_conv_ext": False,
                }
            },
            "optimizer": {
                "type": "AdamW",
                "args": {"lr": 1e-2, "weight_decay": 5e-2},
            },
            "lr_scheduler": {
                "type": "ReduceLROnPlateau",
                "args": {"factor": 0.5, "patience": 5},
            },
        }

    @staticmethod
    def suggest_config(trial: Trial) -> dict:
        config = UNet.default_config()
        config["reconstructor"]["args"].update(
            num_layers=trial.suggest_int("num_layers", 3, 4),
            min_channels=trial.suggest_int("min_channels", 16, 32, step=4),
            kernel_size=trial.suggest_int("kernel_size", 3, 33, step=2),
            min_dilation=trial.suggest_int("min_dilation", 1, 2),
            dilation_rate=trial.suggest_int("dilation_rate", 1, 3),
            out_conv_ext=trial.suggest_categorical("out_conv_ext", [True, False]),
        )
        return config
