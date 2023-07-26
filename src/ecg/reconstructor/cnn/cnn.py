"""ECG reconstruction models using CNN's."""

from collections.abc import Sequence

from optuna import Trial
from torch import nn, Tensor

from ..reconstructor import Reconstructor
from ..unet.unet import DoubleConv

# class StackedCNN(Reconstructor):
#     """An ECG reconstruction model that stacks CNN layers."""
#     max_batch_size = 1024
#     def __init__(
#         self,
#         in_leads: Sequence[int],
#         out_leads: Sequence[int],
#         num_layers: int,
#         base_channels: int,
#     ) -> None:
#         """
#         Initializes the model.

#         Args:
#             in_leads: Indices of the input leads.
#             out_leads: Indices of the output leads.
#             num_layers: Number of convolutional layers.
#             base_channels: If `num_layers > 1`, the number of output channels in the
#               first convolutional layer.
#         """
#         super().__init__()
#         self.layers = nn.ModuleList()
#         if num_layers == 1:
#             self.layers.append(
#                 nn.Conv1d(
#                     len(in_leads),
#                     len(out_leads),
#                     kernel_size=3,
#                     padding="same",
#                 )
#             )
#         else:
#             self.layers.append(
#                 nn.Conv1d(
#                     len(in_leads),
#                     base_channels,
#                     kernel_size=3,
#                     padding="same",
#                 )
#             )
#             for i in range(num_layers - 1):
#                 out_channels = (
#                     len(out_leads) if i == num_layers - 2 else base_channels * (i + 2)
#                 )
#                 self.layers.append(nn.LeakyReLU(inplace="True"))
#                 self.layers.append(
#                     nn.Conv1d(
#                         base_channels * (i + 1),
#                         out_channels,
#                         kernel_size=3,
#                         padding="same",
#                     )
#                 )

#     def forward(self, inputs: Tensor) -> Tensor:
#         outputs = inputs
#         for layer in self.layers:
#             outputs = layer(outputs)
#         return outputs

#     @staticmethod
#     def default_config() -> dict:
#         return {
#             "reconstructor": {
#                 "args": {
#                     "num_layers": 1,
#                     "base_channels": 8,
#                 }
#             },
#             "optimizer": {
#                 "type": "AdamW",
#                 "args": {"lr": 1e-2, "weight_decay": 5e-2},
#             },
#             "lr_scheduler": {
#                 "type": "ReduceLROnPlateau",
#                 "args": {"factor": 0.5, "patience": 5},
#             },
#         }

#     @staticmethod
#     def suggest_config(trial: Trial) -> dict:
#         config = StackedCNN.default_config()
#         config["reconstructor"]["args"].update(
#             num_layers=trial.suggest_int("num_layers", 1, 6)
#         )

#         return config


class StackedCNN(Reconstructor):
    """An ECG reconstruction model that stacks CNN layers."""
    max_batch_size = 16
    def __init__(
        self,
        in_leads: Sequence[int],
        out_leads: Sequence[int],
        num_layers: int,
        base_channels: int,
        kernel_size: int,
        dilation_scale: int = 2,
    ) -> None:
        """
        Initializes the model.

        Args:
            in_leads: Indices of the input leads.
            out_leads: Indices of the output leads.
            num_layers: Number of convolutional layers.
            base_channels: If `num_layers > 1`, the number of output channels in the
              first convolutional layer.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        # if num_layers == 1:
        #     self.layers.append(
        #         nn.Conv1d(
        #             len(in_leads),
        #             len(out_leads),
        #             kernel_size=3,
        #             padding="same",
        #         )
        #     )
        # else:
        self.layers.append(
            nn.Conv1d(
                len(in_leads),
                base_channels,
                kernel_size=3,
                padding="same",
            )
        )
        out_channels = base_channels
        for i in range(num_layers):
            # out_channels = (
            #     len(out_leads) if i == num_layers - 2 else base_channels * (i + 2)
            # )
            out_channels = base_channels * (i + 2)
            dilation = (dilation_scale)**(2*i)
            self.layers.append(DoubleConv(base_channels * (i + 1), 
                                            out_channels,
                                            kernel_size=kernel_size,
                                            dilation=dilation,
                                            dilation_scale=dilation_scale))
        self.layers.append(nn.Conv1d(out_channels, len(out_leads), kernel_size=1))


    def forward(self, inputs: Tensor) -> Tensor:
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    @staticmethod
    def default_config() -> dict:
        return {
            "reconstructor": {
                "args": {
                    "num_layers": 1,
                    "base_channels": 8,
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
        config = StackedCNN.default_config()
        config["reconstructor"]["args"].update(
            num_layers=trial.suggest_int("num_layers", 1, 4),
            base_channels=trial.suggest_int("base_channels", 8, 64, step=8),
            kernel_size=trial.suggest_int("kernel_size", 3, 33, step=2),
            dilation_scale=trial.suggest_int("dilation_scale", 1, 3),
        )

        return config
