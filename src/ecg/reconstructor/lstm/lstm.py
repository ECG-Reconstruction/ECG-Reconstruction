"""ECG reconstruction models using LSTM's."""

from collections.abc import Sequence

from optuna import Trial
from torch import nn, Tensor

from ..reconstructor import Reconstructor
from ..unet.unet import DoubleConv


class SimpleLSTM(Reconstructor):
    """An ECG reconstruction model with a single LSTM layer."""

    def __init__(
        self,
        in_leads: Sequence[int],
        out_leads: Sequence[int],
        num_layers: int,
        bidirectional: bool,
    ) -> None:
        """
        Initializes the model.

        Args:
            in_leads: Indices of the input leads.
            out_leads: Indices of the output leads.
            num_layers: The number of LSTM layers.
            bidirectional: Whether the LSTM layers are bidirectional.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=len(in_leads),
            hidden_size=len(out_leads),
            num_layers=num_layers,
            batch_first=False,
            bidirectional=bidirectional,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, _ = self.lstm(inputs.transpose(1, 2))
        return outputs.transpose(1, 2)

    @staticmethod
    def default_config() -> dict:
        return {
            "reconstructor": {
                "args": {
                    "num_layers": 1,
                    "bidirectional": False,
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
        config = SimpleLSTM.default_config()
        config["reconstructor"]["args"].update(
            num_layers=trial.suggest_int("num_layers", 1, 8),
            num_hidden=trial.suggest_int("num_hidden", 8, 64),
            bidirectional=trial.suggest_categorical("bidirectional", [True, False]),
        )
        return config


class LSTM(Reconstructor):
    """An ECG reconstruction model with an LSTM layer, with additional activation and
    linear layers."""

    max_batch_size = 128

    def __init__(
        self,
        in_leads: Sequence[int],
        out_leads: Sequence[int],
        num_layers: int,
        num_hidden: int,
        bidirectional: bool,
    ) -> None:
        """
        Initializes the model.

        Args:
            in_leads: Indices of the input leads.
            out_leads: Indices of the output leads.
            num_layers: The number of LSTM layers.
            bidirectional: Whether the LSTM layers are bidirectional.
        """
        super().__init__()
        # num_hidden = 16
        self.lstm = nn.LSTM(
            input_size=len(in_leads),
            hidden_size=num_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.activation = nn.LeakyReLU()
        self.linear = nn.Linear(
            2 * num_hidden if bidirectional else num_hidden, len(out_leads)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(inputs.transpose(1, 2))
        outputs = self.activation(outputs)
        outputs = self.linear(outputs)
        return outputs.transpose(1, 2)

    @staticmethod
    def default_config() -> dict:
        return SimpleLSTM.default_config()

    @staticmethod
    def suggest_config(trial: Trial) -> dict:
        return SimpleLSTM.suggest_config(trial)


# class CNNLSTM(Reconstructor):
#     """An ECG reconstruction model with an LSTM layer, with additional activation and
#     linear layers."""
#     max_batch_size = 16
#     def __init__(
#         self,
#         in_leads: Sequence[int],
#         out_leads: Sequence[int],
#         num_lstm_layers: int,
#         bidirectional: bool,
#         num_cnn_layers: int,
#         num_hidden: int = 16,
#         kernel_size: int = 3,
#     ) -> None:
#         """
#         Initializes the model.

#         Args:
#             in_leads: Indices of the input leads.
#             out_leads: Indices of the output leads.
#             num_layers: The number of LSTM layers.
#             bidirectional: Whether the LSTM layers are bidirectional.
#         """
#         super().__init__()
#         self.cnn_layers = nn.ModuleList()
#         if num_cnn_layers == 1:
#             self.layers.append(
#                 nn.Conv1d(
#                     len(in_leads),
#                     len(out_leads),
#                     kernel_size=kernel_size,
#                     padding="same",
#                 )
#             )
#         else:
#             self.cnn_layers.append(
#                 nn.Conv1d(
#                     len(in_leads),
#                     num_hidden,
#                     kernel_size=kernel_size,
#                     padding="same",
#                 )
#             )
#             for i in range(num_cnn_layers - 1):
#                 cnn_out_channels = num_hidden * (i + 2)
#                 self.cnn_layers.append(nn.LeakyReLU(inplace="True"))
#                 self.cnn_layers.append(
#                     nn.Conv1d(
#                         num_hidden * (i + 1),
#                         cnn_out_channels,
#                         kernel_size=kernel_size,
#                         padding="same",
#                     )
#             )
#         self.lstm = nn.LSTM(
#             input_size=cnn_out_channels,
#             hidden_size=cnn_out_channels,
#             num_layers=num_lstm_layers,
#             batch_first=True,
#             bidirectional=bidirectional,
#         )
#         self.activation = nn.LeakyReLU()
#         self.linear = nn.Linear(
#             2 * cnn_out_channels if bidirectional else cnn_out_channels, len(out_leads)
#         )

#     def forward(self, inputs: Tensor) -> Tensor:
#         for layer in self.cnn_layers:
#             inputs = layer(inputs)
#         self.lstm.flatten_parameters()
#         outputs, _ = self.lstm(inputs.transpose(1, 2))
#         outputs = self.activation(outputs)
#         outputs = self.linear(outputs)
#         return outputs.transpose(1, 2)

#     @staticmethod
#     def default_config() -> dict:
#         return SimpleLSTM.default_config()

#     @staticmethod
#     def suggest_config(trial: Trial) -> dict:
#         config = SimpleLSTM.default_config()
#         config["reconstructor"]["args"].update(
#             num_cnn_layers=trial.suggest_int("num_layers", 1, 8),
#             num_lstm_layers=trial.suggest_int("num_layers", 1, 8),
#             num_hidden = trial.suggest_int("num_hidden", 8, 32, step=32),
#             bidirectional=trial.suggest_categorical("bidirectional", [True, False]),
#         )
#         return config


class CNNLSTM(Reconstructor):
    """An ECG reconstruction model with an LSTM layer, with additional activation and
    linear layers."""

    max_batch_size = 16

    def __init__(
        self,
        in_leads: Sequence[int],
        out_leads: Sequence[int],
        num_lstm_layers: int,
        num_hidden: int,
        bidirectional: bool,
        num_cnn_layers: int,
        base_channels: int = 16,
        kernel_size: int = 3,
        dilation_scale: int = 2,
    ) -> None:
        """
        Initializes the model.

        Args:
            in_leads: Indices of the input leads.
            out_leads: Indices of the output leads.
            num_layers: The number of LSTM layers.
            bidirectional: Whether the LSTM layers are bidirectional.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Conv1d(
                len(in_leads),
                base_channels,
                kernel_size=3,
                padding="same",
            )
        )
        out_channels = base_channels
        for i in range(num_cnn_layers):
            out_channels = base_channels * (i + 2)
            dilation = (dilation_scale) ** (2 * i)
            self.layers.append(
                DoubleConv(
                    base_channels * (i + 1),
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dilation_scale=dilation_scale,
                )
            )
        self.lstm = nn.LSTM(
            input_size=out_channels,
            hidden_size=num_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.activation = nn.LeakyReLU()
        self.linear = nn.Linear(
            2 * num_hidden if bidirectional else num_hidden, len(out_leads)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer(inputs)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(inputs.transpose(1, 2))
        outputs = self.activation(outputs)
        outputs = self.linear(outputs)
        return outputs.transpose(1, 2)

    @staticmethod
    def default_config() -> dict:
        config = SimpleLSTM.default_config()
        del config["reconstructor"]["args"]["num_layers"]
        return config

    @staticmethod
    def suggest_config(trial: Trial) -> dict:
        config = CNNLSTM.default_config()
        config["reconstructor"]["args"].update(
            num_cnn_layers=trial.suggest_int("num_cnn_layers", 1, 4),
            num_lstm_layers=trial.suggest_int("num_lstm_layers", 1, 4),
            base_channels=trial.suggest_int("base_channels", 8, 32, step=8),
            num_hidden=trial.suggest_int("num_hidden", 16, 64, step=16),
            bidirectional=trial.suggest_categorical("bidirectional", [True, False]),
            kernel_size=trial.suggest_int("kernel_size", 3, 17, step=2),
            dilation_scale=trial.suggest_int("dilation_scale", 1, 3),
        )
        return config
