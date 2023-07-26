"""ECG reconstruction models using linear layers."""

from collections.abc import Sequence

from optuna import Trial
from torch import nn, Tensor

from ..reconstructor import Reconstructor


class Linear(Reconstructor):
    """An ECG reconstruction model with a single linear layer."""

    def __init__(self, in_leads: Sequence[int], out_leads: Sequence[int]) -> None:
        """
        Initializes the model.

        Args:
            in_leads: Indices of the input leads.
            out_leads: Indices of the output leads.
        """
        super().__init__()
        self.linear = nn.Conv1d(len(in_leads), len(out_leads), kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.linear(inputs)

    @staticmethod
    def default_config() -> dict:
        return {
            "reconstructor": {
                "args": {},
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
        return Linear.default_config()
