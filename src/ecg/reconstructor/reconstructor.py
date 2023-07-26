"""Defines the `Reconstructor` class, the base class for reconstruction models."""

from abc import abstractmethod

from optuna import Trial
from torch import nn, Tensor

from ecg import reconstructor as reconstructor_module


class Reconstructor(nn.Module):
    """The base class for reconstruction models."""

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """Reconstructs signals of the output leads given signals of the input leads."""

    @staticmethod
    @abstractmethod
    def default_config() -> dict:
        """Returns the default training configuration."""

    @staticmethod
    @abstractmethod
    def suggest_config(trial: Trial) -> dict:
        """Suggests a training configuration for the given trial."""

    @staticmethod
    def resolve_type(str_or_type: str | type) -> type["Reconstructor"]:
        """Resolves a reconstructor type from a string or a type object."""
        if isinstance(str_or_type, str):
            path_to_reconstructor_type = str_or_type.split(".")
            reconstructor_type = reconstructor_module
            for name in path_to_reconstructor_type:
                reconstructor_type = getattr(reconstructor_type, name)
            assert isinstance(
                reconstructor_type, type
            ), f"Expected a type object, but got {type(reconstructor_type)}"
        else:
            reconstructor_type = str_or_type
        assert issubclass(
            reconstructor_type, Reconstructor
        ), f"Expected a subclass of Reconstructor, but got {reconstructor_type}"
        return reconstructor_type
