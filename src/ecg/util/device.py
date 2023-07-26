"""
Functions that set (manually or automatically) and get the device used for PyTorch
computation.
"""

import logging
from typing import Optional

import torch

_logger = logging.getLogger(__name__)

_device: Optional[torch.device] = None


def set_device(device: torch.device) -> None:
    """
    Sets the device used for PyTorch computation.

    If a device has already been set and it is different from the device to set, raises
    a `RuntimeError`.
    """
    global _device  # pylint: disable=global-statement,invalid-name
    if _device is None:
        _logger.debug("Manually set device to %s", device)
        _device = device
    elif _device == device:
        _logger.warning("Manually set device to %s while it is already set", _device)
    else:
        raise RuntimeError(
            f"Cannot set device to {device} because it is already set to {_device}"
        )


def get_device() -> torch.device:
    """
    Returns the device used for PyTorch computation.

    If a device has been set, returns that device. Otherwise, tries to set a device
    automatically and returns it.
    """
    global _device  # pylint: disable=global-statement,invalid-name
    if _device is not None:
        return _device
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        _device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        try:
            import intel_extension_for_pytorch  # pylint: disable=import-outside-toplevel,unused-import
        except ImportError:
            _device = torch.device("cpu")
        else:
            _device = torch.device("xpu")
    _logger.debug("Automatically set device to %s", _device)
    return _device
