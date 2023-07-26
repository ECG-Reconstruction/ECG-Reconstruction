"""Utility functions for handling ECG leads."""


def get_lead_names() -> list[str]:
    """Returns the names of the 12 ECG leads."""
    return ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
