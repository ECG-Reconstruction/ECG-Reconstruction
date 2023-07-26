"""Utility functions for path handling."""

import os
from pathlib import Path
from typing import Optional


def get_project_root_dir() -> Path:
    """Returns the root directory of the `ecg-reconstruction` project."""
    # Because this file and its parent directories are in a git repository, we do not
    # expect any symbolic links.
    file_path = Path(__file__).resolve(strict=True)
    util_dir = file_path.parent
    assert util_dir.is_dir() and util_dir.name == "util"
    ecg_dir = util_dir.parent
    assert ecg_dir.is_dir() and ecg_dir.name == "ecg"
    src_dir = ecg_dir.parent
    assert src_dir.is_dir() and src_dir.name == "src"
    project_dir = src_dir.parent
    return project_dir


def resolve_path(
    path: str | os.PathLike, relative_to: Optional[str | os.PathLike] = None
) -> Path:
    """If `path` is relative, resolves it relative to the project root directory or
    the `relative_to` path."""
    path = Path(path)
    if not path.is_absolute():
        if relative_to is None:
            relative_to = get_project_root_dir()
        else:
            relative_to = Path(relative_to)
            if not relative_to.is_absolute():
                relative_to = get_project_root_dir() / relative_to
        path = relative_to / path
    return path.resolve(strict=True)
