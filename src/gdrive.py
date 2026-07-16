r"""
gdrive.py
=========
Locate files synced by the **Google Drive for Desktop** app, so this project can
read your dataset the same way the Colab notebook did with `drive.mount(...)` —
except locally, inside VSCode, with no mounting step.

Google Drive for Desktop exposes "My Drive" as a normal folder on disk. Where it
lives depends on the OS:

  Windows : G:\My Drive   (the drive letter is configurable; often G:, sometimes H:)
  macOS   : ~/Library/CloudStorage/GoogleDrive-<you@gmail.com>/My Drive   (current)
            /Volumes/GoogleDrive/My Drive                                  (older app)
  Linux   : no official client — typically an rclone mount you point to manually

`resolve_drive_root()` tries to auto-detect "My Drive". You can always override it
explicitly in config.py (recommended once you know your path) or via the
GDRIVE_ROOT environment variable.
"""

from __future__ import annotations

import os
import glob
import string
from pathlib import Path


def _candidate_roots() -> list[Path]:
    """Return plausible 'My Drive' locations for the current OS, best guesses first."""
    candidates: list[Path] = []
    home = Path.home()

    # macOS (current Google Drive app) -> ~/Library/CloudStorage/GoogleDrive-*/My Drive
    cloud = home / "Library" / "CloudStorage"
    if cloud.exists():
        for d in sorted(cloud.glob("GoogleDrive-*")):
            candidates.append(d / "My Drive")

    # macOS (older app)
    candidates.append(Path("/Volumes/GoogleDrive/My Drive"))

    # Windows -> scan drive letters for "<X>:\My Drive"
    if os.name == "nt":
        for letter in "GHIJKLMNOPQRSTUVWXYZEF":
            candidates.append(Path(f"{letter}:/My Drive"))
            candidates.append(Path(f"{letter}:/"))  # some setups put files at the root

    # Linux / rclone common spots
    candidates.append(home / "GoogleDrive" / "My Drive")
    candidates.append(home / "gdrive" / "My Drive")
    candidates.append(home / "GoogleDrive")

    return candidates


def resolve_drive_root(explicit: str | None = None) -> Path:
    """
    Resolve the Google Drive 'My Drive' root folder.

    Priority:
      1. `explicit` argument (set DRIVE_ROOT in config.py)
      2. GDRIVE_ROOT environment variable
      3. auto-detection across common OS locations

    Raises a helpful error if nothing is found.
    """
    # 1 + 2: explicit override
    override = explicit or os.environ.get("GDRIVE_ROOT")
    if override:
        p = Path(override).expanduser()
        if p.exists():
            return p
        raise FileNotFoundError(
            f"Configured Google Drive root does not exist:\n  {p}\n"
            "Open config.py and set DRIVE_ROOT to the correct 'My Drive' path, "
            "or set the GDRIVE_ROOT environment variable."
        )

    # 3: auto-detect
    for c in _candidate_roots():
        if c.exists():
            return c

    tried = "\n  ".join(str(c) for c in _candidate_roots())
    raise FileNotFoundError(
        "Could not auto-detect your Google Drive folder.\n"
        "Make sure the 'Google Drive for Desktop' app is installed and running, "
        "then set DRIVE_ROOT explicitly in config.py.\n\n"
        f"Locations I checked:\n  {tried}"
    )


def resolve_dataset_path(relative_path: str, explicit_root: str | None = None) -> Path:
    """
    Build the absolute path to the dataset file inside Google Drive.

    `relative_path` is the path *within* My Drive, e.g. 'voltammetry_dataset_aligned.pkl'
    or 'data/voltammetry_dataset_aligned.pkl'.
    """
    root = resolve_drive_root(explicit_root)
    full = (root / relative_path).expanduser()
    if not full.exists():
        # Try a shallow search so a misremembered subfolder still resolves.
        name = Path(relative_path).name
        matches = glob.glob(str(root / "**" / name), recursive=True)
        if len(matches) == 1:
            return Path(matches[0])
        if len(matches) > 1:
            joined = "\n  ".join(matches[:10])
            raise FileNotFoundError(
                f"Found multiple files named '{name}' under {root}.\n"
                f"Set DATASET_RELATIVE_PATH in config.py to the exact one:\n  {joined}"
            )
        raise FileNotFoundError(
            f"Dataset not found:\n  {full}\n"
            f"(also searched recursively under {root} for '{name}' and found nothing)\n"
            "Check DATASET_RELATIVE_PATH in config.py."
        )
    return full


if __name__ == "__main__":
    # Quick diagnostic: `python -m src.gdrive`
    try:
        print("Drive root:", resolve_drive_root())
    except FileNotFoundError as e:
        print(e)
