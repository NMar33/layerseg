"""Setup sys.path for binarizer imports.

Must be called before importing any binarizer modules.
"""
import sys
from pathlib import Path


_setup_done = False


def setup_binarizer_imports(project_root="."):
    """Add src/ to sys.path so bare imports work."""
    global _setup_done
    if _setup_done:
        return

    src_dir = str(Path(project_root) / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    _setup_done = True
