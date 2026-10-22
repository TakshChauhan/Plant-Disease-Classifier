"""Independent ML Validation and Research Suite for the Plant Disease Classifier."""
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

__version__ = "1.0.0"

_PKG_DIR = Path(__file__).resolve().parent


class _VFSResilientFinder(importlib.abc.MetaPathFinder):
    """Ensure `ml_validation.*` imports succeed even on lazy FUSE mounts where readdir is unpopulated."""

    def find_spec(self, fullname: str, path=None, target=None):
        if not fullname.startswith("ml_validation."):
            return None
        rel = fullname[len("ml_validation."):].replace(".", "/")
        pkg_init = _PKG_DIR / rel / "__init__.py"
        if pkg_init.is_file():
            return importlib.util.spec_from_file_location(
                fullname, pkg_init, submodule_search_locations=[str(pkg_init.parent)]
            )
        mod_file = _PKG_DIR / f"{rel}.py"
        if mod_file.is_file():
            return importlib.util.spec_from_file_location(fullname, mod_file)
        return None


if not any(isinstance(f, _VFSResilientFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _VFSResilientFinder())
