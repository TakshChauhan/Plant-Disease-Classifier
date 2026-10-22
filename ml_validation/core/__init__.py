"""Core backend, preprocessing, and metrics utilities for ml_validation."""
import importlib.util
import sys
from pathlib import Path

_core_dir = Path(__file__).resolve().parent
for _m in ("preprocessing", "backend", "metrics"):
    _m_file = _core_dir / f"{_m}.py"
    _m_name = f"ml_validation.core.{_m}"
    if _m_name not in sys.modules and _m_file.is_file():
        _spec = importlib.util.spec_from_file_location(_m_name, _m_file)
        if _spec and _spec.loader:
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules[_m_name] = _mod
            _spec.loader.exec_module(_mod)
