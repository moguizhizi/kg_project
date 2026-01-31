# services/model_resolver.py

import os
from pathlib import Path
from typing import Optional


def resolve_model_path(
    model_name: str,
    model_root: Optional[str] = None,
) -> str:
    """
    Resolve a model name to a local path if available, otherwise return the name itself.
    """
    if not model_root:
        return model_name

    candidate = Path(model_root) / model_name
    if candidate.exists() and candidate.is_dir():
        return str(candidate)

    return model_name
