from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def extract_llm_name(model_path: PathLike, llm_root: PathLike) -> str:
    model_path = Path(model_path).resolve()
    llm_root = Path(llm_root).resolve()

    try:
        return str(model_path.relative_to(llm_root))
    except ValueError as e:
        raise ValueError(f"Invalid LLM path: {model_path}, root: {llm_root}") from e


def build_llm_path(llm_name: str, llm_root: Union[str, PathLike]) -> str:
    """
    Build absolute local path for an LLM model.

    Returns:
        str: absolute path string
    """
    llm_root = Path(llm_root).resolve()
    return str(llm_root / llm_name)


def validate_llm_path(model_path: PathLike, llm_root: PathLike) -> None:
    model_path = Path(model_path).resolve()
    llm_root = Path(llm_root).resolve()

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if not model_path.is_dir():
        raise ValueError(f"{model_path} is not a directory")

    try:
        model_path.relative_to(llm_root)
    except ValueError:
        raise ValueError(f"{model_path} is not under LLM root {llm_root}")
