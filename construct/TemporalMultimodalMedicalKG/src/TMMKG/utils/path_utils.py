from pathlib import Path


def get_last_dir_name(path: str) -> str:
    """
    Get the last directory or file name from a path.
    """
    return Path(path).name
