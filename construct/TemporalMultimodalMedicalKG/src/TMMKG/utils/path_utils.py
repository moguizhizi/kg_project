from pathlib import Path
import re
import json


def get_last_dir_name(path: str) -> str:
    """
    Get the last directory or file name from a path.
    """
    return Path(path).name


def build_pipeline_paths(result_dir: str):
    """
    自动构建 pipeline 所有输出路径
    """

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    prefix = result_dir.name  # result1

    return {
        "normalized": result_dir / f"{prefix}.normalized.xlsx",
        "attr_facts": result_dir / f"{prefix}_attribute_facts.jsonl",
        "entity_facts": result_dir / f"{prefix}_entity_facts.jsonl",
        "duckdb_attr": result_dir / "attribute_facts.duckdb",
        "duckdb_entity": result_dir / "entity_facts.duckdb",
    }


def sheet_to_result_dir(sheet_name: str, base_dir: str) -> Path:
    """
    Result 11 -> result11
    """

    match = re.search(r"\d+", sheet_name)

    if not match:
        raise ValueError(f"Cannot extract number from sheet name: {sheet_name}")

    number = match.group()

    return Path(base_dir) / f"result{number}"


def save_no_candidates(dis, output_file="no_candidates.jsonl"):
    """将没有候选的疾病名追加到 JSONL 文件"""
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"disease_name": dis}, ensure_ascii=False) + "\n")
