from pathlib import Path
import re
import json


def get_last_dir_name(path: str) -> str:
    """
    Get the last directory or file name from a path.
    """
    return Path(path).name


def build_pipeline_paths(result_dir: str, parquet_dir: str, sheet_name: str):
    """
    自动构建 pipeline 所有输出路径
    """

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    prefix = result_dir.name  # result1

    parquet_dir = Path(parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    safe_sheet = safe_filename(sheet_name)

    return {
        "normalized": result_dir / f"{prefix}.normalized.xlsx",
        "attr_facts": result_dir / f"{prefix}_attribute_facts.jsonl",
        "entity_facts": result_dir / f"{prefix}_entity_facts.jsonl",
        "duckdb_attr": result_dir / "attribute_facts.duckdb",
        "duckdb_entity": result_dir / "entity_facts.duckdb",
        "parquet": parquet_dir / f"{safe_sheet}.parquet",
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


def safe_filename(name: str) -> str:
    """
    将任意sheet名转换为安全文件名

    Result 1      -> result_1
    游戏结果(最终) -> 游戏结果_最终
    A/B Test     -> a_b_test
    """

    name = name.strip().lower()

    # 把所有非 字母/数字/中文 替换成 _
    name = re.sub(r"[^\w\u4e00-\u9fff]+", "_", name)

    # 去掉多余 _
    name = re.sub(r"_+", "_", name).strip("_")

    return name
