"""
xlsx_loader.py

职责：
- 从 XLSX 文件加载数据
- 做最基础、可复用的数据清洗
- 输出统一的 List[Dict] 结构，供三元组抽取使用
"""

from __future__ import annotations

import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Union

import pandas as pd

from TMMKG.extractors.dataframe_processor import (
    drop_empty_rows,
    fill_na_values,
    normalize_columns,
    normalize_record_nans,
    parse_date_fields,
    split_multi_value_fields,
    validate_schema,
)
from TMMKG.utils.xlsx_utils import build_column_mapping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
ONTOLOGY_MAPPINGS_DIR = BASE_DIR / "utils" / "ontology_mappings"
HOME_BASED_USER_TRAINING = (
    BASE_DIR / "utils" / "entity_registry" / "home_based_user_training"
)


# =========================
# 基础加载
# =========================


def load_xlsx(
    path: str,
    sheet_name: Union[str, int, None] = 0,
    header: int = 0,
) -> pd.DataFrame:
    """
    加载 XLSX 文件，返回原始 DataFrame
    """
    logger.info(f"Loading XLSX file: {path}, sheet={sheet_name}")

    if not os.path.exists(path):
        logger.error(f"XLSX file not found: {path}")
        raise FileNotFoundError(f"XLSX file not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet_name, header=header)

    logger.info(f"Loaded DataFrame with shape {df.shape}")
    return df


# =========================
# 统一出口
# =========================


def xlsx_to_records(
    path: str,
    sheet_name: Union[str, int, None] = 0,
    column_mapping: Optional[Dict[str, str]] = None,
    date_fields: Optional[List[str]] = None,
    multi_value_fields: Optional[List[str]] = None,
    required_fields: Optional[List[str]] = None,
) -> List[Dict]:
    """
    XLSX → 干净的 records（List[Dict]）
    """
    logger.info("Starting XLSX to records pipeline")

    df = load_xlsx(path, sheet_name=sheet_name)
    df = normalize_columns(df, column_mapping=column_mapping)
    df = drop_empty_rows(df)
    df = fill_na_values(df)

    # 校验（可选）
    if required_fields:
        validate_schema(df, required_fields)

    # 日期解析（可选）
    if date_fields:
        df = parse_date_fields(df, date_fields)

    # 多值字段拆分（可选）
    if multi_value_fields:
        df = split_multi_value_fields(df, multi_value_fields)

    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    records = normalize_record_nans(records)

    logger.info(f"Generated {len(records)} records")
    return records


def records_to_xlsx(
    records: List[Dict],
    output_path: str,
    sheet_name: str = "records",
) -> None:
    """
    将 List[Dict] records 保存为 XLSX

    - list / dict 字段会序列化为 JSON 字符串
    """
    if not records:
        logger.warning("No records to save, skipping XLSX export")
        return

    logger.info(f"Saving {len(records)} records to XLSX: {output_path}")

    def _serialize(val):
        if isinstance(val, (list, dict)):
            return json.dumps(val, ensure_ascii=False)
        return val

    df = pd.DataFrame(records)

    # 统一序列化复杂字段
    for col in df.columns:
        df[col] = df[col].apply(_serialize)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info("XLSX export completed")


if __name__ == "__main__":
    xlsx_path = "/home/temp/dataset/temp.xlsx"

    with open(os.path.join(ONTOLOGY_MAPPINGS_DIR, "entity_type2label.json"), "r") as f:
        ENTITY_TYPE_2_LABEL = json.load(f)

    with open(os.path.join(ONTOLOGY_MAPPINGS_DIR, "prop2label.json"), "r") as f:
        PROP_2_LABEL = json.load(f)

    with open(os.path.join(HOME_BASED_USER_TRAINING, "column_mapping.json"), "r") as f:
        COLUMN_MAPPING = json.load(f)

    date_fields = [COLUMN_MAPPING["训练日期"]]
    multi_value_fields = [COLUMN_MAPPING["疾病"]]

    records = xlsx_to_records(
        path=xlsx_path,
        sheet_name="Sheet1",
        date_fields=date_fields,
        # multi_value_fields=multi_value_fields,
        column_mapping=COLUMN_MAPPING,
    )

    logger.info(f"Total records loaded: {len(records)}")

    if records:
        logger.info(f"First record: {records[3]}")

    output_xlsx = "/home/temp/dataset/temp.normalized.xlsx"
    records_to_xlsx(records, output_xlsx)
