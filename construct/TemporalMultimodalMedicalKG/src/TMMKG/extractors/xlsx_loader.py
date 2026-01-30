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
from typing import List, Dict, Optional, Union

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
# 字段与数据清洗
# =========================


def normalize_columns(
    df: pd.DataFrame,
    column_mapping: Optional[Dict[str, str]] = None,
    strip_whitespace: bool = True,
) -> pd.DataFrame:
    """
    统一字段名（别名 / 空格 / 全角问题）
    """
    df = df.copy()

    if strip_whitespace:
        old_cols = list(df.columns)
        df.columns = [str(c).strip() for c in df.columns]
        if old_cols != list(df.columns):
            logger.info("Stripped whitespace from column names")

    if column_mapping:
        logger.info(f"Applying column mapping: {column_mapping}")
        df = df.rename(columns=column_mapping)

    logger.debug(f"Final columns: {list(df.columns)}")
    return df


def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    删除全为空的行
    """
    before = len(df)
    df = df.dropna(how="all")
    after = len(df)

    if before != after:
        logger.info(f"Dropped {before - after} empty rows")

    return df


def fill_na_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    保持 DataFrame 内部为 NaN，
    在导出 records 时再转为 None
    """
    logger.info("Keeping NA values as NaN (will normalize at export stage)")
    return df


def parse_date_fields(
    df: pd.DataFrame,
    date_fields: List[str],
    date_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    将日期字段统一转为 ISO 格式字符串
    """
    df = df.copy()

    for field in date_fields:
        if field not in df.columns:
            logger.warning(f"Date field not found, skip: {field}")
            continue

        logger.info(f"Parsing date field: {field}")
        df[field] = pd.to_datetime(
            df[field],
            format=date_format,
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")

    return df


def split_multi_value_fields(
    df: pd.DataFrame,
    fields: List[str],
    sep: str = ",",
) -> pd.DataFrame:
    """
    拆分多值字段（如 疾病：A,B,C）
    注意：这里只做字符串 → list，不做行展开
    """
    df = df.copy()

    for field in fields:
        if field not in df.columns:
            logger.warning(f"Multi-value field not found, skip: {field}")
            continue

        logger.info(f"Splitting multi-value field: {field}")

        def _split(val):
            if pd.isna(val):
                return []
            if isinstance(val, str):
                return [v.strip() for v in val.split(sep) if v.strip()]
            return [val]

        df[field] = df[field].apply(_split)

    return df


# =========================
# 校验
# =========================


def validate_schema(df: pd.DataFrame, required_fields: List[str]) -> None:
    """
    校验必需字段是否存在
    """
    logger.info(f"Validating required fields: {required_fields}")

    missing = [f for f in required_fields if f not in df.columns]
    if missing:
        logger.error(f"Missing required fields: {missing}")
        raise ValueError(f"Missing required fields: {missing}")

    logger.info("Schema validation passed")


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

    if not df.empty:
        logger.info(f"Sample row (first): {df.iloc[0].to_dict()}")

    # 校验（可选）
    # if required_fields:
    #     validate_schema(df, required_fields)

    # 日期解析（可选）
    # if date_fields:
    #     df = parse_date_fields(df, date_fields)

    # 多值字段拆分（可选）
    # if multi_value_fields:
    #     df = split_multi_value_fields(df, multi_value_fields)

    # 🔥 核心：导出 records，并把 NaN → None
    records = df.where(pd.notnull(df), None).to_dict(orient="records")

    logger.info(f"Generated {len(records)} records")
    return records


if __name__ == "__main__":
    xlsx_path = "/home/temp/dataset/temp.xlsx"

    records = xlsx_to_records(
        path=xlsx_path,
        sheet_name="Sheet1",
    )

    logger.info(f"Total records loaded: {len(records)}")

    if records:
        logger.info(f"First record: {records[0]}")
