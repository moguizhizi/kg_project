"""
xlsx_loader.py

职责：
- 从 XLSX 文件加载数据
- 做最基础、可复用的数据清洗
- 输出统一的 List[Dict] 结构，供三元组抽取使用

不负责：
- 三元组抽取
- 本体映射
- LLM 调用
"""

from __future__ import annotations

import os
from typing import List, Dict, Optional, Union

import pandas as pd


# =========================
# 基础加载
# =========================


def load_xlsx(
    path: str, sheet_name: Union[str, int, None] = 0, header: int = 0
) -> pd.DataFrame:
    """
    加载 XLSX 文件，返回原始 DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"XLSX file not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet_name, header=header)
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
        df.columns = [str(c).strip() for c in df.columns]

    if column_mapping:
        df = df.rename(columns=column_mapping)

    return df


def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    删除全为空的行
    """
    return df.dropna(how="all")


def fill_na_values(df: pd.DataFrame, fill_value: Optional[str] = None) -> pd.DataFrame:
    """
    统一 NaN / None
    """
    df = df.copy()
    return df.fillna(fill_value)


def parse_date_fields(
    df: pd.DataFrame, date_fields: List[str], date_format: Optional[str] = None
) -> pd.DataFrame:
    """
    将日期字段统一转为 ISO 格式字符串
    """
    df = df.copy()

    for field in date_fields:
        if field not in df.columns:
            continue

        df[field] = pd.to_datetime(
            df[field], format=date_format, errors="coerce"
        ).dt.strftime("%Y-%m-%d")

    return df


def split_multi_value_fields(
    df: pd.DataFrame, fields: List[str], sep: str = ","
) -> pd.DataFrame:
    """
    拆分多值字段（如 疾病：A,B,C）
    注意：这里只做字符串 → list，不做行展开
    """
    df = df.copy()

    for field in fields:
        if field not in df.columns:
            continue

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
    missing = [f for f in required_fields if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")


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
    df = load_xlsx(path, sheet_name=sheet_name)
    df = normalize_columns(df, column_mapping=column_mapping)
    df = drop_empty_rows(df)
    df = fill_na_values(df, fill_value=None)

    if required_fields:
        validate_schema(df, required_fields)

    if date_fields:
        df = parse_date_fields(df, date_fields)

    if multi_value_fields:
        df = split_multi_value_fields(df, multi_value_fields)

    return df.to_dict(orient="records")
