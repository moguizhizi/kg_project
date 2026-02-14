import pandas as pd
import logging
from typing import List, Dict, Optional, Union
import math


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def normalize_multilabel_series(series: pd.Series) -> pd.Series:
    """
    规范多标签字段：

    蓝色_ 红色 -> 蓝色_红色
    红色_蓝色 -> 蓝色_红色 （排序去重）
    """

    return series.str.split("_").apply(
        lambda parts: "_".join(sorted({p.strip() for p in parts if p and p.strip()}))
    )


def clean_dataframe(
    df: pd.DataFrame, multi_label_keywords: list | None = None
) -> pd.DataFrame:
    """
    高性能清洗函数

    当 multi_label_keywords=None 时：
        -> 不执行多标签规范化
    """

    df.columns = (
        df.columns.str.replace(
            r"[\u200b\u200c\u200d\ufeff]", "", regex=True
        )  # 去隐形字符
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    df = (
        df.fillna("")
        .astype(str)
        .apply(lambda col: col.str.strip())
        .replace(r"\s*_\s*", "_", regex=True)
        .replace(r"\s+", " ", regex=True)
        .replace(r"[\u200b\u200c\u200d\ufeff]", "", regex=True)
    )

    # 关键改动
    if not multi_label_keywords:
        return df

    # 防止有人传字符串，例如 "颜色"
    if isinstance(multi_label_keywords, str):
        multi_label_keywords = [multi_label_keywords]

    target_cols = [
        col for col in df.columns if any(k == col for k in multi_label_keywords)
    ]

    for col in target_cols:
        try:
            df[col] = normalize_multilabel_series(df[col])
        except Exception:
            pass

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


def fill_na_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    保持 DataFrame 内部为 NaN，
    在导出 records 时再转为 None
    """
    logger.info("Keeping NA values as NaN (will normalize at export stage)")
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
        missing_cols = set(column_mapping) - set(df.columns)
        if missing_cols:
            logger.warning(f"Columns not found in DataFrame: {missing_cols}")

        df = df.rename(columns=column_mapping)

    logger.debug(f"Final columns: {list(df.columns)}")
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


def normalize_record_nans(records: List[Dict]) -> List[Dict]:
    for r in records:
        for k, v in r.items():
            if isinstance(v, float) and math.isnan(v):
                r[k] = None
    return records
