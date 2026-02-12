from typing import List, Dict, Optional, Union
import logging
import pandas as pd
import time
from pathlib import Path
from typing import List, Optional

from TMMKG.extractors.dataframe_processor import (
    drop_empty_rows,
    fill_na_values,
    normalize_columns,
    normalize_record_nans,
    parse_date_fields,
    split_multi_value_fields,
    validate_schema,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_parquet_as_dataframe(
    parquet_path: str,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    高速加载 parquet 为 pandas.DataFrame

    参数:
        parquet_path: parquet 文件或目录
        columns: 可选，只读取指定列（强烈建议大数据时使用）

    返回:
        pd.DataFrame
    """

    parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    logger.info(f"Loading parquet -> {parquet_path}")

    start = time.perf_counter()

    df = pd.read_parquet(
        parquet_path,
        engine="pyarrow",
        columns=columns,
    )

    logger.info(
        f"Loaded dataframe | rows={len(df)} cols={len(df.columns)} "
        f"time={time.perf_counter()-start:.2f}s"
    )

    return df


def parquet_to_records(
    path: str,
    column_mapping: Optional[Dict[str, str]] = None,
    date_fields: Optional[List[str]] = None,
    multi_value_fields: Optional[List[str]] = None,
    required_fields: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Parquet → 标准化 records（List[Dict]）

    Pipeline:
        Parquet文件
            ↓
        DataFrame加载
            ↓
        列名规范化 / 空行清理 / 缺失值填充
            ↓
        Schema校验（可选）
            ↓
        日期字段解析（可选）
            ↓
        多值字段拆分（可选）
            ↓
        输出 records（适用于KG / JSON / DB写入）

    Args:
        path: parquet 文件路径
        column_mapping: 列重命名映射
        date_fields: 需要解析为日期的字段
        multi_value_fields: 需要拆分的多值字段
        required_fields: 必须存在的字段（schema校验）

    Returns:
        List[Dict]: 已清洗的结构化数据
    """
    logger.info("Starting Parquet to records pipeline")

    df = load_parquet_as_dataframe(path)
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
