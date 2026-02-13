import pandas as pd
import time
import logging
from pathlib import Path
from typing import Optional, Any
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_parquet_columns(parquet_path: str) -> list[str]:
    """
    获取 parquet 列名
    超快，不加载数据

    Returns:
        list[str]: parquet 文件的列名列表
    """
    schema = pq.read_schema(parquet_path)
    return schema.names


def get_unique_values_per_column(parquet_path: str) -> dict[str, list[Any]]:
    """
    返回:
        dict[str, list[Any]]

        {
            column1: [...unique values...],
            column2: [...]
        }
    """
    parquet_file = pq.ParquetFile(parquet_path)

    result: dict[str, list[Any]] = {}

    for col in parquet_file.schema.names:
        column_data = parquet_file.read(columns=[col]).column(col)
        unique_values = column_data.unique().to_pylist()

        result[col] = unique_values

    return result
