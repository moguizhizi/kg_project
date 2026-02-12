import pandas as pd
import time
import logging
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
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
