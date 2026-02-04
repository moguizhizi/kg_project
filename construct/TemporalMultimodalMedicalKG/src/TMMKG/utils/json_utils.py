import json
import duckdb
import pandas as pd
from typing import Iterator, List, Optional
from typing import Iterable
from collections import defaultdict

from TMMKG.domains.home_based_user_training.table_triple_extractor import (
    extract_facts_from_records,
)
from TMMKG.extractors.xlsx_loader import xlsx_to_records
from TMMKG.meta_type import TypedFact
from TMMKG.sql_templates import ATTRIBUTE_FACT_SQL


def write_facts_jsonl(
    path: str,
    facts: Iterable[TypedFact],
) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for fact in facts:
            # tuple → list，JSON 友好
            f.write(json.dumps(list(fact), ensure_ascii=False))
            f.write("\n")


def iter_duckdb_query_df(
    query: str,
    columns: Optional[List[str]] = None,
    batch_size: int = 100_000,
    database: str = ":memory:",
) -> Iterator[pd.DataFrame]:
    """
    流式执行 DuckDB SQL 查询，按批返回 DataFrame

    参数:
        query: DuckDB SQL 查询语句
        columns: DataFrame 列名（None 则自动从 cursor.description 推断）
        batch_size: 每批 DataFrame 行数
        database: DuckDB 数据库（:memory: 或磁盘路径）

    返回:
        Iterator[pd.DataFrame]
    """
    con = duckdb.connect(database=database)
    cursor = con.execute(query)

    try:
        if columns is None:
            columns = [desc[0] for desc in cursor.description]

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            yield pd.DataFrame(rows, columns=columns)
    finally:
        con.close()


def attribute_df_to_entity_dict(df_chunk):
    result = defaultdict(dict)
    # 结构：result[head_type][head_id] = entity_dict

    for _, row in df_chunk.iterrows():
        head_type = row["head_type"]
        head_id = row["head_id"]
        name = row["head_name"]
        prop = row["relation"]
        value = row["tail"]

        entity_map = result[head_type]

        if head_id not in entity_map:
            entity_map[head_id] = {"id": head_id}

        entity_map[head_id][prop] = value
        entity_map[head_id]["name"] = name

    # 把内层 dict 转成 list
    return {
        head_type: list(entities.values()) for head_type, entities in result.items()
    }
