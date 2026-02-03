import json
import duckdb
import pandas as pd
from typing import Iterator, List, Optional
from typing import Iterable

from TMMKG.domains.home_based_user_training.table_triple_extractor import (
    extract_facts_from_records,
)
from TMMKG.extractors.xlsx_loader import xlsx_to_records
from TMMKG.meta_type import TypedFact


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


def main():
    # xlsx_path = "/home/temp/dataset/temp.normalized.xlsx"

    # records = xlsx_to_records(
    #     path=xlsx_path,
    #     sheet_name="records",
    # )

    # fact_bundle = extract_facts_from_records(records)

    # write_facts_jsonl(
    #     path="/home/temp/dataset/attribute_facts.jsonl",
    #     facts=fact_bundle.attribute_facts,
    # )

    # write_facts_jsonl(
    #     path="/home/temp/dataset/entity_facts.jsonl",
    #     facts=fact_bundle.entity_facts,
    # )

    # print(
    #     f"Wrote "
    #     f"{len(fact_bundle.attribute_facts)} attribute facts and "
    #     f"{len(fact_bundle.entity_facts)} entity facts"
    # )

    path = "/home/temp/dataset/attribute_facts.jsonl"

    query = f"""
    SELECT
        json_extract(json, '$[0]')        AS head,
        json_extract_string(json, '$[1]') AS head_type,
        json_extract_string(json, '$[2]') AS relation,
        json_extract_string(json, '$[3]') AS prop,
        json_extract(json, '$[4]')        AS tail,
        json_extract_string(json, '$[5]') AS tail_type
    FROM read_json_auto('{path}')
    """

    for df_chunk in iter_duckdb_query_df(
        query=query, batch_size=100, database="facts.duckdb"
    ):
        print(df_chunk.shape)
        print(df_chunk.columns.tolist())


if __name__ == "__main__":
    main()
