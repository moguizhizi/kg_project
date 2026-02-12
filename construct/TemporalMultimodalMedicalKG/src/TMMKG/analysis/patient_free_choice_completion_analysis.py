from TMMKG.graph.neo4j_db import fetch_node_ids
from TMMKG.infra.neo4j_db import create_neo4j_driver
from itertools import islice
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password"

driver = create_neo4j_driver(uri=URI, user=USER, password=PASSWORD)


QUERY = """
MATCH (p:Patient)
WITH p
ORDER BY p.id
SKIP $skip
LIMIT $limit

MATCH (p)-[:参加]->(e:TaskInstanceSet)-[:包含]->(t:TaskInstance)
WHERE toFloatOrNull(e.执行年龄) IS NOT NULL

WITH
    p.id AS patient_id,
    e.id AS event_id,
    toFloat(e.执行年龄) AS age,
    MAX(t.任务类型 = "自由") AS has_free,
    MAX(t.任务类型 = "专属") AS has_exclusive

WITH
    patient_id,
    event_id,
    CASE
        WHEN age < $child_age THEN "Child"
        WHEN age < $adult_age THEN "Adult"
        ELSE "Older"
    END AS age_group,
    has_free,
    has_exclusive

WITH
    patient_id,
    age_group,
    SUM(CASE WHEN has_free THEN 1 ELSE 0 END) AS free_events,
    SUM(CASE WHEN has_exclusive THEN 1 ELSE 0 END) AS exclusive_events,
    COUNT(*) AS total_events

RETURN
    age_group AS key,
    {
        patient_id: patient_id,
        total_events: total_events,
        free_events: free_events,
        exclusive_events: exclusive_events,
        free_to_exclusive_ratio:
            CASE 
                WHEN exclusive_events = 0 THEN NULL
                ELSE ROUND(1.0 * free_events / exclusive_events, 2)
            END
    } AS value
ORDER BY patient_id, key
"""


output_files = {
    "Child": "/home/temp/dataset/home_based_user_training_20260123_v2/analysis/patient_free_choice_stats_child.jsonl",
    "Adult": "/home/temp/dataset/home_based_user_training_20260123_v2/analysis/patient_free_choice_stats_adults.jsonl",
    "Older": "/home/temp/dataset/home_based_user_training_20260123_v2/analysis/patient_free_choice_stats_older.jsonl",
}


def batch_iterable(iterable, batch_size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch


def run_patient_stats(driver):

    # 清空文件
    for path in output_files.values():
        open(path, "w", encoding="utf-8").close()

    file_handlers = {k: open(v, "a", encoding="utf-8") for k, v in output_files.items()}

    batch_size = 1000
    params_base = {"child_age": 12, "adult_age": 60}

    try:
        with driver.session() as session:

            all_patient_ids = session.execute_read(fetch_node_ids, "Patient")
            logger.info(f"Total Patient nodes: {len(all_patient_ids)}")

            skip = 0

            for batch in batch_iterable(all_patient_ids, batch_size):

                params = {**params_base, "skip": skip, "limit": len(batch)}

                result = session.run(QUERY, params)

                record_count = 0

                for record in result:
                    age_group = record["key"]
                    value = record["value"]

                    if age_group in file_handlers:
                        file_handlers[age_group].write(
                            json.dumps(value, ensure_ascii=False) + "\n"
                        )
                        record_count += 1

                logger.info(
                    f"Processed patients {skip} ~ {skip + len(batch)} | records={record_count}"
                )

                skip += len(batch)

    finally:
        for f in file_handlers.values():
            f.close()

    logger.info("All patient stats saved.")


# 排序函数（重点优化）
def sort_jsonl_by_ratio(file_path):

    logger.info(f"Sorting file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    # None 排最后
    data.sort(
        key=lambda x: (
            x["free_to_exclusive_ratio"] is None,
            x["free_to_exclusive_ratio"] if x["free_to_exclusive_ratio"] else -1,
        ),
        reverse=True,
    )

    with open(file_path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(f"Finished sorting: {file_path}")


# =====================
# RUN
# =====================

run_patient_stats(driver)

for path in output_files.values():
    sort_jsonl_by_ratio(path)
