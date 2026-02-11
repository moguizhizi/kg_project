from TMMKG.graph.neo4j_db import fetch_node_ids
from TMMKG.infra.neo4j_db import create_neo4j_driver
from itertools import islice
import logging
import json
from itertools import islice

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j 连接信息
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password"

# 创建 driver
driver = create_neo4j_driver(uri=URI, user=USER, password=PASSWORD)

# 获取所有 Patient 节点的 ID
with driver.session() as session:
    all_patient_ids = session.execute_read(fetch_node_ids, "Patient")
    logger.info(f"Total Patient nodes: {len(all_patient_ids)}")


def batch_iterable(iterable, batch_size):
    """将可迭代对象分批生成"""
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch


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
    toFloat(e.执行年龄) AS age,
    e,
    COUNT(t) AS total_tasks,
    SUM(CASE WHEN t.结果 = '完成' THEN 1 ELSE 0 END) AS completed_tasks

WITH
    patient_id,
    CASE 
        WHEN age < $child_age THEN "Child"
        WHEN age < $adult_age THEN "Adult"
        ELSE "Older"
    END AS age_group,
    CASE 
        WHEN total_tasks = completed_tasks THEN 1
        ELSE 0
    END AS completed_flag

WITH
    patient_id,
    age_group,
    COUNT(*) AS total_events,
    SUM(completed_flag) AS completed_events

RETURN
    age_group AS key,
    {
        patient_id: patient_id,
        total_events: total_events,
        completed_events: completed_events,
        completion_pct: ROUND(100.0 * completed_events / total_events, 2)
    } AS value
ORDER BY patient_id, key
"""

# 输出文件
output_files = {
    "Child": "/home/temp/dataset/home_based_user_training_20260123_v2/analysis/patient_stats_child.jsonl",
    "Adult": "/home/temp/dataset/home_based_user_training_20260123_v2/analysis/patient_stats_adults.jsonl",
    "Older": "/home/temp/dataset/home_based_user_training_20260123_v2/analysis/patient_stats_older.jsonl",
}


def run_patient_stats(driver):

    # 先清空
    for path in output_files.values():
        open(path, "w", encoding="utf-8").close()

    file_handlers = {k: open(v, "a", encoding="utf-8") for k, v in output_files.items()}

    batch_size = 1000  #

    params_base = {"child_age": 12, "adult_age": 60}

    try:

        with driver.session() as session:

            # 获取所有 Patient ID
            all_patient_ids = session.execute_read(fetch_node_ids, "Patient")
            logger.info(f"Total Patient nodes: {len(all_patient_ids)}")

            skip = 0

            for batch in batch_iterable(all_patient_ids, batch_size):

                params = {**params_base, "skip": skip, "limit": len(batch)}

                result = session.run(QUERY, params)

                count = 0

                for record in result:
                    age_group = record["key"]
                    value = record["value"]

                    if age_group in file_handlers:
                        file_handlers[age_group].write(
                            json.dumps(value, ensure_ascii=False) + "\n"
                        )
                        count += 1

                logger.info(
                    f"Processed patients {skip} ~ {skip + len(batch)} | records={count}"
                )

                skip += len(batch)

    finally:
        for f in file_handlers.values():
            f.close()

    logger.info("All patient stats saved.")


def sort_jsonl_by_completion(file_path):
    """
    按 completion_pct 降序排序 JSONL 文件
    """

    logger.info(f"Sorting file: {file_path}")

    # 读取
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    # 排序（降序）
    data.sort(key=lambda x: x.get("completion_pct", 0), reverse=True)

    # 写回
    with open(file_path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(f"Finished sorting: {file_path}")


run_patient_stats(driver=driver)
# 排序三个文件
for path in output_files.values():
    sort_jsonl_by_completion(path)
