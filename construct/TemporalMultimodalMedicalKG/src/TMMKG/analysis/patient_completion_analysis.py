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


def calculate_patient_completion(tx, patient_ids):
    """
    计算每个患者的事件完成度
    """
    patient_completion = {}

    for pid in patient_ids:
        query = """
        MATCH (p:Patient {id: $patient_id})-[:参加]->(e:TaskInstanceSet)-[:包含]->(t:TaskInstance)
        WITH e, collect(t.结果) AS task_results
        RETURN e.id AS event_id,
               CASE WHEN all(x IN task_results WHERE x = '完成') THEN true ELSE false END AS completed
        """
        result = tx.run(query, patient_id=pid)
        completed_events = 0
        total_events = 0
        for record in result:
            total_events += 1
            if record["completed"]:
                completed_events += 1

        completion_pct = (
            (completed_events / total_events * 100) if total_events > 0 else 0.0
        )
        patient_completion[pid] = {
            "total_events": total_events,
            "completed_events": completed_events,
            "completion_pct": round(completion_pct, 2),
        }

    return patient_completion


def batch_iterable(iterable, batch_size):
    """将可迭代对象分批生成"""
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch


def calculate_and_save_sample(
    patient_ids, sample_size=5, output_file="patient_completion.jsonl"
):
    """
    计算所有患者事件完成度，并打印部分结果，同时将结果保存到文件（JSON Lines 格式）

    Args:
        patient_ids (list): 所有患者 ID
        sample_size (int): 打印前多少个患者的结果
        output_file (str): 保存文件路径（JSON Lines 格式）
    """
    all_results = {}

    with driver.session() as session:
        # 一次性计算所有患者完成度
        all_results = session.execute_read(calculate_patient_completion, patient_ids)

        # 保存到文件，每行一个患者
        with open(output_file, "a", encoding="utf-8") as f:
            for pid, stats in all_results.items():
                record = {"patient_id": pid, **stats}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Processed {len(all_results)} patients, appended to {output_file}")

    # 打印前 sample_size 个患者的完成度
    logger.info(f"Showing first {sample_size} patients' completion info:")
    for pid, stats in list(all_results.items())[:sample_size]:
        logger.info(f"Patient ID: {pid}")
        logger.info(f"  Total events: {stats['total_events']}")
        logger.info(f"  Completed events: {stats['completed_events']}")
        logger.info(f"  Completion %: {stats['completion_pct']}")
        logger.info("-" * 40)

    return all_results


# 外层批量调度
batch_size_patient = 2000  # 每批处理 2000 个患者
output_file = "patient_completion.jsonl"

# 清空输出文件（可选）
with open(output_file, "w", encoding="utf-8") as f:
    pass

for patient_batch in batch_iterable(all_patient_ids, batch_size_patient):
    calculate_and_save_sample(
        patient_ids=patient_batch,
        sample_size=5,  # 每批打印前 5 个
        output_file=output_file,
    )

input_file = "patient_completion.jsonl"
output_file_sorted = "patient_completion_sorted.jsonl"

# 读取 JSONL 文件
all_patients = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        all_patients.append(record)

# 按 completion_pct 从高到低排序
all_patients_sorted = sorted(
    all_patients, key=lambda x: x["completion_pct"], reverse=True
)

# 保存排序后的结果
with open(output_file_sorted, "w", encoding="utf-8") as f:
    for record in all_patients_sorted:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

logger.info(f"Saved sorted patient completion data to {output_file_sorted}")
