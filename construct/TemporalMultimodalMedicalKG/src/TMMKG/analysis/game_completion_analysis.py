from TMMKG.graph.neo4j_db import fetch_node_ids
from TMMKG.infra.neo4j_db import create_neo4j_driver
from itertools import islice
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j 连接信息
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password"

# 创建 driver
driver = create_neo4j_driver(uri=URI, user=USER, password=PASSWORD)

# 获取所有 Game 节点的 ID
with driver.session() as session:
    all_game_ids = session.execute_read(fetch_node_ids, "Game")
    logger.info(f"Total Game nodes: {len(all_game_ids)}")


def batch_iterable(iterable, batch_size):
    """将可迭代对象分批生成"""
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch


QUERY = """
MATCH (set:TaskInstanceSet)-[:包含]->(t:TaskInstance)-[:游戏]->(g:Game {id:$game_id})
WHERE toFloatOrNull(set.执行年龄) IS NOT NULL

WITH 
    g,
    toFloat(set.执行年龄) AS age,
    t.结果 AS result

WITH
    g,
    CASE
        WHEN age < $child_age THEN "Child"
        WHEN age < $adult_age THEN "Adults"
        ELSE "Older"
    END AS age_group,
    result

WITH
    g,
    age_group,
    COUNT(*) AS total,
    SUM(CASE WHEN result = "完成" THEN 1 ELSE 0 END) AS completed

WITH
    g,
    age_group,
    total,
    completed,
    ROUND(100.0 * completed / total, 2) AS completed_pct

RETURN 
    age_group AS key,
    {
        game_id: g.id,
        game_name: g.name,
        total: total,
        completed: completed,
        completed_pct: completed_pct
    } AS value
"""


def get_taskinstances_by_game_result(
    tx,
    game_id,
    child_age=12,
    adult_age=60,
):
    result = tx.run(
        QUERY,
        game_id=game_id,
        child_age=child_age,
        adult_age=adult_age,
    )

    # 转换为 dict
    age_stats = {record["key"]: record["value"] for record in result}

    return age_stats


output_files = {
    "Child": "/home/temp/dataset/home_based_user_training_20260123_v2/analysis/game_task_stats_child.jsonl",
    "Adults": "/home/temp/dataset/home_based_user_training_20260123_v2/analysis/game_task_stats_adults.jsonl",
    "Older": "/home/temp/dataset/home_based_user_training_20260123_v2/analysis/game_task_stats_older.jsonl",
}

# 清空文件
for path in output_files.values():
    open(path, "w", encoding="utf-8").close()


# 提前打开文件
file_handlers = {
    age: open(path, "a", encoding="utf-8") for age, path in output_files.items()
}

batch_size_game = 50

try:
    with driver.session() as session:
        for game_batch in batch_iterable(all_game_ids, batch_size_game):

            for game_id in game_batch:
                stats = session.execute_read(get_taskinstances_by_game_result, game_id)

                # stats = {"儿童": {...}, "成年": {...}}

                for age_group, value in stats.items():
                    if age_group in file_handlers:
                        file_handlers[age_group].write(
                            json.dumps(value, ensure_ascii=False) + "\n"
                        )

            logger.info(f"Processed batch of {len(game_batch)} games")

finally:
    # 一定要关闭！
    for f in file_handlers.values():
        f.close()


def sort_jsonl_by_completed_pct(file_path):
    """
    按 completed_pct 从大到小排序 JSONL 文件
    """

    # 读取
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    # 排序（降序）
    data.sort(key=lambda x: x.get("completed_pct", 0), reverse=True)

    # 写回
    with open(file_path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Sorted: {file_path}")


# 排序所有文件
for path in output_files.values():
    sort_jsonl_by_completed_pct(path)
