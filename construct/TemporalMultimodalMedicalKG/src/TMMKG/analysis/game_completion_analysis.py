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


def get_taskinstances_by_game_result(tx, game_id, batch_size=10000):
    """
    分页获取指定 Game 关联的 TaskInstance ID，同时统计 result='完成' 的数量，并返回 Game 名称
    Args:
        tx: Neo4j 事务对象
        game_id: Game 节点 ID
        batch_size: 每次获取的节点数量
    Returns:
        dict:
            {
                "game_name": str,     # Game 名称
                "total": int,         # 总 TaskInstance 数量
                "completed": int,     # result='完成' 的数量
                "completed_pct": float # 完成百分比，保留两位小数
            }
    """
    query = """
    MATCH (t:TaskInstance)-[:游戏]->(g:Game {id: $game_id})
    RETURN t.id AS task_id, t.结果 AS result, g.name AS game_name
    SKIP $skip LIMIT $batch_size
    """
    all_task_ids = []
    completed_count = 0
    game_name = None
    skip = 0
    while True:
        result = tx.run(query, game_id=game_id, skip=skip, batch_size=batch_size)
        batch = [
            (record["task_id"], record["result"], record["game_name"])
            for record in result
        ]
        if not batch:
            break
        all_task_ids.extend([tid for tid, _, _ in batch])
        completed_count += sum(1 for _, r, _ in batch if r == "完成")
        # 获取 game_name（每次都一样，只需要取一次）
        if game_name is None and batch:
            game_name = batch[0][2]
        skip += batch_size

    total_count = len(all_task_ids)
    completed_pct = (completed_count / total_count * 100) if total_count > 0 else 0.0

    return {
        "game_name": game_name,
        "total": total_count,
        "completed": completed_count,
        "completed_pct": round(completed_pct, 2),
    }


output_file = "game_task_stats.jsonl"

# 批量处理 Game IDs
game_task_map = {}
batch_size_game = 50  # 每次处理 50 个 Game


all_results = []

# 逐批处理 Game，每处理完一批就写入文件
with driver.session() as session:
    for game_batch in batch_iterable(all_game_ids, batch_size_game):
        batch_results = []
        for game_id in game_batch:
            stats = session.execute_read(get_taskinstances_by_game_result, game_id)
            # 添加 game_id
            stats["game_id"] = game_id
            batch_results.append(stats)

        # 批量写入文件（JSON Lines，每行一个 Game 结果）
        with open(output_file, "a", encoding="utf-8") as f:
            for record in batch_results:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(
            f"Processed batch of {len(game_batch)} games, appended to {output_file}"
        )


input_file = "game_task_stats.jsonl"
output_file = "game_task_stats_sorted.jsonl"

# 读取 JSON Lines 文件
all_results = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        all_results.append(record)

# 按完成度排序（从高到低）
all_results_sorted = sorted(all_results, key=lambda x: x["completed_pct"], reverse=True)

# 写入新的文件
with open(output_file, "w", encoding="utf-8") as f:
    for record in all_results_sorted:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

logger.info(f"Saved sorted results to {output_file}")
