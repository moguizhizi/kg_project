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


############################
# Cypher
############################

QUERY = """
MATCH (set:TaskInstanceSet)-[:包含]->(t:TaskInstance)-[:游戏]->(g:Game {id:$game_id})
WHERE toFloatOrNull(set.执行年龄) IS NOT NULL

WITH 
    g,
    toFloat(set.执行年龄) AS age,
    t.结果 AS result,
    t.任务类型 AS task_type

WITH
    g,
    CASE
        WHEN age < $child_age THEN "Child"
        WHEN age < $adult_age THEN "Adults"
        ELSE "Older"
    END AS age_group,
    result,
    task_type

WITH
    g,
    age_group,
    COUNT(*) AS total,
    SUM(CASE WHEN task_type = "自由" THEN 1 ELSE 0 END) AS free_total,
    SUM(CASE 
        WHEN task_type = "自由" AND result = "完成"
        THEN 1 ELSE 0 
    END) AS free_completed

WITH
    g,
    age_group,
    total,
    free_total,
    free_completed,

    CASE 
        WHEN free_total = 0 THEN 0.0
        ELSE ROUND(100.0 * free_completed / free_total, 2)
    END AS free_completed_pct,

    CASE 
        WHEN total = 0 THEN 0.0
        ELSE ROUND(100.0 * free_completed / total, 2)
    END AS free_completed_over_total_pct

RETURN 
    age_group AS key,
    {
        game_id: g.id,
        game_name: g.name,
        total: total,
        free_total: free_total,
        free_completed: free_completed,
        free_completed_pct: free_completed_pct,
        free_completed_over_total_pct: free_completed_over_total_pct
    } AS value
"""


############################
# 查询函数
############################


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

    return {record["key"]: record["value"] for record in result}


############################
# JSONL排序
############################


def sort_jsonl(file_path, metric):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    data.sort(key=lambda x: x.get(metric, 0), reverse=True)

    with open(file_path, "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Sorted -> {file_path}")


############################
# 主执行逻辑
############################


def run_game_stats(driver):

    # 6个输出文件
    output_files = {
        (
            "Child",
            "free_completed_pct",
        ): "/home/temp/dataset/home_based_user_training_20260123_v2/analysis/child_free_completed_pct.jsonl",
        (
            "Adults",
            "free_completed_pct",
        ): "/home/temp/dataset/home_based_user_training_20260123_v2/analysis/adults_free_completed_pct.jsonl",
        (
            "Older",
            "free_completed_pct",
        ): "/home/temp/dataset/home_based_user_training_20260123_v2/analysis/older_free_completed_pct.jsonl",
        (
            "Child",
            "free_completed_over_total_pct",
        ): "/home/temp/dataset/home_based_user_training_20260123_v2/analysis/child_free_completed_over_total_pct.jsonl",
        (
            "Adults",
            "free_completed_over_total_pct",
        ): "/home/temp/dataset/home_based_user_training_20260123_v2/analysis/adults_free_completed_over_total_pct.jsonl",
        (
            "Older",
            "free_completed_over_total_pct",
        ): "/home/temp/dataset/home_based_user_training_20260123_v2/analysis/older_free_completed_over_total_pct.jsonl",
    }

    # 清空文件
    for path in output_files.values():
        open(path, "w", encoding="utf-8").close()

    file_handlers = {k: open(v, "a", encoding="utf-8") for k, v in output_files.items()}

    try:
        with driver.session() as session:

            all_game_ids = session.execute_read(fetch_node_ids, "Game")
            logger.info(f"Total Game nodes: {len(all_game_ids)}")

            for idx, game_id in enumerate(all_game_ids, 1):

                stats = session.execute_read(get_taskinstances_by_game_result, game_id)

                for age_group, value in stats.items():

                    base_record = {
                        "game_id": value["game_id"],
                        "game_name": value["game_name"],
                        "total": value["total"],
                        "free_total": value["free_total"],
                        "free_completed": value["free_completed"],
                    }

                    # free_completed_pct
                    rec1 = {
                        **base_record,
                        "free_completed_pct": value["free_completed_pct"],
                    }

                    file_handlers[(age_group, "free_completed_pct")].write(
                        json.dumps(rec1, ensure_ascii=False) + "\n"
                    )

                    # free_completed_over_total_pct
                    rec2 = {
                        **base_record,
                        "free_completed_over_total_pct": value[
                            "free_completed_over_total_pct"
                        ],
                    }

                    file_handlers[(age_group, "free_completed_over_total_pct")].write(
                        json.dumps(rec2, ensure_ascii=False) + "\n"
                    )

                if idx % 50 == 0:
                    logger.info(f"Processed {idx} games")

    finally:
        for f in file_handlers.values():
            f.close()

    ############################
    # 排序
    ############################

    for (age, metric), path in output_files.items():
        sort_jsonl(path, metric)

    logger.info("All game stats saved and sorted.")


############################
# 启动
############################

run_game_stats(driver)
driver.close()
