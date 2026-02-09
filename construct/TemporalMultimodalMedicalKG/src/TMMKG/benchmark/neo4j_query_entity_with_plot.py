import time
import logging
import random

import matplotlib.pyplot as plt

from TMMKG.graph.neo4j_db import (
    detect_super_nodes,
    fetch_node_ids,
    query_1hop_count,
    query_2hop_count,
)
from TMMKG.infra.neo4j_db import create_neo4j_driver
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password"


driver = create_neo4j_driver(
    uri=URI,
    user=USER,
    password=PASSWORD,
)


# ------------------------------------------------
# 获取 Patient ids（作为起点最安全）
# ------------------------------------------------
def fetch_patient_ids(tx, limit=None):

    return fetch_node_ids(tx, label="Patient", limit=limit)


# ------------------------------------------------
# 1-hop 查询
# ------------------------------------------------
def query_patient_1hop_count(tx, ids):

    return query_1hop_count(tx, label="Patient", ids=ids)


# ------------------------------------------------
# 2-hop 查询（最接近真实 KG 推理）
# ------------------------------------------------
def query_patient_2hop_count(tx, ids):

    return query_2hop_count(tx, label="Patient", ids=ids)


# ------------------------------------------------
# benchmark 核心
# ------------------------------------------------
def run_benchmark(session, patient_ids, query_func, title):

    sizes = [1, 10, 50, 100, 500, 1000]

    results = []

    logger.info(f"Running {title} benchmark...")
    logger.info("Warmup...")

    session.execute_read(
        query_func,
        random.sample(patient_ids, min(10, len(patient_ids))),
    )

    for size in sizes:

        if size > len(patient_ids):
            break

        ids = random.sample(patient_ids, size)

        start = time.perf_counter()

        triples = session.execute_read(query_func, ids)

        elapsed = time.perf_counter() - start

        logger.info(f"size={size:<6} triples={triples:<10} time={elapsed:.4f}s")

        results.append((size, elapsed))

    return results


# ------------------------------------------------
# 画图
# ------------------------------------------------
def plot_results(results, title):

    sizes = [r[0] for r in results]
    times = [r[1] for r in results]

    plt.figure(figsize=(8, 5))

    plt.plot(sizes, times, marker="o")

    plt.xlabel("Number of Start Nodes")
    plt.ylabel("Query Time (seconds)")
    plt.title(title)

    plt.grid(True)

    base_dir = Path("/home/temp/dataset/home_based_user_training_20260123_v2/benchmark")
    filename = title.lower().replace(" ", "_") + ".png"
    image_path = base_dir / filename

    plt.savefig(filename, dpi=150)

    logger.info(f"Benchmark plot saved to: {image_path}")

    plt.show()


# ------------------------------------------------
# main
# ------------------------------------------------
if __name__ == "__main__":

    logger.info("Starting Neo4j triple benchmark...")

    with driver.session() as session:

        # 🔥 不要全取 TaskInstance！
        patient_ids = session.execute_read(fetch_patient_ids)

        logger.info(f"Total patients: {len(patient_ids)}")

        # -----------------------
        # 1-hop
        # -----------------------
        results_1hop = run_benchmark(
            session,
            patient_ids,
            query_patient_1hop_count,
            "1-Hop Triple Query Performance",
        )

        plot_results(results_1hop, "1-Hop Triple Query Performance")

        # -----------------------
        # 2-hop
        # -----------------------
        results_2hop = run_benchmark(
            session,
            patient_ids,
            query_patient_2hop_count,
            "2-Hop Triple Query Performance",
        )

        plot_results(results_2hop, "2-Hop Triple Query Performance")

        # -----------------------
        # super node 检测
        # -----------------------
        logger.info("Detecting super nodes...")

        super_nodes = session.execute_read(detect_super_nodes, 100000)

        if super_nodes:

            logger.warning("⚠️ SUPER NODES DETECTED!")

            for r in super_nodes:
                logger.warning(
                    f"labels={r['labels']} id={r['id']} degree={r['degree']}"
                )

        else:
            logger.info("✅ No super nodes found.")

    driver.close()
