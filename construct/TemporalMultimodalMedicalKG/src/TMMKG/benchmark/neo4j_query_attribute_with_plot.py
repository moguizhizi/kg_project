import time
import random
import logging

import matplotlib.pyplot as plt

from TMMKG.graph.neo4j_db import fetch_node_ids, query_nodes
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


# ----------------------------
# Benchmark
# ----------------------------
def benchmark(label: str, max_ids: int | None = None):

    with driver.session() as session:

        logger.info(f"Fetching ids for label={label}...")

        all_ids = session.execute_read(
            fetch_node_ids,
            label,
            max_ids,
        )

        logger.info(f"{label} count used for benchmark: {len(all_ids)}")

        sizes = [100, 500, 1000, 5000, 10000, 20000]

        results = []

        # warmup
        logger.info("Running warmup query...")
        session.execute_read(
            query_nodes, label, random.sample(all_ids, min(10, len(all_ids)))
        )

        for size in sizes:

            if size > len(all_ids):
                break

            ids = random.sample(all_ids, size)

            start = time.perf_counter()
            count = session.execute_read(query_nodes, label, ids)
            elapsed = time.perf_counter() - start

            logger.info(
                f"[{label}] size={size:<6} count={count:<6} time={elapsed:.4f}s"
            )

            results.append((size, elapsed))

    return results


# ----------------------------
# 生成趋势图
# ----------------------------
def plot_results(results, title):

    sizes = [r[0] for r in results]
    times = [r[1] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, times, marker="o")

    plt.xlabel("Number of Nodes Queried")
    plt.ylabel("Query Time (seconds)")
    plt.title(title)
    plt.grid(True)

    base_dir = Path("/home/temp/dataset/home_based_user_training_20260123_v2/benchmark")
    filename = title.lower().replace(" ", "_") + ".png"
    image_path = base_dir / filename
    plt.savefig(image_path, dpi=150)

    logger.info(f"Benchmark plot saved to: {image_path}")
    plt.show()


# ----------------------------
# main
# ----------------------------
if __name__ == "__main__":

    logger.info("Starting Neo4j benchmark...")

    # Patient：全量 OK
    patient_results = benchmark(
        label="Patient",
        max_ids=None,
    )

    plot_results(patient_results, title="Patient Query Performance")

    # TaskInstance：只采样 100k（非常重要）
    task_results = benchmark(
        label="TaskInstance",
        max_ids=100_000,
    )

    plot_results(task_results, title="TaskInstance Query Performance")

    driver.close()
