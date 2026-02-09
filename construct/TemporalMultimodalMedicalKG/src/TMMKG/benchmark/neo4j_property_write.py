import time
import logging
import matplotlib.pyplot as plt
from pathlib import Path

from TMMKG.graph.neo4j_db import update_node_property, update_node_property_with_skip
from TMMKG.infra.neo4j_db import create_neo4j_driver


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
# 批量更新函数（核心）
# ------------------------------------------------
def update_patient_property(tx, batch_size):

    result = update_node_property(
        tx,
        label="Patient",
        prop_name="occupation",
        prop_value="工人",
        batch_size=batch_size,
    )

    return result.single()[0]


# ------------------------------------------------
# 推荐写法：使用 SKIP 做全量更新
# ------------------------------------------------
def update_patient_property_with_skip(tx, skip, batch_size):

    return update_node_property_with_skip(
        tx,
        label="Patient",
        prop_name="occupation",
        prop_value="工人",
        skip=skip,
        batch_size=batch_size,
    )


# ------------------------------------------------
# Benchmark
# ------------------------------------------------
def benchmark():

    batch_sizes = [100, 500, 1000, 5000]

    results = []

    with driver.session() as session:

        # 获取总数
        total = session.run("MATCH (p:Patient) RETURN count(p)").single()[0]

        logger.info(f"Total patients: {total}")

        for batch in batch_sizes:

            logger.info(f"Testing batch_size={batch}")

            start = time.perf_counter()

            for skip in range(0, total, batch):

                session.execute_write(update_patient_property_with_skip, skip, batch)

            elapsed = time.perf_counter() - start

            logger.info(f"batch={batch:<6} total_time={elapsed:.3f}s")

            results.append((batch, elapsed))

    return results


# ------------------------------------------------
# 画图
# ------------------------------------------------
def plot_results(results):

    batches = [r[0] for r in results]
    times = [r[1] for r in results]

    plt.figure(figsize=(8, 5))

    plt.plot(batches, times, marker="o")

    plt.xlabel("Batch Size")
    plt.ylabel("Total Write Time (seconds)")
    plt.title("Neo4j Property Write Benchmark")

    plt.grid(True)

    base_dir = Path("/home/temp/dataset/home_based_user_training_20260123_v2/benchmark")
    filename = "neo4j_property_write_benchmark.png"
    image_path = base_dir / filename
    plt.savefig(filename, dpi=150)

    logger.info(f"Saved plot -> {image_path}")

    plt.show()


# ------------------------------------------------
# main
# ------------------------------------------------
if __name__ == "__main__":

    logger.info("Starting property write benchmark...")

    results = benchmark()

    driver.close()

    plot_results(results)
