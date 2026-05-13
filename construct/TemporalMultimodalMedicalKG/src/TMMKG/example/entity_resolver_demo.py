import logging

from TMMKG.services.entity_resolver import init_entity_resolver


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():

    resolver = init_entity_resolver(
        model_name="Qwen3-Embedding-8B",
        base_collection="entity_aliases",
        qdrant_url="http://localhost:6333",
        score_threshold=0.85,
    )

    # -----------------------
    # 4. test input
    # -----------------------
    text = "发烧"
    entity_types = ["symptom"]  # 可改为 ["disease"]

    logger.info(f"Resolving entity for text='{text}', filter={entity_types}")

    # -----------------------
    # 5. resolve
    # -----------------------
    candidates = resolver.resolve(
        text=text,
        top_k=5,
        entity_types=entity_types,
    )

    # -----------------------
    # 6. output
    # -----------------------
    if not candidates:
        logger.info("No candidates found")
        return

    logger.info("Candidates:")

    for c in candidates:
        logger.info(
            f"- entity_id={c.entity_id}, "
            f"type={c.entity_type}, "
            f"alias='{c.alias_label}', "
            f"is_canonical={c.is_canonical}, "
            f"score={c.score:.4f}"
        )


if __name__ == "__main__":
    main()
