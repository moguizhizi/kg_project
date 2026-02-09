from pathlib import Path
import json
import logging

from TMMKG.utils.xlsx_utils import build_column_mapping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
ONTOLOGY_MAPPINGS_DIR = BASE_DIR / "utils" / "ontology_mappings"
HOME_BASED_USER_TRAINING = (
    BASE_DIR / "utils" / "entity_registry" / "home_based_user_training"
)


def load_json(path: Path):
    """带日志的 JSON 加载"""
    logger.info(f"Loading file: {path}")

    if not path.exists():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(path)

    with open(path, "r") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} records from {path.name}")
    return data


# =========================
# main (demo / test)
# =========================
def main():

    logger.info("Starting column mapping build...")

    try:
        entity_type_2_label = load_json(
            ONTOLOGY_MAPPINGS_DIR / "entity_type2label.json"
        )

        prop_2_label = load_json(ONTOLOGY_MAPPINGS_DIR / "prop2label.json")

        column_mapping_raw = load_json(HOME_BASED_USER_TRAINING / "column_mapping.json")

        logger.info("Building column mapping...")

        column_mapping = build_column_mapping(
            excel_to_label=column_mapping_raw,
            property_ontology=entity_type_2_label,
            entity_ontology=prop_2_label,
            strict=True,
        )

        logger.info(f"Column mapping built successfully total={len(column_mapping)}")

        logger.info("Final column_mapping:")
        for k, v in column_mapping.items():
            logger.info(f"{k} -> {v}")

    except Exception:
        logger.exception("Failed to build column mapping")
        raise


if __name__ == "__main__":
    main()
