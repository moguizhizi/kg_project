from TMMKG.domains.home_based_user_training.table_triple_extractor import (
    extract_facts_from_records,
)
from TMMKG.extractors.xlsx_loader import xlsx_to_records
from TMMKG.utils.json_utils import write_facts_jsonl
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    xlsx_path = "/home/temp/dataset/temp.normalized.xlsx"

    if not os.path.exists(xlsx_path):
        logger.error(f"XLSX file does not exist: {xlsx_path}")
        return

    logger.info(f"Loading records from XLSX: {xlsx_path}")
    records = xlsx_to_records(
        path=xlsx_path,
        sheet_name="records",
    )
    logger.info(f"Loaded {len(records)} records from sheet 'records'")

    logger.info("Extracting facts from records...")
    fact_bundle = extract_facts_from_records(records)
    logger.info(
        f"Extracted {len(fact_bundle.attribute_facts)} attribute facts and "
        f"{len(fact_bundle.entity_facts)} entity facts"
    )

    attr_facts_path = "/home/temp/dataset/attribute_facts.jsonl"
    entity_facts_path = "/home/temp/dataset/entity_facts.jsonl"

    logger.info(f"Writing attribute facts to {attr_facts_path}")
    write_facts_jsonl(
        path=attr_facts_path,
        facts=fact_bundle.attribute_facts,
    )

    logger.info(f"Writing entity facts to {entity_facts_path}")
    write_facts_jsonl(
        path=entity_facts_path,
        facts=fact_bundle.entity_facts,
    )

    logger.info("Fact extraction and export completed successfully.")


if __name__ == "__main__":
    main()
