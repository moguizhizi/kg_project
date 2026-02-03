import json
from typing import Iterable

from TMMKG.domains.home_based_user_training.table_triple_extractor import extract_facts_from_records
from TMMKG.extractors.xlsx_loader import xlsx_to_records
from TMMKG.meta_type import TypedFact


def write_facts_jsonl(
    path: str,
    facts: Iterable[TypedFact],
) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for fact in facts:
            # tuple → list，JSON 友好
            f.write(json.dumps(list(fact), ensure_ascii=False))
            f.write("\n")


def main():
    xlsx_path = "/home/temp/dataset/temp.normalized.xlsx"

    records = xlsx_to_records(
        path=xlsx_path,
        sheet_name="records",
    )

    fact_bundle = extract_facts_from_records(records)

    write_facts_jsonl(
        path="/home/temp/dataset/attribute_facts.jsonl",
        facts=fact_bundle.attribute_facts,
    )

    write_facts_jsonl(
        path="/home/temp/dataset/entity_facts.jsonl",
        facts=fact_bundle.entity_facts,
    )

    print(
        f"Wrote "
        f"{len(fact_bundle.attribute_facts)} attribute facts and "
        f"{len(fact_bundle.entity_facts)} entity facts"
    )


if __name__ == "__main__":
    main()
