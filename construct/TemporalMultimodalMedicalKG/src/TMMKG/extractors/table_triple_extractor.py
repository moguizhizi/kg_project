# table_triple_extractor.py

from typing import Any, Dict, List, Optional, Set, Tuple
import os
import json
from pathlib import Path
from datetime import datetime
import logging

from TMMKG.extractors.xlsx_loader import xlsx_to_records

BASE_DIR = Path(__file__).resolve().parent.parent
ONTOLOGY_MAPPINGS_DIR = BASE_DIR / "utils" / "ontology_mappings"
HOME_BASED_USER_TRAINING = (
    BASE_DIR / "utils" / "entity_registry" / "Home-based_user_training"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


with open(os.path.join(ONTOLOGY_MAPPINGS_DIR, "prop2label.json"), "r") as f:
    PROP_2_LABEL = json.load(f)

with open(os.path.join(HOME_BASED_USER_TRAINING, "column_mapping.json"), "r") as f:
    COLUMN_MAPPING = json.load(f)


TypedFact = Tuple[
    Any,  # head
    str,  # head_entity_type (AU_Qxxx)
    str,  # relation (AU_Pxxx)
    str,  # prop / role / slot（可选语义位）
    Any,  # tail
    str,  # tail_entity_type (AU_Qxxx or literal)
]


# =========================
# 顶层入口
# =========================


def extract_triples_from_records(
    records: List[Dict[str, Any]],
) -> List[TypedFact]:
    pass


# =========================
# 属性三元组
# =========================


def _emit_fact(
    facts: List[TypedFact],
    head,
    head_type,
    prop,
    value,
    tail_type,
):
    facts.append(
        (
            head,
            head_type,
            PROP_2_LABEL[prop],
            prop,
            value,
            tail_type,
        )
    )


def extract_attribute_triples(
    record: Dict[str, Any],
    skip_fields: Optional[Set[str]] = None,
) -> List[TypedFact]:
    facts: List[TypedFact] = []
    skip_fields = skip_fields or set()

    # =========================================================
    # 病人（Patient）
    # =========================================================
    patient_key = COLUMN_MAPPING["患者id"]
    patient_id = record.get(patient_key)
    if not patient_id:
        return facts

    patient_type = patient_key

    # ---------- 病人属性 ----------
    patient_props = [
        ("性别", "AU_Q0029"),
    ]

    for col_name, tail_type in patient_props:
        prop = COLUMN_MAPPING[col_name]
        if prop in skip_fields:
            continue
        val = record.get(prop)
        if val not in (None, ""):
            _emit_fact(facts, patient_id, patient_type, prop, val, tail_type)

    # =========================================================
    # 任务（Task）
    # =========================================================
    task_key = COLUMN_MAPPING["任务名称"]
    task_name = record.get(task_key)
    if not task_name:
        return facts

    task_type = task_key

    task_props = [
        ("任务id", "AU_Q0026"),
        ("任务类型", "AU_Q0038"),
    ]

    for col_name, tail_type in task_props:
        prop = COLUMN_MAPPING[col_name]
        if prop in skip_fields:
            continue
        val = record.get(prop)
        if val not in (None, ""):
            _emit_fact(facts, task_name, task_type, prop, val, tail_type)

    # =========================================================
    # 任务实例（Task Instance）
    # =========================================================
    date_str = record.get(COLUMN_MAPPING["训练日期"])
    if not date_str:
        return facts

    formatted_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")
    task_instance_id = (
        f"{patient_id}_{formatted_date}_{record.get(COLUMN_MAPPING['任务id'])}"
    )
    task_instance_type = "AU_Q0012"

    instance_props = [
        ("任务名称结果", "AU_Q0026"),
        ("得分", "AU_Q0036"),
        ("常模分", "AU_Q0036"),
        ("是否活跃", "AU_Q0037"),
        ("任务状态", "AU_Q0026"),
    ]

    for col_name, tail_type in instance_props:
        prop = COLUMN_MAPPING[col_name]
        if prop in skip_fields:
            continue
        val = record.get(prop)
        if val not in (None, ""):
            _emit_fact(
                facts,
                task_instance_id,
                task_instance_type,
                prop,
                val,
                tail_type,
            )

    # =========================================================
    # 实例集合（Task Instance Set）
    # =========================================================
    task_instance_set_id = f"{patient_id}_{formatted_date}"
    task_instance_set_type = "AU_Q0039"

    instance_set_props = [
        ("年龄", "AU_Q0036"),
        ("学历", "AU_Q0036"),
        ("训练日期", "AU_Q0028"),
    ]

    for col_name, tail_type in instance_set_props:
        prop = COLUMN_MAPPING[col_name]
        if prop in skip_fields:
            continue
        val = record.get(prop)
        if val not in (None, ""):
            _emit_fact(
                facts,
                task_instance_set_id,
                task_instance_set_type,
                prop,
                val,
                tail_type,
            )

    return facts


def main():
    xlsx_path = "/home/temp/dataset/temp.normalized.xlsx"

    records = xlsx_to_records(
        path=xlsx_path,
        sheet_name="records",
    )

    logger.info(f"Loaded {len(records)} records")

    all_facts = []

    for idx, record in enumerate(records):
        facts = extract_attribute_triples(record)
        all_facts.extend(facts)

        if idx < 3:  # 只打印前几条做 sanity check
            logger.info(f"[Record {idx}] extracted {len(facts)} facts")
            for f in facts:
                logger.info(f"  {f}")

    logger.info(f"Total facts extracted: {len(all_facts)}")


if __name__ == "__main__":
    main()
