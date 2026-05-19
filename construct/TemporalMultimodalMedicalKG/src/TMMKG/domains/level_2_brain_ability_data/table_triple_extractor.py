from datetime import date, datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from TMMKG.extractors.parquet_loader import parquet_to_records
from TMMKG.meta_type import FactBundle, TypedFact

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent.parent.parent
ONTOLOGY_MAPPINGS_DIR = BASE_DIR / "utils" / "ontology_mappings"
ENTITY_REGISTRY_DIR = (
    BASE_DIR / "utils" / "entity_registry" / "level_2_brain_ability_data"
)

INSTANCE_SET_TYPE = "AU_Q0039"
NUMERIC_TYPE = "AU_Q0036"


with open(ONTOLOGY_MAPPINGS_DIR / "prop2label.json", "r") as f:
    PROP_2_LABEL = json.load(f)

with open(ENTITY_REGISTRY_DIR / "column_mapping.json", "r") as f:
    COLUMN_MAPPING = json.load(f)


L2BA_ATTRIBUTE_FIELDS = [
    "二级_心算",
    "二级_听理解",
    "二级_书写能力",
    "二级_加工速度",
    "二级_问题解决",
    "二级_口语生成",
    "二级_面孔识别",
    "二级_任务切换",
    "二级_反应速度",
    "二级_注意分配",
    "二级_客体识别",
    "二级_语义系统",
    "二级_手眼协调",
    "二级_前瞻记忆",
    "二级_情景记忆",
    "二级_积极情绪",
    "二级_注意广度",
    "二级_节律感知",
    "二级_记忆广度",
    "二级_客体记忆",
    "二级_空间记忆",
    "二级_阅读能力",
    "二级_路径规划",
    "二级_情绪识别",
    "二级_运动知觉",
    "二级_选择注意",
    "二级_空间注意",
    "二级_联结记忆",
    "二级_空间知觉",
    "二级_持续注意",
    "二级_情绪调节",
    "二级_冲突抑制",
    "二级_归纳与推理",
    "二级_表象与想象",
    "二级_工作记忆",
    "总分",
]

def _is_empty(value: Any) -> bool:
    return value is None or str(value).strip() in {"", "None", "nan", "NaN"}


def _normalize_patient_id(value: Any) -> str:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text

    if number.is_integer():
        return str(int(number))
    return text


def _format_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")

    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    return text.replace("-", "").replace("/", "")


def _emit_fact(
    facts: List[TypedFact],
    head_id: Any,
    head_type: str,
    head_name: str,
    prop: str,
    value: Any,
    tail_type: str,
) -> None:
    facts.append(
        (
            head_id,
            head_type,
            head_name,
            PROP_2_LABEL[prop],
            prop,
            value,
            tail_type,
        )
    )


def _build_instance_set(record: Dict[str, Any]) -> Optional[tuple[str, str]]:
    patient_value = record.get(COLUMN_MAPPING["患者id"])
    date_value = record.get(COLUMN_MAPPING["训练日期"])

    if _is_empty(patient_value) or _is_empty(date_value):
        return None

    patient_id = _normalize_patient_id(patient_value)
    formatted_date = _format_date(date_value)
    instance_set_id = f"{patient_id}_{formatted_date}"
    instance_set_name = f"事件_{instance_set_id}"
    return instance_set_id, instance_set_name


def extract_facts_from_records(
    records: List[Dict[str, Any]],
    skip_fields: Optional[Set[str]] = None,
    include_fields: Optional[Set[str]] = None,
) -> FactBundle:
    attribute_facts: List[TypedFact] = []

    for record in records:
        attrs = extract_attribute_facts(
            record,
            skip_fields=skip_fields,
            include_fields=include_fields,
        )
        attribute_facts.extend(attrs)

    return FactBundle(
        attribute_facts=attribute_facts,
        entity_facts=[],
        all_facts=attribute_facts,
    )


def extract_attribute_facts(
    record: Dict[str, Any],
    skip_fields: Optional[Set[str]] = None,
    include_fields: Optional[Set[str]] = None,
) -> List[TypedFact]:
    facts: List[TypedFact] = []
    skip_fields = skip_fields or set()

    instance_set = _build_instance_set(record)
    if instance_set is None:
        return facts

    instance_set_id, instance_set_name = instance_set

    for col_name in L2BA_ATTRIBUTE_FIELDS:
        prop = COLUMN_MAPPING[col_name]
        if prop in skip_fields:
            continue
        if include_fields is not None and prop not in include_fields:
            continue

        value = record.get(prop)
        if not _is_empty(value):
            _emit_fact(
                facts,
                instance_set_id,
                INSTANCE_SET_TYPE,
                instance_set_name,
                prop,
                value,
                NUMERIC_TYPE,
            )

    return facts


def main():
    parquet_path = "/home/temp/dataset/level_2_brain_ability_data/parquet/result_2.parquet"

    date_fields = [COLUMN_MAPPING["训练日期"]]

    logger.info("Loading parquet: %s", parquet_path)
    records = parquet_to_records(
        path=parquet_path,
        column_mapping=COLUMN_MAPPING,
        date_fields=date_fields,
    )

    logger.info("Loaded %d records", len(records))

    fact_bundle = extract_facts_from_records(records)

    logger.info(
        "Facts extracted | attribute=%d, entity=%d, total=%d",
        len(fact_bundle.attribute_facts),
        len(fact_bundle.entity_facts),
        len(fact_bundle.all_facts),
    )

    for idx, record in enumerate(records[:3]):
        attribute_facts = extract_attribute_facts(record)

        logger.info("[Record %d] extracted %d attribute facts", idx, len(attribute_facts))
        for fact in attribute_facts:
            logger.info("  %s", fact)


if __name__ == "__main__":
    main()
