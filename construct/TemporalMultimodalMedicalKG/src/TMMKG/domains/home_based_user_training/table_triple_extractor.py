# table_triple_extractor.py

from typing import Any, Dict, List, Optional, Set, Tuple
import os
import json
from pathlib import Path
from datetime import datetime


import logging

from TMMKG.extractors.xlsx_loader import xlsx_to_records
from TMMKG.meta_type import FactBundle, TypedFact
from TMMKG.services.entity_resolver import EntityResolver
from TMMKG.utils.path_utils import save_no_candidates
import pickle


BASE_DIR = Path(__file__).resolve().parent.parent.parent
ONTOLOGY_MAPPINGS_DIR = BASE_DIR / "utils" / "ontology_mappings"
HOME_BASED_USER_TRAINING = (
    BASE_DIR / "utils" / "entity_registry" / "home_based_user_training"
)

# ------------------ 全局缓存 ------------------
CACHE_FILE = "disease_candidates_cache.pkl"
DIS_TO_CANDIDATES_CACHE: Dict[str, List] = {}

# 模块加载时只加载一次缓存
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        DIS_TO_CANDIDATES_CACHE = pickle.load(f)
else:
    DIS_TO_CANDIDATES_CACHE = {}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


with open(os.path.join(ONTOLOGY_MAPPINGS_DIR, "prop2label.json"), "r") as f:
    PROP_2_LABEL = json.load(f)

with open(os.path.join(HOME_BASED_USER_TRAINING, "column_mapping.json"), "r") as f:
    COLUMN_MAPPING = json.load(f)

output_file = (
    "/home/temp/dataset/home_based_user_training_20260123_v2/no_candidates.jsonl"
)


def build_instance_to_entity(instance_sets, candidates):
    """
    将候选实体映射到实例集合
    返回 instance_to_disease 列表
    """
    instance_to_entity = []
    # 映射 entity_type 到 (PROP_2_LABEL, AU代码, Q代码)
    type_mapping = {
        "disease": ("AU_P0019", "AU_P0019", "AU_Q0013"),
        "unknown": ("AU_P0063", "AU_P0063", "AU_Q0041"),
        "symptom": ("AU_P0062", "AU_P0062", "AU_Q0040"),
    }

    for c in candidates:
        if c.entity_type not in type_mapping:
            continue

        prop_label, prop_id, q_id = type_mapping[c.entity_type]

        for i in instance_sets:
            instance_to_entity.append(
                (i[0], i[1], "NA", PROP_2_LABEL[prop_label], prop_id, c.entity_id, q_id)
            )

    return instance_to_entity


# =========================
# 顶层入口
# =========================
def extract_facts_from_records(
    records: List[Dict[str, Any]], resolver: EntityResolver = None
) -> FactBundle:
    attribute_facts: List[TypedFact] = []
    entity_facts: List[TypedFact] = []

    for i, record in enumerate(records):
        # if i > 20:
        #     break

        attrs = extract_attribute_facts(record)
        attribute_facts.extend(attrs)

        ents = extract_entity_facts(record, attrs, resolver=resolver)
        entity_facts.extend(ents)

    all_facts = attribute_facts + entity_facts

    return FactBundle(
        attribute_facts=attribute_facts,
        entity_facts=entity_facts,
        all_facts=all_facts,
    )


# =========================
# 实体七元组
# =========================
def extract_entity_facts(
    record: Dict[str, Any],
    attribute_facts: List[TypedFact],
    resolver: EntityResolver = None,
) -> List[TypedFact]:
    """
    从 extract_attribute_triples 输出的六元组构建实体层级关系六元组:
      病人 -> 实例集合
      实例集合 -> 任务事例
      任务事例 -> 任务
    Args:
        attribute_facts: 已抽取属性七元组列表
    Returns:
        List[TypedFact]: 实体层级关系七元组
    """
    entity_facts: List[TypedFact] = []

    # 构建映射集合
    patients: Set[Tuple[Any, str]] = set()
    instance_sets: Set[Tuple[str, str]] = set()
    task_instances: Set[Tuple[str, str]] = set()
    game_instances: Set[Tuple[str, str]] = set()

    for head, head_type, _, _, _, _, _ in attribute_facts:
        # 病人
        if head_type == COLUMN_MAPPING["患者id"]:
            patients.add((head, head_type))
        # 实例集合
        elif head_type == "AU_Q0039":
            instance_sets.add((head, head_type))
        # 任务事例
        elif head_type == "AU_Q0012":
            task_instances.add((head, head_type))
        # 游戏事例·
        elif head_type == "AU_Q0023":
            game_instances.add((head, head_type))

    # 构建实体层级关系
    # 病人 -> 实例集合
    patient_to_instance = [
        (p[0], p[1], "NA", PROP_2_LABEL["AU_P0057"], "AU_P0057", i[0], i[1])
        for p in patients
        for i in instance_sets
        if str(i[0]).startswith(str(p[0]))  # 55_20211231 开头是 55
    ]

    # 实例集合 -> 任务事例
    instance_to_task_instance = [
        (i[0], i[1], "NA", PROP_2_LABEL["AU_P0058"], "AU_P0058", t[0], t[1])
        for i in instance_sets
        for t in task_instances
        if str(t[0]).startswith(str(i[0]))  # 55_20211231_33 开头是 55_20211231
    ]

    # 任务事例 -> 任务
    task_instance_to_task = [
        (t[0], t[1], "NA", PROP_2_LABEL["AU_P0056"], "AU_P0056", g[0], g[1])
        for t in task_instances
        for g in game_instances
        if str(t[0]).endswith(str(g[0]))  # 55_20211231_33 的结尾是 33
    ]

    # ----------------- 疾病映射 -----------------
    instance_to_disease = []
    diseases_str = record.get(COLUMN_MAPPING["疾病"])
    if diseases_str:
        diseases_list = [item.strip() for item in diseases_str.split(",")]

        for dis in diseases_list:
            # 缓存中没有才解析
            if dis not in DIS_TO_CANDIDATES_CACHE:
                candidates = resolver.resolve(text=dis, top_k=1)
                DIS_TO_CANDIDATES_CACHE[dis] = candidates if candidates else []
                if not candidates:
                    save_no_candidates(dis)
                else:
                    # 立即追加写入文件
                    with open(CACHE_FILE, "wb") as f:
                        pickle.dump(DIS_TO_CANDIDATES_CACHE, f)

            candidates = DIS_TO_CANDIDATES_CACHE.get(dis, [])
            if candidates:
                instance_to_disease.extend(
                    build_instance_to_entity(instance_sets, candidates)
                )

    entity_facts.extend(patient_to_instance)
    entity_facts.extend(instance_to_task_instance)
    entity_facts.extend(task_instance_to_task)
    entity_facts.extend(instance_to_disease)

    return entity_facts


def extract_entity_facts_test(
    record: Dict[str, Any],
    attribute_facts: List[TypedFact],
    entity_resolver: EntityResolver,
) -> List[TypedFact]:
    """
    从 extract_attribute_triples 输出的六元组构建实体层级关系六元组:
      病人 -> 实例集合
      实例集合 -> 任务事例
      任务事例 -> 任务
    Args:
        attribute_facts: 已抽取属性七元组列表
    Returns:
        List[TypedFact]: 实体层级关系七元组
    """
    entity_facts: List[TypedFact] = []

    # 构建映射集合
    patients: Set[Tuple[Any, str]] = set()
    instance_sets: Set[Tuple[str, str]] = set()
    task_instances: Set[Tuple[str, str]] = set()
    game_instances: Set[Tuple[str, str]] = set()

    for head, head_type, _, _, _, _, _ in attribute_facts:
        # 病人
        if head_type == COLUMN_MAPPING["患者id"]:
            patients.add((head, head_type))
        # 实例集合
        elif head_type == "AU_Q0039":
            instance_sets.add((head, head_type))
        # 任务事例
        elif head_type == "AU_Q0012":
            task_instances.add((head, head_type))
        # 游戏事例·
        elif head_type == "AU_Q0023":
            game_instances.add((head, head_type))

    # 构建实体层级关系
    # 病人 -> 实例集合
    patient_to_instance = [
        (p[0], p[1], "NA", PROP_2_LABEL["AU_P0057"], "AU_P0057", i[0], i[1])
        for p in patients
        for i in instance_sets
        if str(i[0]).startswith(str(p[0]))  # 55_20211231 开头是 55
    ]

    # 实例集合 -> 任务事例
    instance_to_task_instance = [
        (i[0], i[1], "NA", PROP_2_LABEL["AU_P0058"], "AU_P0058", t[0], t[1])
        for i in instance_sets
        for t in task_instances
        if str(t[0]).startswith(str(i[0]))  # 55_20211231_33 开头是 55_20211231
    ]

    # 任务事例 -> 任务
    task_instance_to_task = [
        (t[0], t[1], "NA", PROP_2_LABEL["AU_P0056"], "AU_P0056", g[0], g[1])
        for t in task_instances
        for g in game_instances
        if str(t[0]).endswith(str(g[0]))  # 55_20211231_33 的结尾是 33
    ]

    instance_to_disease = []
    diseases = record[COLUMN_MAPPING["疾病"]]
    if diseases is not None:
        result = [item.strip() for item in diseases.split(",") if item.strip()]

        for dis in result:
            candidates = entity_resolver.resolve(dis, top_k=1)

            if not candidates:
                continue

            best = candidates[0]

            for i in instance_sets:
                instance_to_disease.append(
                    (
                        i[0],
                        i[1],
                        "NA",
                        PROP_2_LABEL["AU_P0019"],
                        "AU_P0019",
                        dis,
                        "AU_Q0013",
                    )
                )

    entity_facts.extend(patient_to_instance)
    entity_facts.extend(instance_to_task_instance)
    entity_facts.extend(task_instance_to_task)
    entity_facts.extend(instance_to_disease)

    return entity_facts


# =========================
# 属性七元组
# =========================


def _emit_fact(
    facts: List[TypedFact],
    head_id,
    head_type,
    head_name,
    prop,
    value,
    tail_type,
):
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


def extract_attribute_facts(
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

    patient_name = f"患者_{patient_id}"
    patient_type = patient_key
    patient_id = str(int(patient_id))

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
            _emit_fact(
                facts, patient_id, patient_type, patient_name, prop, val, tail_type
            )

    # =========================================================
    # 任务（Task）
    # =========================================================
    task_key = COLUMN_MAPPING["任务名称"]
    task_name = record.get(task_key)
    if not task_name:
        return facts

    task_type = task_key

    prop = COLUMN_MAPPING["任务id"]
    task_id = str(int(record.get(prop)))

    task_props = [
        ("任务类型", "AU_Q0038"),
    ]

    for col_name, tail_type in task_props:
        prop = COLUMN_MAPPING[col_name]
        if prop in skip_fields:
            continue
        val = record.get(prop)
        if val not in (None, ""):
            _emit_fact(facts, task_id, task_type, task_name, prop, val, tail_type)

    # =========================================================
    # 任务实例（Task Instance）
    # =========================================================
    date_str = record.get(COLUMN_MAPPING["训练日期"])
    if not date_str:
        return facts

    formatted_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")
    task_instance_id = f"{patient_id}_{formatted_date}_{task_id}"
    task_instance_type = "AU_Q0012"
    task_instance_name = f"任务_{task_instance_id}"

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
                task_instance_name,
                prop,
                val,
                tail_type,
            )

    # =========================================================
    # 实例集合（Task Instance Set）
    # =========================================================
    task_instance_set_id = f"{patient_id}_{formatted_date}"
    task_instance_set_type = "AU_Q0039"
    task_instance_set_name = f"事件_{task_instance_set_id}"

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
                task_instance_set_name,
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

    # ======== 正式抽取（统一入口） ========
    fact_bundle = extract_facts_from_records(records)

    logger.info(
        "Facts extracted | attribute=%d, entity=%d, total=%d",
        len(fact_bundle.attribute_facts),
        len(fact_bundle.entity_facts),
        len(fact_bundle.all_facts),
    )

    # ======== Sanity check：仅打印前几条 record ========
    for idx, record in enumerate(records[:3]):
        attribute_facts = extract_attribute_facts(record)
        entity_facts = extract_entity_facts(record, attribute_facts)

        logger.info(f"[Record {idx}] extracted {len(attribute_facts)} attribute facts")
        for f in attribute_facts:
            logger.info(f"  {f}")

        logger.info(f"[Record {idx}] extracted {len(entity_facts)} entity facts")
        for ef in entity_facts[:5]:  # 防止太长
            logger.info(f"  {ef}")


if __name__ == "__main__":
    main()
