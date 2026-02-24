# table_triple_extractor.py

from typing import Any, Dict, List, Optional, Set, Tuple
import os
import json
from pathlib import Path
from datetime import datetime
import re


import logging

from TMMKG.extractors.xlsx_loader import xlsx_to_records
from TMMKG.meta_type import FactBundle, TypedFact
from TMMKG.services.entity_resolver import EntityResolver
from TMMKG.utils.path_utils import save_no_candidates
import pickle

from TMMKG.utils.secure_utils import short_id
from TMMKG.utils.text_utils import deep_clean

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent.parent.parent
ONTOLOGY_MAPPINGS_DIR = BASE_DIR / "utils" / "ontology_mappings"
HOME_BASED_USER_TRAINING = (
    BASE_DIR / "utils" / "entity_registry" / "output_only_task_labels"
)

with open(os.path.join(ONTOLOGY_MAPPINGS_DIR, "prop2label.json"), "r") as f:
    PROP_2_LABEL = json.load(f)

with open(os.path.join(HOME_BASED_USER_TRAINING, "column_mapping.json"), "r") as f:
    COLUMN_MAPPING = json.load(f)

task_props = [
    ("认知加工深度", "AU_Q0042"),
    ("认知负荷水平", "AU_Q0043"),
    ("外部负荷", "AU_Q0044"),
    ("内部负荷", "AU_Q0045"),
    ("注意力类型", "AU_Q0046"),
    ("记忆力类型", "AU_Q0047"),
    ("思维类型", "AU_Q0048"),
    ("感知类型", "AU_Q0049"),
    ("题目限时", "AU_Q0050"),
    ("规则理解难度", "AU_Q0051"),
    ("题目复杂度", "AU_Q0052"),
    ("信息复杂度", "AU_Q0053"),
    ("指示明确性", "AU_Q0054"),
    ("题目呈现位置", "AU_Q0055"),
    ("题目呈现形式", "AU_Q0056"),
    ("题目是否语音播放", "AU_Q0057"),
    ("训练所需语言能力", "AU_Q0058"),
    ("出题方式", "AU_Q0059"),
    ("答题模式", "AU_Q0060"),
    ("答案唯一性", "AU_Q0061"),
    ("任务阶段划分", "AU_Q0062"),
    ("错误指导", "AU_Q0063"),
    ("答题反馈", "AU_Q0064"),
    ("玩法类型", "AU_Q0065"),
    ("任务动机", "AU_Q0066"),
    ("交互对象", "AU_Q0067"),
    ("操作方式", "AU_Q0068"),
    ("操作频率", "AU_Q0069"),
    ("视角", "AU_Q0070"),
    ("节奏", "AU_Q0071"),
    ("宏观题材分类", "AU_Q0072"),
    ("背景环境", "AU_Q0073"),
    ("情景主题", "AU_Q0074"),
    ("体验情景", "AU_Q0075"),
    ("叙事性强度", "AU_Q0076"),
    ("地域", "AU_Q0077"),
    ("年代", "AU_Q0078"),
    ("核心内容", "AU_Q0079"),
    ("核心内容颜色", "AU_Q0080"),
    ("核心内容形状", "AU_Q0081"),
    ("核心内容类别", "AU_Q0082"),
    ("核心内容动态", "AU_Q0083"),
    ("核心内容引申含义", "AU_Q0084"),
    ("环境内容", "AU_Q0085"),
    ("艺术风格", "AU_Q0086"),
    ("色调冷暖", "AU_Q0087"),
    ("主要颜色", "AU_Q0088"),
    ("辅助颜色", "AU_Q0089"),
    ("是否有渐变色", "AU_Q0090"),
    ("饱和度", "AU_Q0091"),
    ("元素大小", "AU_Q0092"),
    ("音乐风格", "AU_Q0093"),
    ("背景景别", "AU_Q0094"),
    ("背景时间", "AU_Q0095"),
    ("色彩搭配", "AU_Q0096"),
    ("情感氛围", "AU_Q0097"),
    ("色彩数量控制", "AU_Q0098"),
    ("画面结构", "AU_Q0099"),
    ("整体环境风格", "AU_Q0100"),
    ("配色对比度", "AU_Q0101"),
    ("元素密度", "AU_Q0102"),
    ("图形符号化程度", "AU_Q0103"),
    ("场景拟真度", "AU_Q0104"),
    ("深度表现", "AU_Q0105"),
    ("UI边框样式", "AU_Q0106"),
    ("交互按钮类型", "AU_Q0107"),
    ("情绪色彩倾向", "AU_Q0108"),
    ("图像细节丰富度", "AU_Q0109"),
    ("是否含拟人化元素", "AU_Q0110"),
    ("物理引擎", "AU_Q0111"),
    ("碰撞检测", "AU_Q0112"),
    ("任务类型", "AU_Q0113"),
    ("难度星级", "AU_Q0026"),
    ("同屏spine数量上限", "AU_Q0036"),
    ("题目是否含训练关键词（颜色、动物、形状、数字、食物）", "AU_Q0026"),
    ("题目是否含挑战性语言（快、注意、小心等词汇）", "AU_Q0026"),
]


special_trans = {
    "记忆力类型": {"无": ""},
    "思维类型": {"无": ""},
    "题目呈现形式​": {"文字提问+图示": "文字提问_图示"},
    "题目是否含训练关键词（颜色、动物、形状、数字、食物）": {"无": ""},
    "题目是否含挑战性语言（快、注意、小心等词汇）": {"无": ""},
    "题目是否语音播放": {"无": "否", "有": "是"},
    "情景主题": {"无": ""},
    "环境内容​": {"无": ""},
    "交互按钮类型": {"无": ""},
    "背景景别": {"无": ""},
    "背景时间": {"无": ""},
    "UI边框样式": {"无": ""},
    "音乐风格": {"无": ""},
    "色调冷暖": {"暖色": "暖"},
    "是否有渐变色": {"否 (推测)": "否"},
}

task_props = deep_clean(task_props)
special_trans = deep_clean(special_trans)


def apply_special_trans(field, value):
    return special_trans.get(field, {}).get(value, value)


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

    all_facts = attribute_facts + entity_facts

    return FactBundle(
        attribute_facts=attribute_facts,
        entity_facts=entity_facts,
        all_facts=all_facts,
    )


# =========================
# 属性七元组
# =========================

def _emit_fact(
    facts: List[TypedFact],
    head_id,
    head_type,
    head_name,
    relation_name,
    prop,
    value,
    tail_type,
):
    facts.append(
        (
            head_id,
            head_type,
            head_name,
            relation_name,
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
    # 任务（Task）
    # =========================================================

    task_key = COLUMN_MAPPING["CMS-ID"]
    task_id = record.get(task_key)

    if task_id:  # 只在存在时处理

        for col_name, tail_type in task_props:
            prop = COLUMN_MAPPING[col_name]
            if prop in skip_fields:
                continue

            val = record.get(prop)
            val = apply_special_trans(col_name, str(val))

            if val not in (None, "", "None"):
                _emit_fact(
                    facts,
                    task_id,
                    "AU_Q0023",
                    "NA",
                    PROP_2_LABEL[prop],
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

        logger.info(f"[Record {idx}] extracted {len(attribute_facts)} attribute facts")
        for f in attribute_facts:
            logger.info(f"  {f}")


if __name__ == "__main__":
    main()
