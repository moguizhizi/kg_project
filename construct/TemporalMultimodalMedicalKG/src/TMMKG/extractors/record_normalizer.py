"""
record_normalizer.py

职责：
- 对 xlsx_loader 输出的 records 做字段级规范化
- 保证每条 record 结构稳定、类型可预期
- 为三元组抽取提供干净输入

不负责：
- 本体映射
- 三元组生成
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import math

from TMMKG.extractors.xlsx_loader import xlsx_to_records

logger = logging.getLogger(__name__)


# =========================
# 基础工具
# =========================


def is_nan(val: Any) -> bool:
    """判断 NaN（float('nan')）"""
    return isinstance(val, float) and math.isnan(val)


# =========================
# 字段级规范化
# =========================


def normalize_value(val: Any) -> Any:
    """
    统一单值字段：
    - NaN → None
    - 保留 str / int / float / list
    """
    if is_nan(val):
        return None
    return val


def normalize_boolean(val: Any) -> Optional[bool]:
    """
    中文枚举 → bool
    """
    if val in ("是", "Y", "yes", "Yes", True):
        return True
    if val in ("否", "N", "no", "No", False):
        return False
    return None


def normalize_list(val: Any) -> List[Any]:
    """
    统一 list 字段
    """
    if val is None or is_nan(val):
        return []
    if isinstance(val, list):
        return val
    return [val]


# =========================
# 单条 record 规范化
# =========================


def normalize_record(
    record: Dict[str, Any],
    boolean_fields: Optional[List[str]] = None,
    list_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    对单条 record 做规范化
    """
    normalized = {}

    for k, v in record.items():
        normalized[k] = normalize_value(v)

    # bool 字段
    if boolean_fields:
        for f in boolean_fields:
            if f in normalized:
                normalized[f] = normalize_boolean(normalized[f])

    # list 字段
    if list_fields:
        for f in list_fields:
            if f in normalized:
                normalized[f] = normalize_list(normalized[f])

    return normalized


# =========================
# records 批量规范化
# =========================


def normalize_records(
    records: List[Dict[str, Any]],
    boolean_fields: Optional[List[str]] = None,
    list_fields: Optional[List[str]] = None,
    drop_empty: bool = True,
) -> List[Dict[str, Any]]:
    """
    批量规范化 records
    """
    logger.info(f"Normalizing {len(records)} records")

    normalized_records = []

    for r in records:
        nr = normalize_record(
            r,
            boolean_fields=boolean_fields,
            list_fields=list_fields,
        )

        if drop_empty and not any(v is not None and v != [] for v in nr.values()):
            continue

        normalized_records.append(nr)

    logger.info(f"Normalized records count: {len(normalized_records)}")
    return normalized_records


def main():
    xlsx_path = "/home/temp/dataset/temp.xlsx"

    date_fields = ["训练日期"]
    multi_value_fields = ["疾病"]

    # Step 1: 从 XLSX 加载 records
    records = xlsx_to_records(
        path=xlsx_path,
        sheet_name="Sheet1",
        date_fields=date_fields,
        multi_value_fields=multi_value_fields,
    )

    print(f"[INFO] Raw records count: {len(records)}")
    print("[INFO] Raw sample record:")
    print(records[0])

    # Step 2: 规范化 records
    normalized_records = normalize_records(
        records,
        boolean_fields=["是否活跃"],  # 中文布尔字段
        list_fields=["疾病"],  # 多值字段
        drop_empty=True,
    )

    print(f"[INFO] Normalized records count: {len(normalized_records)}")
    print("[INFO] Normalized sample record:")
    print(normalized_records[0])


if __name__ == "__main__":
    main()
