# utils/xlsx_utils.py

import json
import os
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent
ONTOLOGY_MAPPINGS_DIR = BASE_DIR / "ontology_mappings"
HOME_BASED_USER_TRAINING = BASE_DIR / "entity_registry" / "home_based_user_training"


def build_column_mapping(
    excel_to_label: Dict[str, str],
    property_ontology: Dict[str, str],
    entity_ontology: Dict[str, str],
    strict: bool = True,
) -> Dict[str, str]:
    """
    Build XLSX column mapping WITHOUT reversing ontology dicts.

    Excel column -> AU_P / AU_Q
    """

    column_mapping: Dict[str, str] = {}

    for excel_col, label in excel_to_label.items():
        label = str(label).strip()
        resolved_label = None

        # 1. 在属性表中顺序查
        for au_code, au_label in property_ontology.items():
            if au_code == label:
                resolved_label = au_label
                break

        # 2. 如果属性没找到，再查实体表
        if resolved_label is None:
            for au_code, au_label in entity_ontology.items():
                if au_code == label:
                    resolved_label = au_label
                    break

        # 3. 处理结果
        if resolved_label:
            column_mapping[excel_col] = resolved_label
        else:
            if strict:
                raise ValueError(
                    f"Cannot resolve column '{excel_col}' "
                    f"(label='{label}') to AU_P or AU_Q"
                )

    return column_mapping


# =========================
# main (demo / test)
# =========================
def main():

    with open(os.path.join(ONTOLOGY_MAPPINGS_DIR, "entity_type2label.json"), "r") as f:
        ENTITY_TYPE_2_LABEL = json.load(f)

    with open(os.path.join(ONTOLOGY_MAPPINGS_DIR, "prop2label.json"), "r") as f:
        PROP_2_LABEL = json.load(f)

    with open(os.path.join(HOME_BASED_USER_TRAINING, "column_mapping.json"), "r") as f:
        COLUMN_MAPPING = json.load(f)

    column_mapping = build_column_mapping(
        excel_to_label=COLUMN_MAPPING,
        property_ontology=ENTITY_TYPE_2_LABEL,
        entity_ontology=PROP_2_LABEL,
        strict=True,
    )

    print("Final column_mapping:")
    for k, v in column_mapping.items():
        print(f"  {k} -> {v}")


if __name__ == "__main__":
    main()
