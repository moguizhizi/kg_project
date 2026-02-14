import json
from pathlib import Path


def replace_and_merge_keys(data: dict, mapping: dict, existing: dict):
    """
    根据 mapping (AU_Q -> 中文)
    将 data 中的 中文key 替换为 AU_Q

    并合并到 existing 中
    """

    reverse_mapping = {v: k for k, v in mapping.items()}

    missing_keys = []

    for key, values in data.items():

        if key not in reverse_mapping:
            missing_keys.append(key)
            continue

        au_key = reverse_mapping[key]

        # 保证 values 是 list
        if not isinstance(values, list):
            values = [values]

        # 如果 AU_Q 不存在 -> 直接创建
        if au_key not in existing:
            existing[au_key] = sorted(set(values))
            continue

        # 合并 + 去重
        merged = set(existing[au_key]) | set(values)
        existing[au_key] = sorted(merged)

    return existing, missing_keys


def main():

    old_json_path = (
        "/home/temp/dataset/output_only_task_labels/column_unique_values.json"
    )

    mapping_json_path = "/home/project/kg_project/construct/TemporalMultimodalMedicalKG/src/TMMKG/utils/entity_registry/output_only_task_labels/entity_type2label.json"

    output_json_path = "/home/project/kg_project/construct/TemporalMultimodalMedicalKG/src/TMMKG/utils/entity_registry/output_only_task_labels/mapped_output.json"

    missing_json_path = "/home/project/kg_project/construct/TemporalMultimodalMedicalKG/src/TMMKG/utils/entity_registry/output_only_task_labels/missing_keys.json"

    # ===== 读取 =====
    with open(old_json_path, "r", encoding="utf-8") as f:
        old_data = json.load(f)

    with open(mapping_json_path, "r", encoding="utf-8") as f:
        mapping_dict = json.load(f)

    # ⭐ 如果 output 已存在 -> 读取
    if Path(output_json_path).exists():
        with open(output_json_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    # ===== 替换 + 合并 =====
    merged_data, missing = replace_and_merge_keys(
        old_data,
        mapping_dict,
        existing_data,
    )

    # ===== 保存 =====
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    if missing:
        print("\n未找到映射的key：")
        print(missing)

        with open(missing_json_path, "w", encoding="utf-8") as f:
            json.dump(missing, f, ensure_ascii=False, indent=2)

    else:
        print("\n全部key都成功匹配 ✅")


if __name__ == "__main__":
    main()
