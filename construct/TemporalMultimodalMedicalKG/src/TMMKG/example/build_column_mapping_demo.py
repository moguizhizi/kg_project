from pathlib import Path
import json
import os

from TMMKG.utils.xlsx_utils import build_column_mapping

BASE_DIR = Path(__file__).resolve().parent.parent
ONTOLOGY_MAPPINGS_DIR = BASE_DIR / "utils" / "ontology_mappings"
HOME_BASED_USER_TRAINING = (
    BASE_DIR / "utils" / "entity_registry" / "home_based_user_training"
)


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
