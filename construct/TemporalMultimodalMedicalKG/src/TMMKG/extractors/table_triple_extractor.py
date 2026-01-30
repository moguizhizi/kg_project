# table_triple_extractor.py

from typing import Any, Dict, List, Optional, Set, Tuple


Triple = Tuple[Any, str, Any]


# =========================
# 顶层入口
# =========================


def extract_triples_from_records(
    records: List[Dict[str, Any]],
) -> List[Triple]:
    triples: List[Triple] = []

    for record in records:
        triples.extend(extract_attribute_triples(record))
        triples.extend(extract_disease_triples(record))
        triples.extend(extract_task_triples(record))
        triples.extend(extract_temporal_triples(record))

    return triples


# =========================
# 属性三元组
# =========================


def extract_attribute_triples(
    record: Dict[str, Any],
    subject_field: str = "患者id",
    skip_fields: Optional[Set[str]] = None,
) -> List[Triple]:
    triples: List[Triple] = []

    subject = f"患者_{record.get(subject_field)}"

    skip = skip_fields or {
        subject_field,
        "疾病",
        "任务id",
        "任务名称",
        "训练日期",
    }

    for k, v in record.items():
        if k in skip:
            continue
        if v is None or v == []:
            continue
        triples.append((subject, k, v))

    return triples


# =========================
# 疾病三元组
# =========================


def extract_disease_triples(
    record: Dict[str, Any],
    subject_field: str = "患者id",
    disease_field: str = "疾病",
) -> List[Triple]:
    triples: List[Triple] = []

    subject = f"患者_{record.get(subject_field)}"

    diseases = record.get(disease_field, [])
    for d in diseases:
        triples.append((subject, "患有", d))

    return triples


# =========================
# 任务三元组
# =========================


def extract_task_triples(
    record: Dict[str, Any],
    subject_field: str = "患者id",
) -> List[Triple]:
    triples: List[Triple] = []

    patient = f"患者_{record.get(subject_field)}"
    task_id = record.get("任务id")

    if task_id is None:
        return triples

    task = f"任务_{task_id}"

    triples.append((patient, "参与任务", task))

    for k in ("任务名称", "任务类型", "任务状态", "任务名称结果", "得分", "常模分"):
        v = record.get(k)
        if v is not None:
            triples.append((task, k, v))

    return triples


# =========================
# 时间三元组
# =========================


def extract_temporal_triples(
    record: Dict[str, Any],
    subject_field: str = "患者id",
    time_field: str = "训练日期",
) -> List[Triple]:
    triples: List[Triple] = []

    time = record.get(time_field)
    if not time:
        return triples

    patient = f"患者_{record.get(subject_field)}"
    triples.append((patient, "在时间", time))

    if record.get("任务id") is not None:
        task = f"任务_{record.get('任务id')}"
        triples.append((task, "发生时间", time))

    return triples
