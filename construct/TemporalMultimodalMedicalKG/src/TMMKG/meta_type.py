from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

from pydantic import BaseModel

TypedFact = Tuple[
    Any,  # head_id
    str,  # head_entity_type (AU_Qxxx)
    Any,  # head_name
    str,  # relation (AU_Pxxx)
    str,  # prop / role / slot（可选语义位）
    Any,  # tail
    str,  # tail_entity_type (AU_Qxxx or literal)
]

EntityKey = Tuple[Any, Any]  # entity, entity_type


class DiseaseEntity(BaseModel):
    _id: int
    disease_id: str
    label: str


class SymptomEntity(BaseModel):
    _id: int
    symptom_id: str
    label: str


class UnknownEntity(BaseModel):
    _id: int
    unknown_id: str
    label: str

class EntityCandidate(BaseModel):
    entity_id: str
    entity_type: str
    alias_label: str
    is_canonical: bool
    score: float


@dataclass
class FactBundle:
    attribute_facts: List[TypedFact]
    entity_facts: List[TypedFact]
    all_facts: List[TypedFact]


@dataclass
class EntityTypeCandidate:
    entity_type_id: str
    alias_label: str
    is_canonical: bool
    score: float


@dataclass(frozen=True)
class EmbeddingModelSpec:
    name: str
    embedding_dim: int


# -----------------------
# PropertyCandidate
# -----------------------
@dataclass
class PropertyCandidate:
    property_id: str
    alias_label: str
    is_canonical: bool
    score: float
