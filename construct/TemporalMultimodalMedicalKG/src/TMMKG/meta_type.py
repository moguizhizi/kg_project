from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

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
