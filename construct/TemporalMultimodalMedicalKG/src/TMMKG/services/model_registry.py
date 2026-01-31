from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModelSpec:
    name: str
    embedding_dim: int


EMBEDDING_MODEL_REGISTRY: dict[str, EmbeddingModelSpec] = {
    "facebook/contriever": EmbeddingModelSpec(
        name="facebook/contriever",
        embedding_dim=768,
    ),
}


def get_embedding_spec(model_name: str) -> EmbeddingModelSpec:
    """
    Return embedding model specification.
    """
    try:
        return EMBEDDING_MODEL_REGISTRY[model_name]
    except KeyError:
        raise ValueError(
            f"Unknown embedding model: {model_name}. "
            f"Available: {list(EMBEDDING_MODEL_REGISTRY.keys())}"
        )
