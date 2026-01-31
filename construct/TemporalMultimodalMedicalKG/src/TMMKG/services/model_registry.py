from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModelSpec:
    name: str
    embedding_dim: int


LLM_MODEL_REGISTRY: dict[str, EmbeddingModelSpec] = {
    "facebook/contriever": EmbeddingModelSpec(
        name="facebook/contriever",
        embedding_dim=768,
    ),
}


def get_embedding_dim(model_name: str) -> int:
    try:
        return LLM_MODEL_REGISTRY[model_name].embedding_dim
    except KeyError:
        raise ValueError(f"Unknown model_name: {model_name}")
