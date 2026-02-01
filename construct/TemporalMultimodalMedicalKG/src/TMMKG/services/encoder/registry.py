# services/encoder/registry.py

from .text_encoder import TextEncoder
from TMMKG.services.model_registry import get_embedding_spec
from TMMKG.services.llm_path_resolver import build_llm_path

_ENCODER_TABLE = {
    "contriever": "facebook/contriever",
    "Qwen3-Embedding-8B": "Qwen/Qwen3-Embedding-8B",
}


def get_text_encoder(
    name: str,
    model_root: str | None = None,
    **kwargs,
) -> tuple[TextEncoder, int]:
    if name not in _ENCODER_TABLE:
        raise ValueError(
            f"Unknown encoder: {name}. " f"Available: {list(_ENCODER_TABLE.keys())}"
        )

    canonical_model_name = _ENCODER_TABLE[name]

    resolved_model = build_llm_path(
        canonical_model_name,
        model_root,
    )

    spec = get_embedding_spec(canonical_model_name)

    encoder = TextEncoder(
        model_name=resolved_model,
        **kwargs,
    )

    return encoder, spec.embedding_dim


def list_available_encoders() -> list[str]:
    return list(_ENCODER_TABLE.keys())
