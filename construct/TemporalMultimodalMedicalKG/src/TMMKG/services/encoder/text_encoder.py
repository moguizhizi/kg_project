import logging
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TextEncoder:
    """
    Generic text encoder based on HuggingFace encoder models.
    Conforms to EncoderLike interface.
    """

    def __init__(
        self,
        model_name: str,
        device: Union[str, torch.device] = None,
        max_length: int = 512,
        pooling: str = "mean",
    ):
        self.model_name = model_name
        self.model_id = self._normalize_model_id(model_name)
        self.pooling = pooling
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_length = max_length

        logger.info(
            f"Initializing TextEncoder | model={model_name} | pooling={pooling} | device={self.device}"
        )

        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=True,
            trust_remote_code=True,
        )

        logger.info("Loading model...")
        self.model = AutoModel.from_pretrained(
            self.model_name,
            local_files_only=True,
            trust_remote_code=True,
        ).to(self.device)

        self.model.eval()

        self.dim = self.model.config.hidden_size

        logger.info(
            f"TextEncoder ready | max_length={self.max_length} | dim={self.dim}"
        )

    @torch.no_grad()
    def encode(self, text: str) -> List[float]:
        """
        Encode a single text into a dense vector.
        """
        if not isinstance(text, str) or not text.strip():
            logger.error("encode() received empty or invalid text")
            raise ValueError("TextEncoder.encode expects a non-empty string")

        logger.debug(f"Encoding text (len={len(text)})")

        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        if self.pooling == "mean":
            last_hidden = outputs.last_hidden_state  # (1, seq_len, dim)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(
                dim=1
            )
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")

        vector = pooled[0].cpu().tolist()

        logger.debug(f"Encoding finished | dim={len(vector)}")

        return vector

    def _normalize_model_id(self, name: str) -> str:
        # 只取最后一段
        return name.rstrip("/").split("/")[-1]
