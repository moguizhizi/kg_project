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
    """

    def __init__(
        self,
        model_name: str,
        device: Union[str, torch.device] = None,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_length = max_length

        logger.info(
            f"Initializing TextEncoder | model={model_name} | device={self.device}"
        )

        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=True,
            trust_remote_code=False,
        )

        logger.info("Loading model...")
        self.model = AutoModel.from_pretrained(
            self.model_name,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=False,
        ).to(self.device)

        self.model.eval()

        logger.info(f"TextEncoder ready | max_length={self.max_length}")

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

        # mean pooling
        last_hidden = outputs.last_hidden_state  # (1, seq_len, dim)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)

        pooled = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

        vector = pooled[0].cpu().tolist()

        logger.debug(f"Encoding finished | dim={len(vector)}")

        return vector
