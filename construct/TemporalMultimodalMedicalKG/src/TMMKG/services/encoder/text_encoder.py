import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Union


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

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, local_files_only=True, trust_remote_code=False
        )

        self.model = AutoModel.from_pretrained(
            self.model_name,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=False,
        ).to(device)

    @torch.no_grad()
    def encode(self, text: str) -> List[float]:
        """
        Encode a single text into a dense vector.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("TextEncoder.encode expects a non-empty string")

        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        # mean pooling over token embeddings
        last_hidden = outputs.last_hidden_state  # (1, seq_len, dim)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)

        pooled = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

        return pooled[0].cpu().tolist()
