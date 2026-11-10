"""Local model loading and completion generation."""

from __future__ import annotations

import gc
from typing import Any, Protocol

from .config import ModelConfig


class TextGenerator(Protocol):
    """Small boundary that lets workflow tests run without model weights."""

    def generate(self, prompt: str, *, max_new_tokens: int) -> str: ...


class LocalLLM:
    def __init__(self, model: Any, tokenizer: Any, config: ModelConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    @classmethod
    def from_pretrained(cls, config: ModelConfig) -> LocalLLM:
        """Load the local Ministral model and tokenizer on demand."""
        import torch
        from transformers import AutoTokenizer, Mistral3ForConditionalGeneration

        tokenizer = AutoTokenizer.from_pretrained(
            str(config.model_path), local_files_only=config.local_files_only
        )
        cls._release_memory(torch)
        use_bfloat16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
        compute_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        model = Mistral3ForConditionalGeneration.from_pretrained(
            str(config.model_path),
            local_files_only=config.local_files_only,
            device_map=config.device_map,
            torch_dtype=compute_dtype,
            offload_folder=str(config.offload_folder),
            offload_state_dict=True,
            low_cpu_mem_usage=True,
        )
        return cls(model, tokenizer, config)

    @staticmethod
    def _release_memory(torch: Any) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        """Disable sampling and decode only tokens after the input prompt."""
        import torch

        messages = [
            {"role": "system", "content": self.config.system_message},
            {"role": "user", "content": prompt},
        ]
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        generated_ids = None
        try:
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    do_sample=False,
                )
            completion_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
            ]
            response = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)[0]
            return response.strip()
        finally:
            del model_inputs, generated_ids
            self._release_memory(torch)
