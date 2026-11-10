import contextlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from smmnlp.config import ModelConfig
from smmnlp.modeling import LocalLLM


class FakeInputs(dict):
    def to(self, device):
        self.device = device
        return self


def fake_torch(*, cuda_available=True, capability=8):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=Mock(return_value=cuda_available),
            get_device_capability=Mock(return_value=(capability, 0)),
            empty_cache=Mock(),
        ),
        bfloat16="bfloat16",
        float16="float16",
        inference_mode=contextlib.nullcontext,
    )


class LocalModelTests(unittest.TestCase):
    def test_chat_template_completion_decoding_and_generation_flags(self):
        torch = fake_torch()
        inputs = FakeInputs(input_ids=[[10, 20, 30]], attention_mask=[[1, 1, 1]])
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = inputs
        tokenizer.batch_decode.return_value = [" Yes \n"]
        model = Mock(device="cuda:0")
        model.generate.return_value = [[10, 20, 30, 7, 8]]
        generator = LocalLLM(model, tokenizer, ModelConfig("local-model"))
        with patch.dict(sys.modules, {"torch": torch}):
            self.assertEqual(generator.generate("formatted prompt", max_new_tokens=5), "Yes")
        tokenizer.apply_chat_template.assert_called_once_with(
            [
                {"role": "system", "content": "You are a helpful data abstractor."},
                {"role": "user", "content": "formatted prompt"},
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        self.assertEqual(inputs.device, "cuda:0")
        model.generate.assert_called_once_with(
            **inputs, max_new_tokens=5, use_cache=True, do_sample=False
        )
        tokenizer.batch_decode.assert_called_once_with([[7, 8]], skip_special_tokens=True)
        torch.cuda.empty_cache.assert_called_once()

    def test_generation_error_propagates_and_runs_cleanup(self):
        torch = fake_torch()
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = FakeInputs(input_ids=[[1]])
        model = Mock(device="cuda:0")
        model.generate.side_effect = RuntimeError("synthetic generation failure")
        with patch.dict(sys.modules, {"torch": torch}):
            with self.assertRaisesRegex(RuntimeError, "synthetic generation failure"):
                LocalLLM(model, tokenizer, ModelConfig("model")).generate(
                    "prompt", max_new_tokens=50
                )
        torch.cuda.empty_cache.assert_called_once()

    def test_model_loading_preserves_dtype_and_offload_options(self):
        for available, capability, dtype in (
            (True, 8, "bfloat16"),
            (True, 7, "float16"),
            (False, 0, "float16"),
        ):
            with self.subTest(cuda=available, capability=capability):
                torch = fake_torch(cuda_available=available, capability=capability)
                tokenizer_class = Mock()
                model_class = Mock()
                transformers = SimpleNamespace(
                    AutoTokenizer=tokenizer_class, Mistral3ForConditionalGeneration=model_class
                )
                with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
                    result = LocalLLM.from_pretrained(
                        ModelConfig("weights", offload_folder="cache")
                    )
                tokenizer_class.from_pretrained.assert_called_once_with(
                    "weights", local_files_only=True
                )
                model_class.from_pretrained.assert_called_once_with(
                    "weights",
                    local_files_only=True,
                    device_map="auto",
                    torch_dtype=dtype,
                    offload_folder="cache",
                    offload_state_dict=True,
                    low_cpu_mem_usage=True,
                )
                self.assertIs(result.model, model_class.from_pretrained.return_value)
                self.assertIs(result.tokenizer, tokenizer_class.from_pretrained.return_value)


if __name__ == "__main__":
    unittest.main()
