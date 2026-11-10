# Local model

Place your model checkpoint and tokenizer in `Ministral3-bnb-4bit/`, or provide
their existing directory with `--model-path` or `SMMNLP_MODEL_PATH`.

Inference uses `Mistral3ForConditionalGeneration` with local-only loading.
Model weights are separate from the application code.
