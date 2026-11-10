# Implementation walkthrough

## Describe the project

SMMnlp is a local LLM application for abstracting an SMM diagnosis and its date
from clinical notes. It classifies notes, selects each positive patient's first
positive note, distinguishes historical and new diagnoses, and extracts the
historical diagnosis date together with a review flag. Separate reference
annotations support evaluation at the note and patient levels.

## Explain the code organization

| Module | Responsibility |
|---|---|
| `config.py` | Model location and loading settings |
| `prompts.py` | Task names, prompt templates, token limits, and output columns |
| `modeling.py` | Tokenizer/model loading, chat formatting, generation, decoding, cleanup |
| `inference.py` | One ordered note-processing loop for all three tasks |
| `preprocessing.py` | Label expansion, patient selection, sampling, and date assignment |
| `pipeline.py` | Compose the inference stages while sharing one model instance |
| `parsing.py` | Convert responses into labels, fields, and normalized partial dates |
| `evaluation.py` | Calculate and present classification metrics |
| `io.py` | Read and write JSON, Excel, and CSV tables |
| `paths.py` | Resolve project-relative data paths and environment overrides |

The `TextGenerator` interface has one method: `generate(prompt, max_new_tokens=...)`.
The workflow accepts either the local model adapter or a scripted generator.
This makes routing and data processing testable without loading model weights.

## Walk through inference

`LocalLLM.from_pretrained` loads a local tokenizer and
`Mistral3ForConditionalGeneration` checkpoint. It uses bfloat16 when the CUDA
device capability supports the configured selection rule, and float16 otherwise.
The application retains automatic device placement and disk offloading settings.

The generator constructs a system message and a user message, applies the
tokenizer's chat template, and calls model generation under inference mode.
Sampling is disabled. After generation, it slices away the input-token prefix,
so the decoder returns only the model's continuation.

Classification requests allow 5 new tokens; date extraction allows 50.
One list of documents is created before the loop. The same model instance serves
all inference stages, and importing the package does not load model weights.

## Explain important data decisions

The patient-level label is positive if any note is positive. Date extraction
uses the earliest positive note after a separate history classification step.
Partial dates follow explicit conventions: a year becomes July 1, and a month
and year become the first day of that month.

Date outputs must retain the dataframe index of the input notes. A normalized
response table otherwise receives a new index, which can attach a date to the
wrong patient. Tests cover nonconsecutive and duplicate indices.

Missing-value fills assign the returned series directly. External date-presence
evaluation includes missing model dates in the complete reference cohort; date
differences use only complete date pairs.

The two history parsers have different fallback rules. Explain those rules
explicitly when interpreting evaluation results. The diagnosis parser also
uses substring matching, so strict schema validation would be a separately
evaluated improvement.

## Discuss the evidence accurately

The synthetic demo proves workflow wiring with predefined responses. Automated
tests cover prompts, parsing, model adapter arguments, completion-only decoding,
patient selection, date assignment, file outputs, and notebook structure.

Those checks do not establish real-model clinical accuracy or GPU throughput.
Measure those using the selected checkpoint, its working dependency versions,
and representative labeled clinical data. The project uses pretrained-model
inference; it contains no weight-training step.

If asked about production priorities, discuss response validation, stable note
keys, representative regression data, pinned model environments, and profiling
before choosing batching or a different serving runtime.
