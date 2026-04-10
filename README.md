# LLM Classification Finetuning

Fine-tuning a quantized Qwen 2.5 model with LoRA for the Kaggle `llm-classification-finetuning` competition. The project trains a 3-class sequence classifier that predicts whether `model_a`, `model_b`, or a `tie` is preferred for a given prompt and pair of responses.

## Project Overview

This repository is centered around a single Kaggle notebook:

- [llm-classification-finetuning-comp (9).ipynb](./llm-classification-finetuning-comp%20(9).ipynb)

The notebook builds an end-to-end training pipeline that:

- loads the competition CSV files from Kaggle input mounts
- converts one-hot winner columns into a single label: `A`, `B`, or `C`
- applies response-swap augmentation to reduce position bias
- formats prompt/response triples into an instruction-style training sample
- truncates long conversations with a head-plus-tail strategy
- tokenizes the dataset for sequence classification
- fine-tunes `Qwen2.5-7B-Instruct` in 4-bit mode with LoRA adapters
- evaluates accuracy on a validation split
- generates `submission.csv` for Kaggle

## Approach

The training workflow in the notebook is:

1. Install dependencies in offline Kaggle mode from a local wheel directory.
2. Load `train.csv` from the Kaggle competition dataset.
3. Map labels:
   - `winner_model_a -> A`
   - `winner_model_b -> B`
   - `winner_tie -> C`
4. Randomly swap roughly 50% of samples between A and B, including text, model names, and labels.
5. Build an instruction prompt with sections for the original prompt, response A, and response B.
6. Truncate long text while keeping both the beginning and the end of each field.
7. Split the dataset into train/eval sets.
8. Tokenize to a max length of `512`.
9. Load a 4-bit quantized Qwen model and attach LoRA adapters.
10. Train a 3-label classifier and save the final adapter checkpoint.
11. Run batched inference on `test.csv` and export `submission.csv`.

## Model Setup

Base model used in the notebook:

- `Qwen/Qwen2.5-7B-Instruct` via a Kaggle-mounted local model path

Fine-tuning configuration:

- task type: sequence classification
- labels: `A`, `B`, `C`
- quantization: 4-bit `bitsandbytes` with `nf4`
- PEFT method: LoRA
- LoRA rank: `16`
- LoRA alpha: `32`
- LoRA dropout: `0.05`
- target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`

Training arguments used in the notebook:

- learning rate: `2e-4`
- epochs: `1`
- train batch size per device: `4`
- eval batch size per device: `4`
- gradient accumulation steps: `4`
- optimizer: `paged_adamw_8bit`
- evaluation strategy: every `100` steps
- logging steps: `50`
- mixed precision: `fp16`
- train subset used by the notebook: first `10,000` train rows
- eval subset used by the notebook: first `1,000` eval rows

## Expected Kaggle Inputs

The notebook assumes Kaggle-style mounted paths similar to:

```text
/kaggle/input/competitions/llm-classification-finetuning/train.csv
/kaggle/input/competitions/llm-classification-finetuning/test.csv
/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1
/kaggle/input/datasets/azizkarray/my-qwen-2-5-wheels/my_wheels
```

If your Kaggle inputs are mounted differently, update the path variables in the notebook before running it.

## Dependencies

The notebook installs these libraries:

- `transformers`
- `peft`
- `bitsandbytes`
- `trl`
- `accelerate`
- `datasets`
- `torch`
- `pandas`
- `numpy`
- `scikit-learn`

## Outputs

Running the notebook produces:

- `./qwen-chatbot-arena-model` during training
- `./qwen-chatbot-arena-final` with the saved LoRA adapter and tokenizer
- `submission.csv` for Kaggle upload

## How To Run

Best fit: Kaggle Notebook with GPU enabled.

1. Upload or attach the competition dataset.
2. Attach the local wheel dataset used for offline installs.
3. Attach the local Qwen 2.5 model dataset.
4. Open [llm-classification-finetuning-comp (9).ipynb](./llm-classification-finetuning-comp%20(9).ipynb).
5. Verify the input paths in the first few cells.
6. Run cells in order from top to bottom.
7. Download or submit the generated `submission.csv`.

## Repository Structure

```text
.
|-- llm-classification-finetuning-comp (9).ipynb
`-- README.md
```

## Notes

- The workflow is optimized for constrained GPU memory by combining 4-bit loading, LoRA, and small inference batches.
- Prompt construction uses parsed multi-turn conversations and preserves both early and late context during truncation.
- The notebook is currently notebook-first; there is no separate `src/` package or standalone training script yet.

## Possible Next Improvements

- move training and inference code into reusable Python modules
- add a reproducible `requirements.txt` or `environment.yml`
- parameterize dataset paths and hyperparameters
- save metrics and validation reports more explicitly
- add local training instructions outside Kaggle
