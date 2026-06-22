# Qwen2.5-Coder-3B → HumanEval (QLoRA, instruction-tuned)

Fine-tuned the **Qwen2.5-Coder-3B base** model into an instruction-following Python
code generator using **QLoRA (4-bit)** via [Unsloth](https://github.com/unslothai/unsloth),
trained and evaluated **leak-free** with experiment tracking in Weights & Biases.

## Results (HumanEval, pass@1, greedy + EvalPlus sanitize)

| Model | HumanEval | HumanEval+ |
|-------|:---------:|:----------:|
| Base Qwen2.5-Coder-3B *(completion format)* | 0.433 | 0.354 |
| **Fine-tuned QLoRA *(instruction format)*** | **0.768** | **0.720** |
| **Improvement** | **+33.5 pts (+77.4% rel.)** | +36.6 pts |

The fine-tuned model approaches the *official* Qwen2.5-Coder-3B-Instruct (~0.84)
while being trained on a single consumer GPU.

> **Comparison note:** the base is scored in its native **completion** format (how
> code base models are benchmarked) and the fine-tune in the **instruction**
> format it was trained on. This is the standard base→instruct comparison (e.g.
> DeepSeek-Coder-base → Magicoder-S), stated explicitly for transparency.

## Integrity - why these numbers are honest

This project started on Llama-3.2-3B and originally reported an inflated score
because it was **training on the test set** (`bigcode/humanevalpack` *is*
HumanEval). Every result here is built to be defensible:

- **No benchmark data in training.** Trained on **Magicoder Evol-Instruct-110K**,
  disjoint from HumanEval.
- **13-gram decontamination.** Every training example is checked for 13-gram
  overlap against the HumanEval prompts and dropped if it matches (**124 examples
  removed** from this run).
- **Matched, reproducible eval.** Greedy decoding (deterministic pass@1) + output
  truncation + [EvalPlus](https://github.com/evalplus/evalplus) `sanitize`; base
  and fine-tuned scored with the identical harness.

A control experiment on Llama-3.2-3B with the same honest pipeline gained only
+2.5 pts - which correctly identified that the **base model**, not the training
recipe, was the ceiling. Switching to a code-specialized base + instruction
tuning is what produced the +33.5 pt jump.

## Training setup

| | |
|---|---|
| Base model | `unsloth/Qwen2.5-Coder-3B` (base, not instruct) |
| Method | QLoRA, 4-bit, LoRA r=64 / α=64, rsLoRA, all attention + MLP projections |
| Format | Qwen **ChatML** instruction format (train and eval matched) |
| Data | Magicoder Evol-Instruct-110K → 15K sampled, 124 dropped by decontamination → 14,250 train / 750 val |
| Hyperparams | lr 1e-4, cosine, 2 epochs, effective batch 16, adamw_8bit |
| Hardware | single NVIDIA A100-40GB (~59 min) |
| Tracking | Weights & Biases (live train/eval-loss curves + base-vs-finetuned comparison) |

## Reproduce

Open `finetune_qwen2.5-Coder-3B.ipynb` in Colab (L4 recommended) and run top to
bottom. It loads the base model, builds the decontaminated ChatML dataset, scores
the base model, fine-tunes, scores the fine-tune, and logs everything to W&B. No
secrets are hard-coded - the W&B key is entered at runtime and the HF push reads a
token via `getpass`.

## Model

Fine-tuned LoRA adapter on Hugging Face:
https://huggingface.co/WASP-193b/qwen25coder-3b-humaneval-ft

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    "WASP-193b/qwen25coder-3b-humaneval-ft", load_in_4bit=True,
)
# prompt with the Qwen ChatML template (see the notebook's eval cell)
```

## What I learned

- **Base model choice dominates.** General Llama-3.2-3B (~0.28 base) caps how far
  fine-tuning can go; a code-specialized base (~0.43) has the latent ability that
  instruction tuning unlocks.
- **Format alignment > raw data quality.** Matching the eval's instruction format
  mattered more than swapping in "higher quality" data.
- **A clean eval harness is part of the result.** Greedy + sanitization, and
  scoring base and fine-tuned identically, is what makes a number trustworthy.
