# Anaxonic Model

Anaxonic is a small, hackable foundation-model lab: a decoder-only transformer,
byte-level tokenizer, streaming text dataset, training loop, and sampling CLI.

The first target is not scale. It is correctness, observability, and a codebase
we can grow without losing the plot.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m anaxonic.train --config configs/tiny.json --data data/sample.txt
python -m anaxonic.sample --checkpoint runs/tiny/latest.pt --prompt "Once upon"
```

For chat response training:

```bash
python -m anaxonic.train --config configs/hf_chat_quick.json --data data/hf_chat.jsonl --format chat
python -m anaxonic.sample --checkpoint runs/hf_chat_quick/latest.pt --chat --prompt "Hello."
python3 chat.py
```

## Train With Hugging Face Data

Install the optional dataset tools:

```bash
pip install -e ".[data,dev]"
```

Convert a Hugging Face dataset split into Anaxonic's JSONL chat format:

```bash
python -m anaxonic.prepare_hf \
  --dataset yahma/alpaca-cleaned \
  --split train \
  --output data/hf_chat.jsonl \
  --format chat \
  --max-examples 10000 \
  --min-chars 20 \
  --max-chars 2000
```

Then train:

```bash
python -m anaxonic.train --config configs/hf_chat.json --data data/hf_chat.jsonl --format chat
python -m anaxonic.sample --checkpoint runs/hf_chat/best.pt --chat --prompt "Explain training data simply."
```

For the stronger local chat run, convert a larger capped UltraChat slice, blend it
with the local Anaxonic examples, and train the balanced frontier config:

```bash
python -m anaxonic.prepare_hf \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split train_sft \
  --output data/hf_ultrachat_12000.jsonl \
  --format chat \
  --streaming \
  --max-examples 12000 \
  --min-chars 20 \
  --max-chars 1800

python -m anaxonic.build_chat_data \
  --input data/basic_responses.jsonl \
  --input data/frontier_seed_chat.jsonl \
  --input data/frontier_chat.jsonl \
  --input data/hf_chat.jsonl \
  --input data/hf_ultrachat_12000.jsonl \
  --output data/hf_chat_frontier_long.jsonl \
  --min-response-chars 2 \
  --max-response-chars 1000

PYTHONUNBUFFERED=1 python -m anaxonic.train \
  --config configs/hf_chat_frontier_balanced.json \
  --data data/hf_chat_frontier_long.jsonl \
  --format chat
```

`prepare_hf` handles common chat and instruction-tuning shapes:

- `messages` records with `role`/`content`.
- ShareGPT-style `conversations` records with `from`/`value`.
- Prompt pairs such as `prompt`/`response`, `instruction`/`output`, and `question`/`answer`.
- Plain text corpora with `--format text --text-field text`.

Start with a capped, high-quality slice before using huge datasets. Keep examples
close to the behavior you want, compare validation loss across runs, and sample
chat outputs after each checkpoint.

## Talk To It

Train the chat checkpoint:

```bash
source .venv/bin/activate
pip install -e ".[dev]"
python -m anaxonic.train --config configs/hf_chat_quick.json --data data/hf_chat.jsonl --format chat
```

Send a prompt:

```bash
python -m anaxonic.sample --checkpoint runs/hf_chat_quick/latest.pt --chat --prompt "Reply with yes."
```

Or start an interactive chat loop:

```bash
python3 chat.py --system "Be brief."
```

The chat loop uses retrieval and built-in safety answers before model sampling.
By default it avoids sampling unknown prompts; pass `--generate-unknown` when you
want to inspect raw model completions.

The chat loop keeps a small local teaching log at `data/self_learning.jsonl`.
Use `/teach prompt => response` to add a correction, or `/good` to save the last
answer as a training example. The next chat session can retrieve those examples,
and the next training run can include the file as additional JSONL data.

Try a few more:

```bash
python -m anaxonic.sample --checkpoint runs/hf_chat_quick/latest.pt --chat --prompt "Hello."
python -m anaxonic.sample --checkpoint runs/hf_chat_quick/latest.pt --chat --prompt "What is your name?"
python -m anaxonic.sample --checkpoint runs/hf_chat_quick/latest.pt --chat --system "Be brief." --prompt "Can you answer basic questions?"
```

If you have not installed the package into your virtualenv yet, prefix commands with
`PYTHONPATH=src`, for example:

```bash
PYTHONPATH=src .venv/bin/python -m anaxonic.sample --checkpoint runs/hf_chat_quick/latest.pt --chat --prompt "Hello."
```

## Training Accuracy

`configs/hf_chat_quick.json` trains longer than the smoke config and reserves a small
validation split. Training logs report `val_loss` and `val_accuracy` when that
split is enabled, which gives a cleaner signal than measuring only on sampled
training examples. The trainer writes `best.pt` whenever validation loss improves
and `latest.pt` at normal checkpoint intervals. It also writes a JSONL training
curve at `runs/<run>/metrics.jsonl`, including gradient norm and approximate
tokens-per-second for quick stability and throughput checks.

For byte-level training, a random model starts near `ln(256) = 5.545` loss. A
two-step smoke run around 5.5 means the model is still close to random, not that
the target loss should already be near zero. To check whether the model and data
pipeline can actually learn, run the no-validation overfit diagnostic and look
for training/eval loss to fall sharply:

```bash
python -m anaxonic.train \
  --config configs/overfit_chat.json \
  --data data/basic_responses.jsonl \
  --format chat
```

For a faster CI-style check:

```bash
python -m anaxonic.train --config configs/ci_smoke_chat.json --data data/basic_responses.jsonl --format chat
```

## Data And Prompt Evals

Build a cleaned local chat dataset by mixing the existing response memory with
the stronger frontier seed examples:

```bash
PYTHONPATH=src python3 -m anaxonic.build_chat_data \
  --input data/basic_responses.jsonl \
  --input data/frontier_seed_chat.jsonl \
  --output data/frontier_chat.jsonl
```

Train the larger, longer local frontier config:

```bash
PYTHONPATH=src python3 -m anaxonic.train \
  --config configs/frontier_chat.json \
  --data data/frontier_chat.jsonl \
  --format chat
```

Run the fixed prompt eval set against any checkpoint:

```bash
PYTHONPATH=src python3 -m anaxonic.eval_chat \
  --checkpoint runs/frontier_chat/best.pt \
  --eval-data data/prompt_eval.jsonl \
  --output runs/frontier_chat/eval_best.json
```

Gate a checkpoint before release:

```bash
PYTHONPATH=src python3 -m anaxonic.release_gate \
  --report runs/frontier_chat/eval_best.json \
  --min-pass-rate 0.8 \
  --min-category-pass-rate 0.7 \
  --max-quality-failures 0
```

Benchmark multiple checkpoints with the same prompt suite:

```bash
PYTHONPATH=src python3 -m anaxonic.benchmark \
  --checkpoint runs/basic_chat/best.pt \
  --checkpoint runs/hf_chat_quick/best.pt \
  --eval-data data/prompt_eval.jsonl \
  --output-dir runs/benchmarks
```

Use the same eval file for every run. Treat a run as more convincing when
validation loss, prompt eval pass rate, and fixed-prompt samples improve
together. The eval summary also reports per-category pass rates and rejects
obvious degenerate repetition, so you can spot whether a checkpoint improved
broad behavior or only one easy slice.

This repo now covers the local versions of several production practices:
checkpointing, metrics history, cleaned/deduped data builds, fixed prompt evals,
quality checks, early stopping, and release gates. Kubernetes/MPI launchers,
blob checkpoint storage, Prometheus/Grafana, GPU health checks, custom
scheduling, preference modeling, and RLAIF are still scale-out layers to add
after the single-process training loop is consistently learning.

## Ticketing

Use Linear as the planning source of truth for this repo. Keep issue keys in
branch names, PRs, and commits so the Linear GitHub integration can automate
status moves. See `docs/TICKETING.md`.

## Shape

- `src/anaxonic/model.py` contains a compact GPT-style decoder transformer.
- `src/anaxonic/tokenizer.py` uses a deterministic byte-level tokenizer.
- `src/anaxonic/data.py` turns plain text or JSONL conversations into autoregressive batches.
- `src/anaxonic/train.py` trains with gradient accumulation, response masking, and checkpoints.
- `src/anaxonic/sample.py` generates text or chat-style responses from a checkpoint.
- `src/anaxonic/chat.py` provides an interactive terminal chat loop.
- `configs/tiny.json` is the smallest useful local training config.
- `configs/basic_chat.json` is a compact response-training config.
- `configs/hf_chat_quick.json` is the default local Hugging Face chat-training config.
- `configs/frontier_chat.json` is a larger, longer chat-training config for cleaned data.
- `configs/ci_smoke_chat.json` keeps CI model checks fast.

## Philosophy

We start with a transparent baseline, then earn each complication:

1. Make a tiny model train end to end.
2. Add reliable evaluation and checkpoint metadata.
3. Improve data quality and mixture controls.
4. Scale model size only after throughput and loss curves behave.
5. Add distributed training, better tokenization, and alignment workflows.
