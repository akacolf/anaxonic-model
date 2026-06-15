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
and `latest.pt` at normal checkpoint intervals.

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

Use the same eval file for every run. Treat a run as more convincing when
validation loss, prompt eval pass rate, and fixed-prompt samples improve
together.

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
