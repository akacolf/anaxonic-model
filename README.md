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

For basic send/receive response training:

```bash
python -m anaxonic.train --config configs/basic_chat.json --data data/basic_responses.jsonl --format chat
python -m anaxonic.sample --checkpoint runs/basic_chat/latest.pt --chat --prompt "Hello."
```

## Talk To It

Train the starter chat checkpoint:

```bash
source .venv/bin/activate
pip install -e ".[dev]"
python -m anaxonic.train --config configs/basic_chat.json --data data/basic_responses.jsonl --format chat
```

Send a prompt:

```bash
python -m anaxonic.sample --checkpoint runs/basic_chat/latest.pt --chat --prompt "Reply with yes."
```

Try a few more:

```bash
python -m anaxonic.sample --checkpoint runs/basic_chat/latest.pt --chat --prompt "Hello."
python -m anaxonic.sample --checkpoint runs/basic_chat/latest.pt --chat --prompt "What is your name?"
python -m anaxonic.sample --checkpoint runs/basic_chat/latest.pt --chat --system "Be brief." --prompt "Can you answer basic questions?"
```

If you have not installed the package into your virtualenv yet, prefix commands with
`PYTHONPATH=src`, for example:

```bash
PYTHONPATH=src .venv/bin/python -m anaxonic.sample --checkpoint runs/basic_chat/latest.pt --chat --prompt "Hello."
```

This is still a tiny byte-level model trained on a starter dataset, so expect basic,
fragile answers until the dataset and model scale up.

## Shape

- `src/anaxonic/model.py` contains a compact GPT-style decoder transformer.
- `src/anaxonic/tokenizer.py` uses a deterministic byte-level tokenizer.
- `src/anaxonic/data.py` turns plain text or JSONL conversations into autoregressive batches.
- `src/anaxonic/train.py` trains with gradient accumulation, response masking, and checkpoints.
- `src/anaxonic/sample.py` generates text or chat-style responses from a checkpoint.
- `configs/tiny.json` is the smallest useful local training config.
- `configs/basic_chat.json` is the starter config for response training.
- `configs/ci_smoke_chat.json` keeps CI model checks fast.

## Philosophy

We start with a transparent baseline, then earn each complication:

1. Make a tiny model train end to end.
2. Add reliable evaluation and checkpoint metadata.
3. Improve data quality and mixture controls.
4. Scale model size only after throughput and loss curves behave.
5. Add distributed training, better tokenization, and alignment workflows.
