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

## Shape

- `src/anaxonic/model.py` contains a compact GPT-style decoder transformer.
- `src/anaxonic/tokenizer.py` uses a deterministic byte-level tokenizer.
- `src/anaxonic/data.py` turns plain text into autoregressive batches.
- `src/anaxonic/train.py` trains and writes checkpoints.
- `src/anaxonic/sample.py` generates text from a checkpoint.
- `configs/tiny.json` is the smallest useful local training config.

## Philosophy

We start with a transparent baseline, then earn each complication:

1. Make a tiny model train end to end.
2. Add reliable evaluation and checkpoint metadata.
3. Improve data quality and mixture controls.
4. Scale model size only after throughput and loss curves behave.
5. Add distributed training, better tokenization, and alignment workflows.

