from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COURSE_EXAMPLES: list[dict[str, Any]] = [
    {
        "messages": [
            {
                "role": "user",
                "content": "How can I tell if a model is overfitting?",
            },
            {
                "role": "assistant",
                "content": (
                    "A model is probably overfitting when it performs much better on the "
                    "training data than on held-out validation or test data. It has learned "
                    "training-set details or noise instead of patterns that generalize."
                ),
            },
        ],
        "source": "course_distilled",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "How can I tell if a model is underfitting?",
            },
            {
                "role": "assistant",
                "content": (
                    "A model is probably underfitting when both training and validation "
                    "performance are poor. That usually means the model, training time, "
                    "features, or optimization setup is too weak to capture the pattern."
                ),
            },
        ],
        "source": "course_distilled",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "What does a good fit look like during training?",
            },
            {
                "role": "assistant",
                "content": (
                    "A good fit improves on training data while also improving on held-out "
                    "validation data. The sweet spot is before validation loss starts getting "
                    "worse while training loss keeps improving."
                ),
            },
        ],
        "source": "course_distilled",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "What should I do if training loss is low but validation loss is high?",
            },
            {
                "role": "assistant",
                "content": (
                    "Treat it as an overfitting warning. Use more representative data, a "
                    "larger validation set, early stopping, dropout, weight decay, lower "
                    "capacity, or stronger deduplication and data cleaning."
                ),
            },
        ],
        "source": "course_distilled",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "What should I do if training and validation losses are both high?",
            },
            {
                "role": "assistant",
                "content": (
                    "That points toward underfitting or weak optimization. Try training "
                    "longer, using a stronger model, improving the learning-rate schedule, "
                    "checking the labels/data format, or adding better task-relevant data."
                ),
            },
        ],
        "source": "course_distilled",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Why do we need a validation set?",
            },
            {
                "role": "assistant",
                "content": (
                    "A validation set estimates how the model behaves on data it did not "
                    "train on. It helps choose checkpoints and settings without trusting "
                    "training loss alone."
                ),
            },
        ],
        "source": "course_distilled",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the practical difference between batch, SGD, and mini-batch training?",
            },
            {
                "role": "assistant",
                "content": (
                    "Batch training averages gradients over the whole dataset before an "
                    "update, SGD updates from one example at a time, and mini-batch training "
                    "updates from small groups. Mini-batches are usually the practical "
                    "balance between speed, stability, and noisy exploration."
                ),
            },
        ],
        "source": "course_distilled",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Why do we shuffle training data each epoch?",
            },
            {
                "role": "assistant",
                "content": (
                    "Shuffling reduces order bias. It keeps mini-batches from always seeing "
                    "the same local pattern, which makes gradient updates less brittle."
                ),
            },
        ],
        "source": "course_distilled",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "How does dropout help with overfitting?",
            },
            {
                "role": "assistant",
                "content": (
                    "Dropout randomly disables some activations during training. That makes "
                    "the network rely less on any one path and usually improves "
                    "generalization."
                ),
            },
        ],
        "source": "course_distilled",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "How does weight decay or regularization help?",
            },
            {
                "role": "assistant",
                "content": (
                    "Regularization discourages unnecessarily large or complex weights. It "
                    "nudges the model toward simpler solutions that are less likely to "
                    "memorize noise."
                ),
            },
        ],
        "source": "course_distilled",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Why does more data help generalization?",
            },
            {
                "role": "assistant",
                "content": (
                    "More diverse, representative data reduces the chance that the model "
                    "only learns quirks of a small sample. It gives the model more chances "
                    "to learn stable patterns."
                ),
            },
        ],
        "source": "course_distilled",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "How should I judge our Anaxonic training run?",
            },
            {
                "role": "assistant",
                "content": (
                    "Compare sampled train loss, validation loss, prompt eval pass rate, and "
                    "raw no-memory outputs. If validation improves with train loss, keep "
                    "training. If train improves while validation worsens, stop or regularize."
                ),
            },
        ],
        "source": "course_distilled",
    },
]


def write_course_dataset(output: str | Path, repeat: int = 1) -> int:
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for _ in range(repeat):
            for example in COURSE_EXAMPLES:
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Write distilled AI-course chat examples.")
    parser.add_argument("--output", default="data/course_ai_distilled.jsonl")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    count = write_course_dataset(args.output, repeat=args.repeat)
    print(f"wrote {count} distilled course examples to {args.output}")


if __name__ == "__main__":
    main()
