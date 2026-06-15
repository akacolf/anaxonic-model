from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from anaxonic.eval_chat import evaluate_cases, load_cases, print_summary


def benchmark_checkpoint(
    checkpoint: str,
    eval_data: str,
    memory_data: str | None,
    memory_only: bool,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> dict[str, Any]:
    from anaxonic.sample import sample

    cases = load_cases(eval_data)

    def responder(case: dict[str, Any]) -> str:
        return sample(
            checkpoint,
            case["prompt"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            chat=True,
            system=case.get("system"),
            memory_data=memory_data,
            generate_unknown=not memory_only,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

    report = evaluate_cases(cases, responder)
    report["checkpoint"] = checkpoint
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one or more chat checkpoints.")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--eval-data", default="data/prompt_eval.jsonl")
    parser.add_argument("--output-dir", default="runs/benchmarks")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--memory-data")
    parser.add_argument("--memory-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for checkpoint in args.checkpoint:
        report = benchmark_checkpoint(
            checkpoint,
            args.eval_data,
            args.memory_data,
            args.memory_only,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            args.repetition_penalty,
        )
        print(f"checkpoint={checkpoint}")
        print_summary(report)
        summary.append(
            {
                "checkpoint": checkpoint,
                "passed": report["passed"],
                "total": report["total"],
                "pass_rate": report["pass_rate"],
                "categories": report["categories"],
            }
        )
        safe_name = Path(checkpoint).with_suffix("").as_posix().replace("/", "__")
        (output_dir / f"{safe_name}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
