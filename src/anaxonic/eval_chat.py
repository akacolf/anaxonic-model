from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from anaxonic.responses import LOW_CONFIDENCE_RESPONSE


def normalize(value: str) -> str:
    return " ".join(value.lower().split())


def score_response(response: str, case: Mapping[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    normalized = normalize(response)
    if not response.strip():
        failures.append("empty response")
    if case.get("allow_low_confidence") is not True and response.strip() == LOW_CONFIDENCE_RESPONSE:
        failures.append("low confidence fallback")
    for expected in case.get("expected_contains", []):
        if normalize(str(expected)) not in normalized:
            failures.append(f"missing: {expected}")
    for forbidden in case.get("expected_not_contains", []):
        if normalize(str(forbidden)) in normalized:
            failures.append(f"forbidden: {forbidden}")
    max_chars = case.get("max_chars")
    if max_chars is not None and len(response) > int(max_chars):
        failures.append(f"too long: {len(response)} > {max_chars}")
    min_chars = case.get("min_chars")
    if min_chars is not None and len(response) < int(min_chars):
        failures.append(f"too short: {len(response)} < {min_chars}")
    return not failures, failures


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    cases = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        if "prompt" not in payload:
            raise ValueError(f"{path}:{line_number} is missing prompt")
        for field in ("expected_contains", "expected_not_contains"):
            if field in payload and not isinstance(payload[field], list):
                raise ValueError(f"{path}:{line_number} {field} must be a list")
        cases.append(payload)
    if not cases:
        raise ValueError(f"{path} did not contain any eval cases")
    return cases


def evaluate_cases(
    cases: list[dict[str, Any]],
    responder: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    results = []
    passed = 0
    for case in cases:
        response = responder(case)
        ok, failures = score_response(response, case)
        passed += int(ok)
        results.append(
            {
                "id": case.get("id", case["prompt"]),
                "category": case.get("category", "general"),
                "prompt": case["prompt"],
                "response": response,
                "passed": ok,
                "failures": failures,
            }
        )
    return {
        "passed": passed,
        "total": len(cases),
        "pass_rate": passed / len(cases),
        "results": results,
    }


def print_summary(report: Mapping[str, Any]) -> None:
    print(f"pass_rate={report['pass_rate']:.3f} passed={report['passed']}/{report['total']}")
    for result in report["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} {result['id']}: {result['response']}")
        for failure in result["failures"]:
            print(f"  - {failure}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompt-level chat evals against a checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-data", default="data/prompt_eval.jsonl")
    parser.add_argument("--output")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--memory-data")
    parser.add_argument("--memory-only", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.eval_data)

    from anaxonic.sample import sample

    def responder(case: dict[str, Any]) -> str:
        return sample(
            args.checkpoint,
            case["prompt"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            chat=True,
            system=case.get("system"),
            memory_data=args.memory_data,
            generate_unknown=not args.memory_only,
        )

    report = evaluate_cases(cases, responder)
    print_summary(report)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
