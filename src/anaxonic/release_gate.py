from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def gate_failures(
    report: dict[str, Any],
    *,
    min_pass_rate: float,
    min_category_pass_rate: float | None = None,
    max_quality_failures: int = 0,
) -> list[str]:
    failures: list[str] = []
    pass_rate = float(report.get("pass_rate", 0.0))
    if pass_rate < min_pass_rate:
        failures.append(f"pass_rate {pass_rate:.3f} < required {min_pass_rate:.3f}")

    if min_category_pass_rate is not None:
        for category, score in report.get("categories", {}).items():
            category_rate = float(score.get("pass_rate", 0.0))
            if category_rate < min_category_pass_rate:
                failures.append(
                    f"category {category} pass_rate "
                    f"{category_rate:.3f} < required {min_category_pass_rate:.3f}"
                )

    quality_failures = sum(
        len(result.get("quality_failures", [])) for result in report.get("results", [])
    )
    if quality_failures > max_quality_failures:
        failures.append(f"quality_failures {quality_failures} > allowed {max_quality_failures}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail the run when an eval report misses gates.")
    parser.add_argument("--report", required=True)
    parser.add_argument("--min-pass-rate", type=float, required=True)
    parser.add_argument("--min-category-pass-rate", type=float)
    parser.add_argument("--max-quality-failures", type=int, default=0)
    args = parser.parse_args()

    failures = gate_failures(
        load_report(args.report),
        min_pass_rate=args.min_pass_rate,
        min_category_pass_rate=args.min_category_pass_rate,
        max_quality_failures=args.max_quality_failures,
    )
    if failures:
        print("release_gate=fail")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("release_gate=pass")


if __name__ == "__main__":
    main()
