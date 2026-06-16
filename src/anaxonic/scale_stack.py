from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StackComponent:
    name: str
    status: str
    repo_artifact: str | None
    purpose: str
    next_step: str

    @property
    def is_implemented(self) -> bool:
        return self.status in {"implemented", "local_implemented", "scaffolded"}


def load_scale_stack(path: str | Path = "configs/scale_stack.json") -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_scale_stack(payload)
    return payload


def load_benchmark_manifest(path: str | Path = "configs/benchmark_manifest.json") -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_benchmark_manifest(payload)
    return payload


def validate_scale_stack(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("scale stack schema_version must be 1")
    sources = payload.get("public_research_basis")
    if not isinstance(sources, list) or not sources:
        raise ValueError("scale stack must include public_research_basis")
    components = payload.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("scale stack must include components")

    names: set[str] = set()
    for component in components:
        for field in ("name", "status", "purpose", "next_step"):
            if not component.get(field):
                raise ValueError(f"scale stack component missing {field}")
        name = str(component["name"])
        if name in names:
            raise ValueError(f"duplicate scale stack component: {name}")
        names.add(name)


def validate_benchmark_manifest(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("benchmark manifest schema_version must be 1")
    required_fields = payload.get("reporting_required_fields")
    if not isinstance(required_fields, list) or "model_checkpoint" not in required_fields:
        raise ValueError("benchmark manifest must define reporting_required_fields")
    benchmarks = payload.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError("benchmark manifest must include benchmarks")

    names: set[str] = set()
    for benchmark in benchmarks:
        for field in ("name", "stage", "capability", "adapter_status", "metric"):
            if not benchmark.get(field):
                raise ValueError(f"benchmark missing {field}")
        name = str(benchmark["name"])
        if name in names:
            raise ValueError(f"duplicate benchmark: {name}")
        names.add(name)


def stack_summary(payload: dict[str, Any]) -> dict[str, Any]:
    components = [StackComponent(**component) for component in payload["components"]]
    implemented = [component.name for component in components if component.is_implemented]
    planned = [component.name for component in components if not component.is_implemented]
    return {
        "components": len(components),
        "implemented_or_scaffolded": implemented,
        "planned": planned,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize Anaxonic scale-stack manifests.")
    parser.add_argument("--scale-stack", default="configs/scale_stack.json")
    parser.add_argument("--benchmark-manifest", default="configs/benchmark_manifest.json")
    args = parser.parse_args()

    scale_stack = load_scale_stack(args.scale_stack)
    load_benchmark_manifest(args.benchmark_manifest)
    print(json.dumps(stack_summary(scale_stack), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
