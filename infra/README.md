# Anaxonic Scale Stack

This directory holds deployable scaffolds for moving Anaxonic from a local
research loop toward frontier-style operations. The current repo is still a
single-process trainer, so these files deliberately stop at container and
Kubernetes job boundaries while the codebase grows toward distributed training.

## Implemented Or Scaffolded

- `Dockerfile` builds a reproducible CPU container for local train/eval commands.
- `infra/kubernetes/anaxonic-train-job.yaml` runs the existing trainer as a
  Kubernetes Job with separate persistent volumes for data and checkpoints.
- `configs/scale_stack.json` records which scale components are implemented,
  scaffolded, partial, or planned.
- `configs/benchmark_manifest.json` records benchmark stages, scaffold
  dependencies, metrics, and reporting requirements.

## Next Layers

1. Add CUDA image builds once the target GPU environment is known.
2. Add `torchrun` or FSDP/DeepSpeed entrypoints before requesting multiple GPUs.
3. Add object-store checkpoint sync with checksums.
4. Export metrics to Prometheus from `runs/<run>/metrics.jsonl`.
5. Add DCGM/NVML preflight checks for GPU hosts.
6. Add preference-pair and supervised revision data before attempting RLAIF/RL.
7. Promote benchmark reports only when they include checkpoint, scaffold,
   split, budget, and failure analysis.
