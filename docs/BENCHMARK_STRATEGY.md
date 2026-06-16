# Benchmark Strategy

Anaxonic should aim to be broadly capable, but broad capability has to be
earned with narrow, reproducible evidence. This document turns the benchmark
ambition into a staged plan: first make local evals honest, then add public
benchmark adapters, then add agent scaffolds for long-horizon tasks.

## Target Benchmarks

| Benchmark | Capability signal | What Anaxonic needs |
| --- | --- | --- |
| MMLU / MMLU-Pro | Broad academic knowledge and multiple-choice reasoning | Exact-choice evaluator, prompt templates, calibration, contamination checks |
| GPQA / GPQA Diamond | Graduate-level science reasoning | Multiple-choice evaluator, answer extraction, refusal/uncertainty tracking |
| SWE-bench / SWE-bench Pro | Repository-level software engineering | Code-editing agent, test runner, patch generation, sandboxed repo tasks |
| FrontierCode-style coding evals | Competitive and frontier coding skill | Code execution harness, hidden/public tests, pass@k reporting |
| GDPval / GDPval-AA | Economically valuable work deliverables | Artifact generation, rubric grading, model-judge and human-review workflow |
| Blueprint-Bench | Spatial/multimodal reconstruction | Vision input pipeline, structured floor-plan output, geometry scoring |
| OSWorld | GUI computer-use agent behavior | Multimodal desktop agent, screenshot grounding, action execution, task state checks |
| Terminal-Bench | Long-horizon command-line work | Terminal agent scaffold, command budget, environment reset, execution-based scoring |

Some of these are model benchmarks; others are agent benchmarks. Scores on
SWE-bench Pro, OSWorld, Terminal-Bench, GDPval, and Blueprint-Bench depend on
the scaffold around the model as much as the weights themselves.

## Evaluation Ladder

1. **Local smoke evals**
   - Keep `data/prompt_eval.jsonl` as a fast regression suite.
   - Require stable pass rates through `anaxonic.release_gate`.
   - Track raw generation separately from routed or memory-assisted answers.

2. **Knowledge and reasoning evals**
   - Add a generic multiple-choice evaluator.
   - Report accuracy, invalid-answer rate, abstention rate, and per-subject
     breakdowns.
   - Start with MMLU-Pro-style and GPQA-style formats before claiming broad
     knowledge improvement.

3. **Code execution evals**
   - Add a task runner that can prompt the model, write candidate code, run
     tests, and score pass/fail.
   - Report pass@1 and pass@k with deterministic seeds and fixed budgets.
   - Keep training data and eval task sources separated to reduce leakage.

4. **Agent evals**
   - Build one common agent trace format for tool calls, observations, file
     edits, command output, screenshots, and final answers.
   - Reuse that trace format across SWE-style, Terminal-Bench-style, OSWorld,
     and GDPval-style tasks.
   - Store failed trajectories for error analysis before adding them to any
     training set.

5. **External submissions**
   - Only submit checkpoints/scaffolds that pass local gates.
   - Pin model checkpoint, decoding settings, scaffold version, task split,
     dependency versions, wall-clock limits, and token/tool budgets.
   - Publish both headline scores and failure categories.

## Scale Stack

Benchmarks prove capability only when the training and evaluation stack is
controlled. Public frontier-lab infrastructure points to a consistent pattern:
containerized jobs, Kubernetes scheduling for large GPU fleets, MPI-style
distributed training, regular checkpoint restart, blob storage for datasets and
checkpoints, Prometheus/Grafana monitoring, GPU health checks, quota-aware
scheduling, custom kernels where profiling justifies them, and explicit release
governance.

The current repo implements the local floor and scaffolds the next layer:

- `Dockerfile` builds a reproducible training/eval container.
- `infra/kubernetes/anaxonic-train-job.yaml` runs the existing trainer as a
  Kubernetes Job with persistent data and checkpoint volumes.
- `configs/scale_stack.json` tracks infra, data, monitoring, alignment, kernel,
  and release-governance components by status and next step.
- `configs/benchmark_manifest.json` tracks benchmark stages, scaffold
  dependency, metrics, and mandatory reporting fields.
- `python -m anaxonic.scale_stack` validates both manifests and prints the
  implemented/planned stack summary.

Do not treat Docker or Kubernetes as a substitute for model quality. They are
reproducibility and scale controls. The ladder still starts with local loss,
fixed prompt evals, and release gates before expensive distributed runs.

## Capability Workstreams

### Foundation Model

- Scale parameters, context length, and training tokens only after loss curves
  and local evals improve together.
- Replace or augment the byte tokenizer when throughput or compression becomes
  a bottleneck.
- Add mixture accounting so each training run records dataset names, versions,
  counts, weights, and filtering rules.
- Add distributed training only after checkpoint compatibility, deterministic
  restart, and per-rank metrics are covered by tests.
- Profile operator bottlenecks before adding custom Triton kernels or other
  low-level GPU code.

### Reasoning

- Train on high-quality step-by-step reasoning traces, but evaluate final-answer
  accuracy separately from explanation style.
- Add self-consistency and pass@k only as explicit eval modes, not as hidden
  defaults.
- Track uncertainty: wrong confident answers are worse than honest abstentions
  on science and expert-work benchmarks.
- Add preference-pair, critique, and supervised-revision data before attempting
  RLAIF-style reinforcement learning.

### Coding

- Train on repository-level edits, failing-test-to-fix traces, and multi-file
  patches.
- Add a code agent that can inspect files, run tests, patch files, and retry
  within a fixed budget.
- Measure compile/test failures, overfitting to visible tests, and excessive
  edits as first-class failure modes.
- Run code evals in containers with pinned dependencies and clean worktrees.

### Computer Use

- Add screenshot/image inputs before attempting OSWorld or Blueprint-Bench.
- Separate perception failures from planning failures and action failures.
- Prefer execution-based scoring whenever a benchmark provides it.
- Record tool calls, screenshots, file edits, command output, and final state in
  one trace schema so agent failures are comparable across benchmarks.

### Economically Valuable Work

- Build deliverable-oriented tasks: spreadsheets, docs, PDFs, slides, reports,
  analysis memos, and structured JSON.
- Use rubric grading plus sampled human review before treating model-judge
  scores as trustworthy.
- Measure cost, latency, revision count, and user intervention rate alongside
  quality.

## Reporting Standard

Every benchmark report should include:

- Model checkpoint and git commit.
- Training data manifest and exclusion policy.
- Eval split and benchmark version.
- Scaffold version and tool permissions.
- Decoding settings and random seeds.
- Token, time, command, and retry budgets.
- Aggregate score plus per-category scores.
- Invalid outputs, abstentions, crashes, and timeouts.
- Representative failures with traces.

## Near-Term Milestones

1. Add a multiple-choice eval harness for MMLU/GPQA-style datasets.
2. Extend `configs/benchmark_manifest.json` with real public split/source
   metadata as each adapter lands.
3. Add a code-execution eval harness with a tiny local task set.
4. Add a unified agent trace schema.
5. Add CI gates for smoke evals and benchmark report validation.
6. Add contamination controls: dataset manifests, hash lists, and eval-source
   exclusions.
7. Run public benchmark dry-runs before any leaderboard submission.
8. Add CUDA image builds, object-store checkpoint sync, and Prometheus export
   only after local runs are stable.

## Public Research Basis

These docs use public sources only, not private-stack claims:

- OpenAI, "Scaling Kubernetes to 7,500 nodes":
  https://openai.com/index/scaling-kubernetes-to-7500-nodes/
- OpenAI, "Introducing Triton":
  https://openai.com/index/triton/
- Anthropic, "Constitutional AI: Harmlessness from AI Feedback":
  https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback
- Anthropic, "Responsible Scaling Policy Version 3.0":
  https://www.anthropic.com/news/responsible-scaling-policy-v3

## Claim Discipline

Do not claim "strong in almost everything" from prompt evals alone. A defensible
claim should look like:

> This checkpoint improves over the previous Anaxonic release on local prompt
> evals, MMLU-style multiple choice, GPQA-style science reasoning, and the local
> code-execution suite, with no regression in quality gates.

After public benchmark submissions, the claim can cite exact scores, versions,
and scaffolds. Until then, use the benchmark ladder as the map, not the trophy.
