# Agentic Evaluation in 2D Mazes

This repository contains the code, benchmark instances, evaluation harness, and reproducibility artifacts accompanying an anonymous submission on evaluating interactive multimodal agents in structured 2D maze environments.

## Overview

The benchmark studies agent behavior in environments that require long-horizon interaction rather than single-step prediction. Agents must navigate partially observed mazes, execute actions, interact with mechanisms, recover from mistakes, and reach a target state.

The evaluation infrastructure includes:

* declarative 2D maze specifications
* maze validation and solvability checks
* environment implementations
* model-facing observation and action interfaces
* prompt and context-management configurations
* adapters for evaluated models
* episode execution and logging
* mechanism-aware progress scoring
* scripts for running and reproducing evaluations
* the maze corpus used by the reported experiments

## Installation

Python 3.10 or later is recommended.

```bash
conda create -n maze-benchmark python=3.10
conda activate maze-benchmark
pip install -e ".[dev,visual]"
```

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,visual]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,visual]"
```

## Verify the Installation

The core test suite does not require model API keys or a GPU.

```bash
pytest
```

Load-sensitive performance tests are excluded by default. To run them explicitly:

```bash
pytest -m slow
```

## Repository Structure

```text
.
├── gridworld/               Maze specifications, fixtures, validation,
│                            difficulty computation, and environment logic
├── interface/               Episode runner, agent interfaces, observations,
│                            action parsing, and interaction protocol
├── prompting_experiments/   Prompt templates and protocol configurations
├── pipeline/                Evaluation-pipeline utilities
├── scorer/                  Static and runtime progress scoring
├── scripts/                 Evaluation and analysis entry points
├── demo/                    Human-playable evaluation interface
├── ogbench/                 Vendored benchmark maze corpus and supporting code
├── tests/                   Test suite
├── assets/                  Evaluation figures and visualizations
├── RUNME.md                 Extended operator/reproduction instructions
└── pyproject.toml           Package and dependency configuration
```

## Maze Benchmark

Maze instances are represented as structured task specifications. Tasks may contain corridors, dead ends, distractors, keys, doors, switches, and other mechanisms that constrain the valid solution sequence.

The benchmark includes validation utilities that check whether a task is solvable and compute properties related to its difficulty.

To validate the included task specifications:

```bash
python -m gridworld.task_validator
```

## Evaluation Protocol

An evaluation is defined by two main inputs:

1. a **run configuration**, specifying models and inference settings;
2. a **manifest**, specifying the maze instances included in the evaluation.

The canonical evaluation pipeline is:

```bash
python -m scripts.run_pipeline \
  --run-config <run-config.json> \
  --manifest <manifest.json> \
  --seeds 0
```

Evaluation artifacts are written under the run artifact directory and include episode-level trajectories and scores.

## Smoke Test

A small smoke configuration can be used to verify the complete model-evaluation loop:

```bash
python -m scripts.run_pipeline \
  --run-config gridworld/fixtures/run_config.smoke_claude_sonnet.json \
  --manifest gridworld/fixtures/manifest.smoke_eval.json \
  --seeds 0
```

This requires an appropriate model-provider API key.

For example:

```bash
export ANTHROPIC_API_KEY=<your-key>
```

The smoke configuration is intended to verify the pipeline, not to reproduce the reported benchmark results.

## Reproducing the Main Evaluation

The configuration corresponding to the main evaluation is included in the repository:

```bash
python -m scripts.run_pipeline \
  --run-config gridworld/fixtures/run_config.r1.json \
  --manifest gridworld/fixtures/manifest.r1_balanced_03.json \
  --seeds 0
```

The evaluated model set includes remotely hosted and locally served models. Reproducing the complete evaluation therefore requires the corresponding provider credentials and, for locally served models, suitable GPU infrastructure.

Relevant provider credentials can be supplied through environment variables such as:

```bash
export ANTHROPIC_API_KEY=<your-key>
export MOONSHOT_API_KEY=<your-key>
```

Locally served models use an OpenAI-compatible endpoint configured in the corresponding run configuration.

Running the complete evaluation may incur API and compute costs.

## Scoring

Scoring is integrated into the evaluation pipeline.

The scorer combines task-level structure with runtime episode behavior to measure progress through the required interaction sequence.

Existing episode outputs can also be rescored independently:

```bash
multinet-score-json --help
```

The scoring implementation is located in:

```text
scorer/
```

## Model Interface

Agents interact with the benchmark through a common interface.

To add another model, implement an agent exposing the expected generation interface in:

```text
interface/agents/
```

and register the corresponding provider in the evaluation pipeline.

The harness handles:

* prompt construction
* observation formatting
* action parsing
* environment stepping
* episode termination
* trajectory logging
* scoring

This allows different models to be evaluated under the same environment and interaction protocol.

## Human-Playable Interface

A human-playable version of the maze environment is also included.

For example:

```bash
python play_task.py <path-to-maze.json>
```

A directory or manifest can also be supplied to browse multiple tasks.

The human interface uses the same underlying environment and observation infrastructure as the model-facing evaluation pipeline.

## Reproducibility

The repository contains the components required to reconstruct the evaluation described in the accompanying submission, including:

* benchmark maze instances
* evaluation manifests
* model run configurations
* prompts and interaction protocol
* environment implementation
* task validator
* evaluation harness
* scoring implementation
* test suite
* execution scripts

Model API responses may vary over time for externally hosted systems. The provided configurations record the inference settings used for the reported evaluation.

## Additional Instructions

More detailed execution and operator instructions are available in:

```text
RUNME.md
```

## License

This anonymous review artifact is distributed under the MIT License. Attribution has been anonymized for double-blind review and will be restored in the public version following review.
