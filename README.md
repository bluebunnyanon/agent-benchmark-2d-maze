# Agentic Evaluation in 2D Mazes

This repository contains the code and benchmark artifacts accompanying an anonymous submission on evaluating interactive agents in structured 2D maze environments.

## Overview

The benchmark evaluates multimodal agents on maze-based tasks requiring navigation, planning, mechanism interaction, error recovery, and long-horizon action execution.

The repository includes:

- maze specifications and benchmark instances
- environment implementations
- evaluation harness
- prompting configurations
- model adapters
- scoring utilities
- scripts for running the experiments
- artifacts required to reproduce reported results

## Installation

```bash
conda create -n multinet-v2 python=3.10
conda activate multinet-v2
pip install -e ".[dev,visual]"