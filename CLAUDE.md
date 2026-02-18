# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reinforcement learning research project for studying multitask preplay and model-based planning in JAX. It implements various RL algorithms on Craftax (a procedurally generated Minecraft-like environment) and Housemaze environments.

## Development Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

## Common Commands

### Running Experiments

```bash
# Local debug run
JAX_DEBUG_NANS=True JAX_DISABLE_JIT=1 HYDRA_FULL_ERROR=1 JAX_TRACEBACK_FILTERING=off \
uv run python -m ipdb -c continue craftax_trainer.py \
  app.parallel=none \
  app.debug=True \
  app.wandb=False \
  app.search=ql

# SLURM submission
uv run python craftax_trainer.py \
  app.parallel=slurm \
  app.search=usfa
```

### Code Formatting

```bash
uv run ruff format <file.py>
```

Ruff config: 2-space indent, 88-char line length, Python 3.10 target.

## Architecture

### Algorithms (in root directory)
- `qlearning_craftax.py` - Recurrent Q-learning with auxiliary achievement prediction
- `usfa_craftax.py` - Universal Successor Feature Approximators
- `dyna_craftax.py` - Dyna-style model-based RL
- `multitask_preplay_craftax_v2.py` - Multitask preplay algorithm (main research contribution)
- `alphazero_craftax.py` - AlphaZero with MCTS

### Core Components
- `craftax_env.py` - Modified Craftax environment with:
  - Configurable world seeds for reproducibility
  - Structured observations: `Observation(image, achievable, achievements, task_w, previous_action)`
  - Achievement prediction utilities (`get_possible_achievements`)
- `craftax_trainer.py` - Main training orchestrator using Hydra configs
- `networks.py` - Neural network architectures (`CraftaxObsEncoder`, `CraftaxMultiGoalObsEncoder`)

### Configuration
Hydra configs in `configs/`:
- `alg/` - Algorithm configs (ql.yaml, usfa.yaml, dyna.yaml, preplay.yaml)
- `rlenv/` - Environment configs (craftax-10m.yaml, craftax-1m-dyna.yaml)
- `config_craftax.yaml` - Main config with defaults

## Key Patterns

### Agent Structure
Agents follow the pattern:
```python
class RnnAgent(nn.Module):
    observation_encoder: nn.Module  # CraftaxObsEncoder
    rnn: vbb.ScannedRNN
    q_fn: nn.Module  # DuellingMLP
```

### Training Loop
Training uses `vbb.make_train()` which takes:
- `make_agent` - Agent constructor
- `make_loss_fn_class` - Loss function (e.g., R2D2LossFn)
- `make_optimizer` - Optax optimizer
- `make_actor` - Action selection
- `ObserverCls` - Logging observer

### Structured Observations
When `STRUCTURED_INPUTS=True`, observations contain:
- `image` - Craftax symbolic observation
- `achievable` - Binary vector of currently achievable achievements
- `achievements` - Achievements completed this timestep
- `task_w` - Task weight vector (for USFA/multitask)
