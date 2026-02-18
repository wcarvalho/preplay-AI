# Migration Guide: Making Algorithms Self-Contained

This document describes how to migrate algorithms to use `base_algorithm.py` instead of `jaxneurorl` library imports. This makes algorithms self-contained with only two dependencies: the algorithm file itself and `base_algorithm.py`.

## Overview

The migration consolidates components from multiple `jaxneurorl` modules into a single `base_algorithm.py` file:

| Source File | Components |
|-------------|------------|
| `jaxneurorl/agents/basics.py` | `StepType`, `TimeStep` |
| `jaxneurorl/agents/value_based_basics.py` | `ScannedRNN`, `RlRnnCell`, `DummyRNN`, `RecurrentLossFn`, `CustomTrainState`, `RNNInput`, `Actor`, `Transition`, `RunnerState`, `AcmeBatchData`, `make_train`, `collect_trajectory`, `learn_step`, `log_performance`, type aliases |
| `jaxneurorl/agents/qlearning.py` | `MLP`, `epsilon_greedy_act`, `LinearDecayEpsilonGreedy`, `FixedEpsilonGreedy`, `make_logger`, `make_optimizer`, `make_actor` |
| `jaxneurorl/losses.py` | `q_learning_lambda_target`, `q_learning_lambda_td` |
| `jaxneurorl/loggers.py` | `Logger`, `default_make_logger`, `default_gradient_logger`, `default_learner_logger`, `default_experience_logger` |
| `jaxneurorl/observers.py` | `Observer`, `BasicObserver`, `BasicObserverState` |
| `jaxneurorl/agents/value_based_pqn.py` | `BatchRenorm`, `get_activation_fn` |

## Migration Steps for an Algorithm File

### Step 1: Update Imports

Replace jaxneurorl imports with base_algorithm imports.

**Before:**
```python
from jaxneurorl.agents.basics import TimeStep
from jaxneurorl.agents import value_based_basics as vbb
from jaxneurorl.agents.qlearning import *
from jaxneurorl import losses
```

**After:**
```python
import base_algorithm as base
from base_algorithm import TimeStep
```

### Step 2: Update Type Aliases

Update type aliases to use base_algorithm.

**Before:**
```python
RNNInput = vbb.RNNInput
```

**After:**
```python
RNNInput = base.RNNInput
```

### Step 3: Update Class References

Update class inheritance and references.

**Before:**
```python
rnn: vbb.ScannedRNN
class R2D2LossFn(vbb.RecurrentLossFn):
```

**After:**
```python
rnn: base.ScannedRNN
class R2D2LossFn(base.RecurrentLossFn):
```

### Step 4: Update Loss Function References

Update loss function calls.

**Before:**
```python
losses.q_learning_lambda_td(...)
```

**After:**
```python
base.q_learning_lambda_td(...)
```

### Step 5: Update MLP References

Update MLP usage.

**Before:**
```python
MLP(...)  # from qlearning import *
```

**After:**
```python
base.MLP(...)
```

### Step 6: Re-export Functions for craftax_trainer.py

At the bottom of your algorithm file, re-export functions that craftax_trainer.py expects:

```python
# Re-export functions from base_algorithm that craftax_trainer.py expects
make_optimizer = base.make_optimizer
make_actor = base.make_actor
```

## Migration Steps for networks.py

Update the import:

**Before:**
```python
from jaxneurorl.agents.value_based_pqn import MLP, BatchRenorm, get_activation_fn
```

**After:**
```python
from base_algorithm import MLP, BatchRenorm, get_activation_fn
```

## Migration Steps for craftax_trainer.py

### Step 1: Add Import for New Algorithm

```python
import her  # or your new algorithm
import base_algorithm
```

### Step 2: Add Algorithm Block

Add a new elif block in the `run_single` function:

```python
elif config["ALG"] == "her":
    train_fn = base_algorithm.make_train(
        config=config,
        save_path=save_path,
        online_trajectory_log_fn=default_craftax_log_fn,
        env=vec_env,
        make_agent=her.make_craftax_agent,
        make_optimizer=her.make_optimizer,
        make_loss_fn_class=her.make_loss_fn_class,
        make_actor=her.make_actor,
        make_logger=partial(
            make_logger, learner_log_extra=her.learner_log_extra
        ),
        train_env_params=env_params,
        test_env_params=test_env_params,
        ObserverCls=craftax_observer.Observer,
        vmap_env=vmap_env,
    )
```

### Step 3: Add Sweep Config

Add a sweep config in the `sweep()` function as an `elif` block:

```python
elif search == "her":
    sweep_config = {
        "metric": metric,
        "parameters": {
            "NUM_ENV_SEEDS": {"values": [0]},
            "SEED": {"values": list(range(1, 2))},
        },
        "overrides": ["alg=her", "rlenv=craftax-10m", "user=wilka"],
        "group": "her-1",
    }
```

## Creating Config File

Create `configs/alg/her.yaml` with:

```yaml
ALG: 'her'

# Copy parameters from ql.yaml and adjust as needed
```

## Testing

Run the following command to test:

```bash
HYDRA_FULL_ERROR=1 JAX_TRACEBACK_FILTERING=off python -m ipdb -c continue craftax_trainer.py \
  app.debug=True \
  app.wandb=False \
  app.search=her
```

## Example: Complete Migration (her.py)

The `her.py` file demonstrates a complete migration. Key changes:

1. **Imports**: Only imports from `base_algorithm`, `flax`, `jax`, `rlax`, `wandb`, and local `networks.py`
2. **Types**: Uses `base.RNNInput`, `base.ScannedRNN`, `base.RecurrentLossFn`, `base.MLP`
3. **Loss Functions**: Uses `base.q_learning_lambda_td`
4. **Re-exports**: `make_optimizer = base.make_optimizer` and `make_actor = base.make_actor`

## Files Modified in her Migration

1. **Created**: `base_algorithm.py` (~1200 lines) - consolidated jaxneurorl components
2. **Modified**: `her.py` - updated imports and references
3. **Modified**: `networks.py` - updated import for MLP, BatchRenorm, get_activation_fn
4. **Modified**: `craftax_trainer.py` - added her algorithm support and sweep config
5. **Created**: `configs/alg/her.yaml` - algorithm configuration

## Notes

- `base_algorithm.py` contains everything needed for value-based RL algorithms
- The `MLP` class in `base_algorithm.py` supports `norm_type` and `activation` parameters
- The training loop in `make_train` handles prioritized replay buffer, target network updates, and logging
- Observer classes track episode statistics for logging
