"""
Self-contained base algorithm module for value-based reinforcement learning.

This module provides the foundational components for reinforcement learning algorithms:
- StepType/TimeStep: Basic RL data structures
- Neural Networks: MLP, BatchRenorm, ScannedRNN, DummyRNN
- Loss Functions: q_learning_lambda_target, q_learning_lambda_td
- Episode Metrics: compute_episode_metrics
- Loggers: Logger and default logging functions
- Training Infrastructure: make_train, collect_trajectory, learn_step
"""

from pprint import pprint
from tqdm import tqdm
import collections
import copy
import functools
import os
import pickle
from functools import partial
from typing import (
  Any,
  Callable,
  Dict,
  NamedTuple,
  Optional,
  Tuple,
  TypeVar,
  Union,
)

import chex
import flashbax as fbx
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import tree
import wandb
from flax import struct
from flax.core import FrozenDict
from flax.linen.normalization import _canonicalize_axes, _compute_stats, _normalize
from flax.struct import field
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict
from gymnax.environments import environment
from jax.nn import initializers
from safetensors.flax import save_file

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


##############################
# StepType / TimeStep (from basics.py)
##############################


class StepType(jnp.uint8):
  FIRST: jax.Array = jnp.asarray(0, dtype=jnp.uint8)
  MID: jax.Array = jnp.asarray(1, dtype=jnp.uint8)
  LAST: jax.Array = jnp.asarray(2, dtype=jnp.uint8)


class TimeStep(struct.PyTreeNode):
  state: struct.PyTreeNode

  step_type: StepType
  reward: jax.Array
  discount: jax.Array
  observation: jax.Array

  def first(self):
    return self.step_type == StepType.FIRST

  def mid(self):
    return self.step_type == StepType.MID

  def last(self):
    return self.step_type == StepType.LAST


##############################
# Data types
##############################
Config = Dict
Action = flax.struct.PyTreeNode
Agent = nn.Module
PRNGKey = jax.random.PRNGKey
Params = flax.core.FrozenDict
AgentState = flax.struct.PyTreeNode
Predictions = flax.struct.PyTreeNode
Env = environment.Environment
EnvParams = environment.EnvParams
Array = Any
Shape = Tuple[int, ...]
Dtype = Any

Number = Union[int, float, np.float32, jnp.float32]

ActorStepFn = Callable[
  [TrainState, AgentState, TimeStep, PRNGKey], Tuple[Predictions, AgentState]
]
EnvStepFn = Callable[[PRNGKey, TimeStep, Action, EnvParams], TimeStep]


class RNNInput(NamedTuple):
  obs: jax.Array
  reset: jax.Array


class Actor(NamedTuple):
  train_step: ActorStepFn
  eval_step: ActorStepFn


class Transition(NamedTuple):
  timestep: TimeStep
  action: jax.Array
  extras: Optional[FrozenDict[str, jax.Array]] = None


class RunnerState(NamedTuple):
  train_state: TrainState
  timestep: TimeStep
  agent_state: jax.Array
  rng: jax.random.PRNGKey
  buffer_state: Optional[fbx.trajectory_buffer.TrajectoryBufferState] = None


class AcmeBatchData(flax.struct.PyTreeNode):
  timestep: TimeStep
  action: jax.Array
  extras: FrozenDict

  @property
  def is_last(self):
    return self.timestep.last()

  @property
  def discount(self):
    return self.timestep.discount

  @property
  def reward(self):
    return self.timestep.reward


class CustomTrainState(TrainState):
  target_network_params: flax.core.FrozenDict
  timesteps: int = 0
  n_updates: int = 0
  n_logs: int = 0


##############################
# Neural Network Components (from value_based_pqn.py)
##############################


class BatchRenorm(nn.Module):
  """BatchRenorm Module, implemented based on the Batch Renormalization paper
  (https://arxiv.org/abs/1702.03275) and adapted from Flax's BatchNorm implementation.
  """

  use_running_average: Optional[bool] = None
  axis: int = -1
  momentum: float = 0.999
  epsilon: float = 0.001
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Callable[[jax.random.PRNGKey, Shape, Dtype], Array] = initializers.zeros
  scale_init: Callable[[jax.random.PRNGKey, Shape, Dtype], Array] = initializers.ones
  axis_name: Optional[str] = None
  axis_index_groups: Any = None
  use_fast_variance: bool = True

  @nn.compact
  def __call__(self, x, use_running_average: Optional[bool] = None):
    use_running_average = nn.merge_param(
      "use_running_average", self.use_running_average, use_running_average
    )
    feature_axes = _canonicalize_axes(x.ndim, self.axis)
    reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
    feature_shape = [x.shape[ax] for ax in feature_axes]

    ra_mean = self.variable(
      "batch_stats",
      "mean",
      lambda s: jnp.zeros(s, jnp.float32),
      feature_shape,
    )
    ra_var = self.variable(
      "batch_stats", "var", lambda s: jnp.ones(s, jnp.float32), feature_shape
    )

    r_max = self.variable(
      "batch_stats",
      "r_max",
      lambda s: s,
      3,
    )
    d_max = self.variable(
      "batch_stats",
      "d_max",
      lambda s: s,
      5,
    )
    steps = self.variable(
      "batch_stats",
      "steps",
      lambda s: s,
      0,
    )

    if use_running_average:
      mean, var = ra_mean.value, ra_var.value
      custom_mean = mean
      custom_var = var
    else:
      mean, var = _compute_stats(
        x,
        reduction_axes,
        dtype=self.dtype,
        axis_name=self.axis_name if not self.is_initializing() else None,
        axis_index_groups=self.axis_index_groups,
        use_fast_variance=self.use_fast_variance,
      )
      custom_mean = mean
      custom_var = var
      if not self.is_initializing():
        # The code below is implemented following the Batch Renormalization paper
        r = 1
        d = 0
        std = jnp.sqrt(var + self.epsilon)
        ra_std = jnp.sqrt(ra_var.value + self.epsilon)
        r = jax.lax.stop_gradient(std / ra_std)
        r = jnp.clip(r, 1 / r_max.value, r_max.value)
        d = jax.lax.stop_gradient((mean - ra_mean.value) / ra_std)
        d = jnp.clip(d, -d_max.value, d_max.value)
        tmp_var = var / (r**2)
        tmp_mean = mean - d * jnp.sqrt(custom_var) / r

        # Warm up batch renorm for 1000 steps to build up proper running statistics
        warmed_up = jnp.greater_equal(steps.value, 1000).astype(jnp.float32)
        custom_var = warmed_up * tmp_var + (1.0 - warmed_up) * custom_var
        custom_mean = warmed_up * tmp_mean + (1.0 - warmed_up) * custom_mean

        ra_mean.value = self.momentum * ra_mean.value + (1 - self.momentum) * mean
        ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var
        steps.value += 1

    return _normalize(
      self,
      x,
      custom_mean,
      custom_var,
      reduction_axes,
      feature_axes,
      self.dtype,
      self.param_dtype,
      self.epsilon,
      self.use_bias,
      self.use_scale,
      self.bias_init,
      self.scale_init,
    )


def get_activation_fn(k: str):
  if k == "relu":
    return nn.relu
  elif k == "leaky_relu":
    return nn.leaky_relu
  elif k == "tanh":
    return nn.tanh
  raise NotImplementedError(k)


class MLP(nn.Module):
  hidden_dim: int
  out_dim: Optional[int] = 0
  num_layers: int = 1
  norm_type: str = "none"
  activation: str = "relu"
  activate_final: bool = True
  use_bias: bool = False

  @nn.compact
  def __call__(self, x, train: bool = False):
    activation_fn = get_activation_fn(self.activation)

    if self.norm_type == "none":
      normalize = lambda x: x
    elif self.norm_type == "layer_norm":
      normalize = lambda x: nn.LayerNorm()(x)
    elif self.norm_type == "batch_norm":
      normalize = lambda x: BatchRenorm(use_running_average=not train)(x)
    else:
      raise NotImplementedError(self.norm_type)

    for _ in range(self.num_layers):
      x = nn.Dense(self.hidden_dim, use_bias=self.use_bias)(x)
      x = normalize(x)
      x = activation_fn(x)

    if self.out_dim == 0:
      return x

    x = nn.Dense(self.out_dim or self.hidden_dim, use_bias=self.use_bias)(x)

    if self.activate_final:
      x = normalize(x)
      x = activation_fn(x)

    return x


##############################
# RNN Components (from value_based_basics.py)
##############################


class RlRnnCell(nn.Module):
  hidden_dim: int
  cell_type: str = "OptimizedLSTMCell"

  def setup(self):
    cell_constructor = getattr(nn, self.cell_type)
    self.cell = cell_constructor(self.hidden_dim)
    nn.OptimizedLSTMCell

  def __call__(
    self,
    state: struct.PyTreeNode,
    x: jax.Array,
    reset: jax.Array,
    rng: PRNGKey,
  ):
    """Applies the module."""

    def conditional_reset(cond, init, prior):
      if cond.ndim == 1:
        return jnp.where(cond[:, np.newaxis], init, prior)
      else:
        return jnp.where(cond[np.newaxis], init, prior)

    init_state = self.initialize_carry(rng=rng, batch_dims=x.shape[:-1])
    if "lstm" in self.cell_type.lower():
      input_state = tuple(
        conditional_reset(reset, init, prior) for init, prior in zip(init_state, state)
      )
    elif "gru" in self.cell_type.lower():
      input_state = conditional_reset(reset, init_state, state)
    else:
      raise NotImplementedError(self.cell_type)

    return self.cell(input_state, x)

  def output_from_state(self, state):
    if "lstm" in self.cell_type.lower():
      return state[1]
    elif "gru" in self.cell_type.lower():
      return state
    else:
      raise NotImplementedError(self.cell_type)

  def initialize_carry(
    self, rng: PRNGKey, batch_dims: Tuple[int, ...]
  ) -> Tuple[jax.Array, jax.Array]:
    """Initialize the RNN cell carry."""
    return self.cell.initialize_carry(rng, input_shape=batch_dims + (1,))


class ScannedRNN(nn.Module):
  hidden_dim: int
  cell_type: str = "OptimizedLSTMCell"
  unroll_output_state: bool = False

  def initialize_carry(self, *args, **kwargs):
    """Initializes the RNN state."""
    return self.cell.initialize_carry(*args, **kwargs)

  def setup(self):
    self.cell = RlRnnCell(
      cell_type=self.cell_type, hidden_dim=self.hidden_dim, name=self.cell_type
    )

  def __call__(self, state, x: RNNInput, rng: PRNGKey):
    """Applies the module."""
    return self.cell(state=state, x=x.obs, reset=x.reset, rng=rng)

  def unroll(self, state, xs: RNNInput, rng: PRNGKey):
    """Unroll over sequence."""

    def body_fn(cell, state, inputs):
      x, reset = inputs
      state, out = cell(state, x, reset, rng)
      if self.unroll_output_state:
        return state, state
      return state, out

    scan = nn.scan(
      body_fn,
      variable_broadcast="params",
      split_rngs={"params": False},
      in_axes=0,
      out_axes=0,
    )

    return scan(self.cell, state, (xs.obs, xs.reset))

  def output_from_state(self, state):
    return self.cell.output_from_state(state)


class DummyRNN(nn.Module):
  hidden_dim: int = 0
  cell_type: str = "OptimizedLSTMCell"
  unroll_output_state: bool = False

  def __call__(self, state, x: RNNInput, rng: PRNGKey):
    return state, x.obs

  def unroll(self, state, xs: RNNInput, rng: PRNGKey):
    if self.unroll_output_state:
      return state, (xs.obs, xs.obs)
    return state, xs.obs

  def output_from_state(self, state):
    return state

  def initialize_carry(
    self, rng: PRNGKey, batch_dims: Tuple[int, ...]
  ) -> Tuple[jax.Array, jax.Array]:
    del rng
    mem_shape = batch_dims + (self.hidden_dim,)
    return jnp.zeros(mem_shape), jnp.zeros(mem_shape)


##############################
# Loss Functions (from losses.py)
##############################


def q_learning_lambda_target(
  q_t: jax.Array,
  r_t: jax.Array,
  discount_t: jax.Array,
  is_last_t: jax.Array,
  a_t: jax.Array,
  lambda_: jax.Array,
  stop_target_gradients: bool = True,
) -> jax.Array:
  """MINOR change to rlax.lambda_returns to incorporate is_last_t."""
  v_t = rlax.batched_index(q_t, a_t)
  lambda_ = jax.numpy.ones_like(discount_t) * lambda_ * (1 - is_last_t)
  target_tm1 = rlax.lambda_returns(
    r_t, discount_t, v_t, lambda_, stop_target_gradients=stop_target_gradients
  )
  return target_tm1


def q_learning_lambda_td(
  q_tm1: jax.Array,
  a_tm1: jax.Array,
  target_q_t: jax.Array,
  a_t: jax.Array,
  r_t: jax.Array,
  discount_t: jax.Array,
  is_last_t: jax.Array,
  lambda_: jax.Array,
  stop_target_gradients: bool = True,
  tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR,
):
  """Q-learning lambda TD error computation."""
  q_tm1_transformed = tx_pair.apply(q_tm1)
  target_q_t_transformed = tx_pair.apply(target_q_t)

  v_tm1 = rlax.batched_index(q_tm1_transformed, a_tm1)
  target_mt1 = q_learning_lambda_target(
    r_t=r_t,
    q_t=target_q_t_transformed,
    a_t=a_t,
    discount_t=discount_t,
    is_last_t=is_last_t,
    lambda_=lambda_,
    stop_target_gradients=stop_target_gradients,
  )

  v_tm1, target_mt1 = tx_pair.apply_inv(v_tm1), tx_pair.apply_inv(target_mt1)

  return v_tm1, target_mt1


##############################
# Episode Metrics
##############################


def compute_episode_metrics(trajectory):
  """Compute avg episode return and length from trajectory [T, B]."""
  rewards = trajectory.timestep.reward
  is_last = trajectory.timestep.last()
  is_first = trajectory.timestep.first()

  def accumulate_fn(carry, step):
    running_return, running_length = carry
    reward, first, last = step
    running_return = jnp.where(first, 0.0, running_return)
    running_length = jnp.where(first, 0, running_length)
    running_return = running_return + reward
    running_length = running_length + 1
    return (running_return, running_length), (
      running_return * last,
      running_length * last,
      last,
    )

  n_envs = rewards.shape[1]
  init = (jnp.zeros(n_envs), jnp.zeros(n_envs, dtype=jnp.int32))
  _, (ep_returns, ep_lengths, dones) = jax.lax.scan(
    accumulate_fn, init, (rewards, is_first, is_last)
  )
  mask = dones.astype(jnp.float32)
  n_episodes = mask.sum()
  avg_return = jnp.where(n_episodes > 0, (ep_returns * mask).sum() / n_episodes, 0.0)
  avg_length = jnp.where(n_episodes > 0, (ep_lengths * mask).sum() / n_episodes, 0.0)
  return {"avg_return": avg_return, "avg_length": avg_length, "n_episodes": n_episodes}


##############################
# Loggers (from loggers.py)
##############################


def default_gradient_logger(
  train_state: TrainState, gradients: dict, key: str = "gradients"
):
  if "params" in gradients:
    gradients = gradients["params"]
  gradient_metrics = {
    f"{key}/{k}_norm": optax.global_norm(v) for k, v in gradients.items()
  }
  if wandb.run is not None:
    gradient_metrics.update(
      {
        f"{key}/num_actor_steps": train_state.timesteps,
        f"{key}/num_learner_updates": train_state.n_updates,
      }
    )
    wandb.log(gradient_metrics)


def default_learner_logger(
  train_state: TrainState, learner_metrics: dict, key: str = "learner"
):
  learner_metrics = jax.tree_util.tree_map(lambda x: x.mean(), learner_metrics)
  metrics = {f"{key}/{k}": v for k, v in learner_metrics.items()}
  metrics.update(
    {
      f"{key}/num_actor_steps": train_state.timesteps,
      f"{key}/num_learner_updates": train_state.n_updates,
    }
  )
  if wandb.run is not None:
    wandb.log(metrics)


def default_experience_logger(
  train_state: TrainState,
  trajectory,
  key: str = "train",
  **kwargs,
):
  ep_metrics = compute_episode_metrics(trajectory)
  metrics = {
    f"{key}/avg_episode_length": ep_metrics["avg_length"],
    f"{key}/avg_episode_return": ep_metrics["avg_return"],
    f"{key}/num_actor_steps": train_state.timesteps,
    f"{key}/num_learner_updates": train_state.n_updates,
  }
  if wandb.run is not None:
    wandb.log(metrics)


@struct.dataclass
class Logger:
  gradient_logger: Callable[[TrainState, dict, str], Any]
  learner_logger: Callable[[TrainState, dict, str], Any]
  experience_logger: Callable[[TrainState, Any, str], Any]
  learner_log_extra: Optional[Callable[[Any], Any]] = None


def default_make_logger(
  config: dict, env: environment.Environment, env_params: environment.EnvParams
):
  return Logger(
    gradient_logger=default_gradient_logger,
    learner_logger=default_learner_logger,
    experience_logger=default_experience_logger,
  )


##############################
# Training Infrastructure
##############################


def save_training_state(
  params: Dict,
  config: Dict,
  save_path: str,
  alg_name: str,
  idx: int = None,
  n_updates: int = None,
) -> None:
  """Save model parameters and config to disk."""
  os.makedirs(save_path, exist_ok=True)

  if idx is not None:
    param_path = os.path.join(save_path, f"{alg_name}_{idx}.safetensors")
  else:
    param_path = os.path.join(save_path, f"{alg_name}.safetensors")
  flattened_dict = flatten_dict(params, sep=",")
  save_file(flattened_dict, param_path)

  prefix = f"update {n_updates}: " if n_updates is not None else ""
  print(f"{prefix}Parameters saved in {param_path}")

  config_path = os.path.join(save_path, f"{alg_name}.config")
  if not os.path.exists(config_path):
    with open(config_path, "wb") as f:
      pickle.dump(config, f)
    print(f"{prefix}Config saved in {config_path}")


def masked_mean(x, mask):
  z = jnp.multiply(x, mask)
  return (z.sum(0)) / (mask.sum(0) + 1e-5)


def batch_to_sequence(values: jax.Array) -> jax.Array:
  return jax.tree_util.tree_map(
    lambda x: jnp.transpose(x, axes=(1, 0, *range(2, len(x.shape)))), values
  )


@struct.dataclass
class RecurrentLossFn:
  """Recurrent loss function with burn-in structured modelled after R2D2."""

  network: nn.Module
  discount: float = 0.99
  lambda_: float = 0.9
  step_cost: float = 0.001
  max_priority_weight: float = 0.0
  importance_sampling_exponent: float = 0.0
  tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR
  burn_in_length: int = None

  data_wrapper: flax.struct.PyTreeNode = AcmeBatchData
  logger: Logger = Logger

  def __call__(
    self,
    params: Params,
    target_params: Params,
    batch: fbx.trajectory_buffer.BufferSample,
    key_grad: PRNGKey,
    steps: int,
  ):
    """Calculate a loss on a single batch of data."""
    unroll = functools.partial(self.network.apply, method=self.network.unroll)

    online_state = batch.experience.extras.get("agent_state")
    online_state = jax.tree_util.tree_map(lambda x: x[:, 0], online_state)
    target_state = online_state

    data = batch_to_sequence(batch.experience)

    burn_in_length = self.burn_in_length
    if burn_in_length:
      burn_data = jax.tree_util.tree_map(lambda x: x[:burn_in_length], data)
      key_grad, rng_1, rng_2 = jax.random.split(key_grad, 3)
      _, online_state = unroll(params, online_state, burn_data.timestep, rng_1)
      _, target_state = unroll(target_params, target_state, burn_data.timestep, rng_2)
      data = jax.tree_util.tree_map(lambda seq: seq[burn_in_length:], data)

    key_grad, rng_1, rng_2 = jax.random.split(key_grad, 3)
    online_preds, _ = unroll(params, online_state, data.timestep, rng_1)
    target_preds, _ = unroll(target_params, target_state, data.timestep, rng_2)

    data = self.data_wrapper(
      timestep=data.timestep, action=data.action, extras=data.extras
    )

    elemwise_error, batch_loss, metrics = self.error(
      data=data,
      online_preds=online_preds,
      online_state=online_state,
      target_preds=target_preds,
      target_state=target_state,
      params=params,
      target_params=target_params,
      steps=steps,
      key_grad=key_grad,
    )

    abs_td_error = jnp.abs(elemwise_error).astype(jnp.float32)
    max_priority = self.max_priority_weight * jnp.max(abs_td_error, axis=0)
    mean_priority = (1 - self.max_priority_weight) * jnp.mean(abs_td_error, axis=0)
    priorities = max_priority + mean_priority

    importance_weights = (1.0 / (batch.probabilities + 1e-6)).astype(jnp.float32)
    importance_weights **= self.importance_sampling_exponent
    importance_weights /= jnp.max(importance_weights)
    batch_loss = jnp.mean(importance_weights * batch_loss)

    updates = dict(
      priorities=priorities,
    )

    return batch_loss, (updates, metrics)


AgentResetFn = Callable[[Params, TimeStep], AgentState]
EnvResetFn = Callable[[PRNGKey, EnvParams], TimeStep]
MakeAgentFn = Callable[
  [Config, Env, EnvParams, TimeStep, jax.random.PRNGKey],
  Tuple[nn.Module, Params, AgentResetFn],
]
MakeOptimizerFn = Callable[[Config], optax.GradientTransformation]
MakeLossFnClass = Callable[[Config], RecurrentLossFn]
MakeActorFn = Callable[[Config, Agent], Actor]
MakeLoggerFn = Callable[[Config, Env, EnvParams, Agent], Logger]


def log_params(params):
  size_of_tree = lambda t: sum(tree.flatten(t))
  sizes = tree.map_structure(jnp.size, params)
  total_params = size_of_tree(sizes.values())
  print("=" * 50)
  print(f"Total number of params: {total_params:,}")
  [print(f"\t{k}: {size_of_tree(v.values()):,}") for k, v in sizes.items()]


def collect_trajectory(
  runner_state: RunnerState,
  num_steps: int,
  actor_step_fn: ActorStepFn,
  env_step_fn: EnvStepFn,
  env_params: EnvParams,
):
  def _env_step(rs: RunnerState, unused):
    del unused
    rng = rs.rng
    prior_timestep = rs.timestep
    prior_agent_state = rs.agent_state

    rng, rng_a, rng_s = jax.random.split(rng, 3)

    preds, action, agent_state = actor_step_fn(
      rs.train_state, prior_agent_state, prior_timestep, rng_a
    )

    transition = Transition(
      prior_timestep,
      action=action,
      extras=FrozenDict(preds=preds, agent_state=prior_agent_state),
    )

    timestep = env_step_fn(rng_s, prior_timestep, action, env_params)

    rs = rs._replace(
      timestep=timestep,
      agent_state=agent_state,
      rng=rng,
    )

    return rs, transition

  return jax.lax.scan(f=_env_step, init=runner_state, xs=None, length=num_steps)


def learn_step(
  train_state: CustomTrainState,
  rng: jax.random.PRNGKey,
  buffer,
  buffer_state,
  loss_fn,
):
  rng, _rng = jax.random.split(rng)
  learn_trajectory = buffer.sample(buffer_state, _rng)

  rng, _rng = jax.random.split(rng)
  (_, (updates, metrics)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
    train_state.params,
    train_state.target_network_params,
    learn_trajectory,
    _rng,
    train_state.n_updates,
  )

  train_state = train_state.apply_gradients(grads=grads)
  train_state = train_state.replace(
    n_updates=train_state.n_updates + 1,
  )

  new_priorities = updates["priorities"]
  buffer_state = buffer.set_priorities(
    buffer_state, learn_trajectory.indices, new_priorities
  )

  metrics.update(
    {
      "0.grad_norm": optax.global_norm(grads),
      "0.param_norm": optax.global_norm(train_state.params),
    }
  )

  return train_state, buffer_state, metrics, grads


def compute_performance(
  config: dict,
  agent_reset_fn: AgentResetFn,
  actor_train_step_fn: ActorStepFn,
  actor_eval_step_fn: ActorStepFn,
  env_reset_fn: EnvResetFn,
  env_step_fn: EnvStepFn,
  train_env_params: EnvParams,
  test_env_params: EnvParams,
  runner_state: RunnerState,
):
  results = {}

  ########################
  # TESTING PERFORMANCE
  ########################
  rng = runner_state.rng
  rng, _rng = jax.random.split(rng)
  init_timestep = env_reset_fn(_rng, test_env_params)

  rng, _rng = jax.random.split(rng)
  init_agent_state = agent_reset_fn(
    runner_state.train_state.params, init_timestep, _rng
  )

  rng, _rng = jax.random.split(rng)
  eval_runner_state = RunnerState(
    train_state=runner_state.train_state,
    timestep=init_timestep,
    agent_state=init_agent_state,
    rng=_rng,
  )

  _, trajectory = collect_trajectory(
    runner_state=eval_runner_state,
    num_steps=config["EVAL_STEPS"] * config["EVAL_EPISODES"],
    actor_step_fn=actor_eval_step_fn,
    env_step_fn=env_step_fn,
    env_params=test_env_params,
  )
  results["eval_trajectory"] = trajectory

  ########################
  # TRAINING PERFORMANCE
  ########################
  rng = runner_state.rng
  rng, _rng = jax.random.split(rng)
  init_timestep = env_reset_fn(_rng, train_env_params)

  rng, _rng = jax.random.split(rng)
  init_agent_state = agent_reset_fn(
    runner_state.train_state.params, init_timestep, _rng
  )

  rng, _rng = jax.random.split(rng)
  eval_runner_state = RunnerState(
    train_state=runner_state.train_state,
    timestep=init_timestep,
    agent_state=init_agent_state,
    rng=_rng,
  )

  _, trajectory = collect_trajectory(
    runner_state=eval_runner_state,
    num_steps=config["EVAL_STEPS"] * config["EVAL_EPISODES"],
    actor_step_fn=actor_train_step_fn,
    env_step_fn=env_step_fn,
    env_params=train_env_params,
  )
  results["actor_trajectory"] = trajectory

  return results


def make_train(
  config: dict,
  env: environment.Environment,
  train_env_params: environment.EnvParams,
  make_agent: MakeAgentFn,
  make_optimizer: MakeOptimizerFn,
  make_loss_fn_class: MakeLossFnClass,
  make_actor: MakeActorFn,
  make_logger: MakeLoggerFn = default_make_logger,
  test_env_params: Optional[environment.EnvParams] = None,
  vmap_env: bool = True,
  initial_params: Optional[Params] = None,
  save_path: Optional[str] = None,
  online_trajectory_log_fn=None,
):
  """Creates a train function that does learning after unrolling agent for K timesteps."""

  config["NUM_UPDATES"] = int(
    config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"] // config["TRAINING_INTERVAL"]
  )
  test_env_params = test_env_params or copy.deepcopy(train_env_params)

  if vmap_env:

    def vmap_reset(rng, env_params):
      return jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, config["NUM_ENVS"]), env_params
      )

    def vmap_step(rng, env_state, action, env_params):
      return jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        jax.random.split(rng, config["NUM_ENVS"]), env_state, action, env_params
      )

  else:
    vmap_reset = env.reset
    vmap_step = env.step

  def train(rng: jax.random.PRNGKey):
    logger = make_logger(config, env, train_env_params)

    ##############################
    # INIT ENV
    ##############################
    rng, _rng = jax.random.split(rng)
    init_timestep = vmap_reset(_rng, train_env_params)

    ##############################
    # INIT NETWORK
    ##############################
    rng, _rng = jax.random.split(rng)
    agent, network_params, agent_reset_fn = make_agent(
      config=config,
      env=env,
      env_params=train_env_params,
      example_timestep=init_timestep,
      rng=_rng,
    )
    if initial_params is not None:
      network_params = initial_params

    log_params(network_params["params"])

    rng, _rng = jax.random.split(rng)
    init_agent_state = agent_reset_fn(network_params, init_timestep, _rng)

    ##############################
    # INIT Actor
    ##############################
    rng, _rng = jax.random.split(rng)
    actor = make_actor(config=config, agent=agent, rng=_rng)

    ##############################
    # INIT OPTIMIZER
    ##############################
    tx = make_optimizer(config)

    train_state = CustomTrainState.create(
      apply_fn=agent.apply,
      params=network_params,
      target_network_params=jax.tree_util.tree_map(
        lambda x: jnp.copy(x), network_params
      ),
      tx=tx,
      timesteps=0,
      n_updates=0,
      n_logs=0,
    )

    ##############################
    # INIT BUFFER
    ##############################
    period = config.get("SAMPLING_PERIOD", 1)
    total_batch_size = config.get("TOTAL_BATCH_SIZE")
    sample_batch_size = config["BUFFER_BATCH_SIZE"]
    sample_sequence_length = config.get("SAMPLE_LENGTH")
    if sample_sequence_length is None:
      sample_sequence_length = total_batch_size // sample_batch_size

    buffer = fbx.make_prioritised_trajectory_buffer(
      max_length_time_axis=config["BUFFER_SIZE"] // config["NUM_ENVS"],
      min_length_time_axis=sample_sequence_length,
      add_batch_size=config["NUM_ENVS"],
      sample_batch_size=config["BUFFER_BATCH_SIZE"],
      sample_sequence_length=sample_sequence_length,
      period=period,
      priority_exponent=config.get("PRIORITY_EXPONENT", 0.9),
    )
    buffer = buffer.replace(
      init=jax.jit(buffer.init),
      add=jax.jit(buffer.add, donate_argnums=0),
      sample=jax.jit(buffer.sample),
      can_sample=jax.jit(buffer.can_sample),
    )

    dummy_rng = jax.random.PRNGKey(0)
    init_preds, action, _ = actor.train_step(
      train_state, init_agent_state, init_timestep, dummy_rng
    )
    init_transition = Transition(
      init_timestep,
      action=action,
      extras=FrozenDict(preds=init_preds, agent_state=init_agent_state),
    )
    init_transition_example = jax.tree_util.tree_map(lambda x: x[0], init_transition)

    buffer_state = buffer.init(init_transition_example)

    ##############################
    # INIT LOSS FN
    ##############################
    loss_fn_class = make_loss_fn_class(config)
    loss_fn = loss_fn_class(network=agent, logger=logger)

    dummy_rng = jax.random.PRNGKey(0)

    # Add dummy transitions so buffer can be sampled for shape inference
    dummy_transitions = jax.tree_util.tree_map(
      lambda x: jnp.repeat(x[:, None], sample_sequence_length, axis=1),
      init_transition,
    )
    dummy_buffer_state = buffer.add(buffer_state, dummy_transitions)

    _, _, dummy_metrics, dummy_grads = learn_step(
      train_state=train_state,
      rng=dummy_rng,
      buffer=buffer,
      buffer_state=dummy_buffer_state,
      loss_fn=loss_fn,
    )

    # Re-initialize buffer_state since the original was donated/deleted by buffer.add
    buffer_state = buffer.init(init_transition_example)

    ##############################
    # DEFINE TRAINING LOOP
    ##############################
    print("=" * 50)
    print("TRAINING")
    print("=" * 50)

    loss_name = loss_fn.__class__.__name__

    # Create dummy traj_batch for initial carry (shape: TRAINING_INTERVAL, NUM_ENVS, ...)
    dummy_traj_batch = jax.tree_util.tree_map(
      lambda x: jnp.zeros((config["TRAINING_INTERVAL"],) + x.shape, dtype=x.dtype),
      init_transition,
    )

    ##############################
    # WARMUP PHASE: fill buffer before learning
    ##############################
    warmup_steps = (
      config["LEARNING_STARTS"] // config["NUM_ENVS"] // config["TRAINING_INTERVAL"]
    )
    # Ensure buffer has enough data to sample
    min_warmup = (sample_sequence_length // config["TRAINING_INTERVAL"]) + 1
    warmup_steps = max(warmup_steps, min_warmup)

    def _warmup_step(runner_state, unused):
      del unused
      runner_state, traj_batch = collect_trajectory(
        runner_state=runner_state,
        num_steps=config["TRAINING_INTERVAL"],
        actor_step_fn=actor.train_step,
        env_step_fn=vmap_step,
        env_params=train_env_params,
      )
      train_state = runner_state.train_state
      timesteps = (
        train_state.timesteps + config["NUM_ENVS"] * config["TRAINING_INTERVAL"]
      )
      train_state = train_state.replace(timesteps=timesteps)

      buffer_traj_batch = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(x, 0, 1), traj_batch
      )
      buffer_state = buffer.add(runner_state.buffer_state, buffer_traj_batch)

      runner_state = runner_state._replace(
        train_state=train_state,
        buffer_state=buffer_state,
      )
      return runner_state, None

    rng, _rng = jax.random.split(rng)
    runner_state = RunnerState(
      train_state=train_state,
      buffer_state=buffer_state,
      timestep=init_timestep,
      agent_state=init_agent_state,
      rng=_rng,
    )

    print(f"Warmup: {warmup_steps} steps to fill buffer")

    @jax.jit
    def run_warmup(runner_state):
      return jax.lax.scan(_warmup_step, runner_state, None, warmup_steps)

    runner_state, _ = run_warmup(runner_state)

    ##############################
    # TRAINING LOOP (chunked Python loop)
    ##############################
    num_train_steps = config["NUM_UPDATES"] - warmup_steps
    num_chunks = min(config.get("NUM_CHUNKS", 100), num_train_steps)
    inner_steps = max(1, num_train_steps // num_chunks)
    # Adjust num_chunks to not exceed total steps
    num_chunks = num_train_steps // inner_steps

    print(
      f"Training: {num_train_steps} updates in {num_chunks} chunks of {inner_steps} steps"
    )

    def _train_step(carry, unused):
      del unused
      runner_state, _, _, _ = carry

      # 1. Collect trajectory (K steps)
      runner_state, traj_batch = collect_trajectory(
        runner_state=runner_state,
        num_steps=config["TRAINING_INTERVAL"],
        actor_step_fn=actor.train_step,
        env_step_fn=vmap_step,
        env_params=train_env_params,
      )

      rng = runner_state.rng
      train_state = runner_state.train_state
      buffer_state = runner_state.buffer_state

      timesteps = (
        train_state.timesteps + config["NUM_ENVS"] * config["TRAINING_INTERVAL"]
      )
      train_state = train_state.replace(timesteps=timesteps)

      # 2. Add to buffer (swap axes: num_steps x num_envs -> num_envs x num_steps)
      buffer_traj_batch = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(x, 0, 1), traj_batch
      )
      buffer_state = buffer.add(buffer_state, buffer_traj_batch)

      # 3. Learn step (unconditional — buffer is guaranteed full after warmup)
      rng, _rng = jax.random.split(rng)
      train_state, buffer_state, metrics, grads = learn_step(
        train_state=train_state,
        rng=_rng,
        buffer=buffer,
        buffer_state=buffer_state,
        loss_fn=loss_fn,
      )

      # 4. Target network update
      train_state = jax.lax.cond(
        train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
        lambda train_state: train_state.replace(
          target_network_params=train_state.params,
        ),
        lambda train_state: train_state,
        operand=train_state,
      )

      # 5. Return updated state (metrics/grads/traj overwritten each step, only last survives)
      next_runner_state = runner_state._replace(
        train_state=train_state, buffer_state=buffer_state, rng=rng
      )

      return (next_runner_state, metrics, grads, traj_batch), None

    @jax.jit
    def run_chunk(carry):
      return jax.lax.scan(_train_step, carry, None, inner_steps)

    @jax.jit
    def jit_eval(runner_state):
      return compute_performance(
        config=config,
        agent_reset_fn=agent_reset_fn,
        actor_train_step_fn=actor.train_step,
        actor_eval_step_fn=actor.eval_step,
        env_reset_fn=vmap_reset,
        env_step_fn=vmap_step,
        train_env_params=train_env_params,
        test_env_params=test_env_params,
        runner_state=runner_state,
      )

    def log_eval_results(runner_state, eval_results):
      """Log eval results in regular Python (outside jit)."""
      if "eval_trajectory" in eval_results:
        logger.experience_logger(
          runner_state.train_state,
          eval_results["eval_trajectory"],
          "evaluator_performance",
        )
      if "actor_trajectory" in eval_results:
        logger.experience_logger(
          runner_state.train_state,
          eval_results["actor_trajectory"],
          "actor_performance",
        )

    carry = (runner_state, dummy_metrics, dummy_grads, dummy_traj_batch)

    for chunk in tqdm(range(num_chunks), desc="Training"):
      carry, _ = run_chunk(carry)
      runner_state, last_metrics, last_grads, last_traj = carry

      # --- Logging (every chunk = ~1% of training) ---
      # 1. Learner metrics
      logger.learner_logger(runner_state.train_state, last_metrics, key=loss_name)

      # 2. Gradient norms
      logger.gradient_logger(runner_state.train_state, last_grads)

      # 3. Eval performance (jit compute, then log in Python)
      eval_results = jit_eval(runner_state)
      log_eval_results(runner_state, eval_results)

      # 4. Online trajectory logging
      if online_trajectory_log_fn is not None:
        online_trajectory_log_fn(last_traj, runner_state.train_state.n_updates, config)

      # --- Save params every 10 chunks (~10% of training) ---
      if save_path is not None and chunk % 10 == 0:
        save_training_state(
          runner_state.train_state.params,
          config,
          save_path,
          config["ALG"],
          idx=chunk // 10,
          n_updates=int(runner_state.train_state.n_updates),
        )

    # Final eval
    eval_results = jit_eval(runner_state)
    log_eval_results(runner_state, eval_results)

    # Final save
    if save_path is not None:
      save_training_state(
        runner_state.train_state.params,
        config,
        save_path,
        config["ALG"],
      )

    return {"runner_state": runner_state}

  return train


##############################
# Epsilon Greedy (from qlearning.py)
##############################


def epsilon_greedy_act(q, eps, key):
  key_a, key_e = jax.random.split(key, 2)
  greedy_actions = jnp.argmax(q, axis=-1)
  random_actions = jax.random.randint(
    key_a, shape=greedy_actions.shape, minval=0, maxval=q.shape[-1]
  )
  pick_random = jax.random.uniform(key_e, greedy_actions.shape) < eps
  chosen_actions = jnp.where(pick_random, random_actions, greedy_actions)
  return chosen_actions


class LinearDecayEpsilonGreedy:
  """Epsilon Greedy action selection with linear decay."""

  def __init__(self, start_e: float, end_e: float, duration: int):
    self.start_e = start_e
    self.end_e = end_e
    self.duration = duration
    self.slope = (end_e - start_e) / duration

  @partial(jax.jit, static_argnums=0)
  def get_epsilon(self, t: int):
    e = self.slope * t + self.start_e
    return jnp.clip(e, self.end_e)

  @partial(jax.jit, static_argnums=0)
  def choose_actions(self, q_vals: jnp.ndarray, t: int, rng: chex.PRNGKey):
    eps = self.get_epsilon(t)
    rng = jax.random.split(rng, q_vals.shape[0])
    return jax.vmap(epsilon_greedy_act, in_axes=(0, None, 0))(q_vals, eps, rng)


class FixedEpsilonGreedy:
  """Epsilon Greedy action selection with fixed epsilon per environment."""

  def __init__(self, epsilons: float):
    self.epsilons = epsilons

  @partial(jax.jit, static_argnums=0)
  def choose_actions(self, q_vals: jnp.ndarray, t: int, rng: chex.PRNGKey):
    rng = jax.random.split(rng, q_vals.shape[0])
    return jax.vmap(epsilon_greedy_act, in_axes=(0, 0, 0))(q_vals, self.epsilons, rng)


def make_optimizer(config: dict) -> optax.GradientTransformation:
  def linear_schedule(count):
    frac = 1.0 - (count / config["NUM_UPDATES"])
    return config["LR"] * frac

  lr = linear_schedule if config.get("LR_LINEAR_DECAY", False) else config["LR"]

  return optax.chain(
    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
    optax.adam(learning_rate=lr, eps=config["EPS_ADAM"]),
  )


def make_actor(config: dict, agent: Agent, rng: jax.random.PRNGKey) -> Actor:
  fixed_epsilon = config.get("FIXED_EPSILON", 1)
  assert fixed_epsilon in (0, 1, 2)
  if fixed_epsilon:
    if fixed_epsilon == 1:
      vals = np.logspace(
        start=config.get("EPSILON_MIN", 1),
        stop=config.get("EPSILON_MAX", 3),
        num=config.get("NUM_EPSILONS", 256),
        base=config.get("EPSILON_BASE", 0.1),
      )
    else:
      vals = np.logspace(
        num=config.get("NUM_EPSILONS", 256),
        start=config.get("EPSILON_MIN", 0.05),
        stop=config.get("EPSILON_MAX", 0.9),
        base=config.get("EPSILON_BASE", 0.1),
      )
    num_envs = config["NUM_ENVS"]
    add_greedy = config.get("ADD_GREEDY_EPSILON", True)

    def train_choose_actions(q_vals, t, rng):
      rng, rng_eps = jax.random.split(rng)
      epsilons = jax.random.choice(rng_eps, vals, shape=(num_envs,))
      if add_greedy:
        epsilons = epsilons.at[-1].set(0.0)
      action_rngs = jax.random.split(rng, num_envs)
      return jax.vmap(epsilon_greedy_act)(q_vals, epsilons, action_rngs)
  else:
    explorer = LinearDecayEpsilonGreedy(
      start_e=config["EPSILON_START"],
      end_e=config["EPSILON_FINISH"],
      duration=config.get("EPSILON_ANNEAL_TIME") or config["TOTAL_TIMESTEPS"],
    )
    train_choose_actions = explorer.choose_actions

  eval_epsilon = jnp.full(config["NUM_ENVS"], config.get("EVAL_EPSILON", 0.1))
  eval_explorer = FixedEpsilonGreedy(eval_epsilon)

  def actor_step(
    train_state: TrainState,
    agent_state: jax.Array,
    timestep: TimeStep,
    rng: jax.random.PRNGKey,
  ):
    preds, agent_state = agent.apply(train_state.params, agent_state, timestep, rng)

    action = train_choose_actions(preds.q_vals, train_state.timesteps, rng)

    return preds, action, agent_state

  def eval_step(
    train_state: TrainState,
    agent_state: jax.Array,
    timestep: TimeStep,
    rng: jax.random.PRNGKey,
  ):
    preds, agent_state = agent.apply(train_state.params, agent_state, timestep, rng)

    action = eval_explorer.choose_actions(preds.q_vals, train_state.timesteps, rng)

    return preds, action, agent_state

  return Actor(train_step=actor_step, eval_step=eval_step)
