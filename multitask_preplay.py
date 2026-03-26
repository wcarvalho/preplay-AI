"""
multitask preplay.

TODO:
- remove DuellingDotMLP.
"""

from typing import Any, Tuple, Optional, Callable
import functools
from functools import partial

import distrax
import jax
import jax.numpy as jnp
import flax
from flax import struct
import optax
import flax.linen as nn
from gymnax.environments import environment
import numpy as np
import rlax
import jax.tree_util as jtu

import matplotlib.pyplot as plt
import wandb

import craftax_env
import craftax_web_env
from craftax_web_env import Achiement_to_idx

import losses
from jaxneurorl.agents.basics import TimeStep
from jaxneurorl.agents import value_based_basics as vbb
from jaxneurorl.agents import qlearning as base_agent
from networks import MLP, CraftaxObsEncoder, CraftaxMultiGoalObsEncoder
from networks import CategoricalJaxmazeObsEncoder

from visualizer import plot_frames
from jaxmaze import renderer


def info(x):
  return jax.tree_util.tree_map(lambda y: (y.shape, y.dtype), x)


def symlog(x):
  """Symmetric log transform from DreamerV3 (Hafner et al., 2023)."""
  return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def normalize_td_error(td_error, mask, style, scale_from=None):
  """Normalize TD-errors before squaring.

  Args:
    td_error: raw TD-errors [T-1] or [T-1, A]
    mask: loss mask [T-1] (broadcastable to td_error)
    style: "none", "symlog", "per_goal", or "symlog+per_goal"
    scale_from: optional tensor to compute the per_goal normalization scale
      from (instead of td_error itself). Use this when normalizing multiple
      TD-error streams (e.g. online + model) by a shared scale so you
      equalize across goals without distorting the ratio between streams.
  Returns:
    normalized TD-errors, same shape as td_error
  """
  if style == "none" or style == "":
    return td_error

  parts = style.split("+")
  out = td_error
  for part in parts:
    if part == "symlog":
      out = symlog(out)
    elif part == "per_goal":
      eps = 1e-6
      # normalize by mean absolute TD-error across time
      ref = out if scale_from is None else scale_from
      # apply symlog to ref too if it was already applied to out
      if "symlog" in parts and scale_from is not None:
        ref = symlog(ref)
      abs_mean = (jnp.abs(ref) * mask).sum(0) / mask.sum(0).clip(min=1)
      out = out / jnp.maximum(abs_mean, eps)
    else:
      raise ValueError(f"Unknown td_norm_style component: {part}")
  return out


Agent = nn.Module
Params = flax.core.FrozenDict
Qvalues = jax.Array
RngKey = jax.Array
make_actor = base_agent.make_actor

RnnState = jax.Array


##############################
# Dataclasses
##############################
@struct.dataclass
class AgentState:
  timestep: jax.Array
  rnn_state: jax.Array


@struct.dataclass
class Predictions:
  q_vals: jax.Array
  state: struct.PyTreeNode


@struct.dataclass
class SimulationOutput:
  actions: jax.Array
  predictions: Optional[Predictions] = None


@struct.dataclass
class LogEnvState:
  env_state: Any
  episode_returns: float
  episode_lengths: int
  returned_episode_returns: float
  returned_episode_lengths: int
  timestep: int


class GymnaxWrapper(object):
  """Base class for Gymnax wrappers."""

  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)


class LogWrapper(GymnaxWrapper):
  """Log the episode returns and lengths."""

  def __init__(self, env):
    super().__init__(env)

  def reset(self, key, params=None):
    obs, env_state = self._env.reset(key, params)
    state = LogEnvState(env_state, 0.0, 1, 0.0, 0, 0)
    return obs, state

  def step(self, key, state, action, params=None):
    obs, env_state, reward, done, info = self._env.step(
      key, state.env_state, action, params
    )
    new_episode_return = state.episode_returns + reward
    new_episode_length = state.episode_lengths + 1
    done_float = done.astype(jnp.float32)
    done_int = done.astype(jnp.int32)
    state = LogEnvState(
      env_state=env_state,
      episode_returns=new_episode_return * (1 - done_float),
      episode_lengths=new_episode_length * (1 - done_int),
      returned_episode_returns=(
        state.returned_episode_returns * (1 - done_float)
        + new_episode_return * done_float
      ),
      returned_episode_lengths=state.returned_episode_lengths * (1 - done_int)
      + new_episode_length * done_int,
      timestep=state.timestep + 1,
    )
    return obs, state, reward, done, info


##############################
# Utility functions
##############################
def make_float(x):
  return x.astype(jnp.float32)


def concat_pytrees(tree1, tree2, **kwargs):
  return jax.tree_util.tree_map(
    lambda x, y: jnp.concatenate((x, y), **kwargs), tree1, tree2
  )


def add_time(v):
  return jax.tree_util.tree_map(lambda x: x[None], v)


def concat_first_rest(first, rest):
  first = add_time(first)  # [N, ...] --> [1, N, ...]
  # rest: [T, N, ...]
  # output: [T+1, N, ...]
  return jax.vmap(concat_pytrees, 1, 1)(first, rest)


def is_truncated(timestep):
  non_terminal = timestep.discount
  is_last = make_float(timestep.last())
  truncated = (non_terminal + is_last) > 1
  return make_float(1 - truncated)


def simulation_finished_mask(initial_mask, next_timesteps):
  non_terminal = next_timesteps.discount[1:]
  is_last_t = make_float(next_timesteps.last()[1:])
  term_cumsum_t = jnp.cumsum(is_last_t, 0)
  loss_mask_t = make_float((term_cumsum_t + non_terminal) < 2)
  full_mask = jnp.concatenate((initial_mask[None], loss_mask_t), axis=0)
  # If initial state was masked (truncated/terminal), mask entire simulation
  return full_mask * initial_mask[None]


def make_optimizer(config: dict) -> optax.GradientTransformation:
  num_updates = int(config["NUM_UPDATES"] + config.get("NUM_EXTRA_REPLAY", 0))
  lr_scheduler = optax.linear_schedule(
    init_value=config["LR"], end_value=1e-10, transition_steps=num_updates
  )
  lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]
  return optax.chain(
    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
    optax.adam(learning_rate=lr, eps=config["EPS_ADAM"]),
  )


def repeat(x, N: int):
  def identity(y, n):
    return y

  return jax.vmap(identity, (None, 0), 0)(x, jnp.arange(N))


def get_in_episode(timestep):
  non_terminal = timestep.discount
  is_last = timestep.last()
  term_cumsum = jnp.cumsum(is_last, -1)
  in_episode = (term_cumsum + non_terminal) < 2
  return in_episode


def _episode_level_mask(q_not_declining, is_first, threshold):
  """Convert per-timestep non-declining mask to episode-level mask.

  For each episode, computes proportion of intra-episode transitions
  where Q is non-declining. If proportion >= threshold, mask=1 for the
  entire episode; otherwise mask=0.
  """
  T = is_first.shape[0]
  # Ensure the first timestep is treated as an episode start so it gets
  # a valid (non-negative) segment ID in the cumsum below.
  is_first = is_first.at[0].set(1.0)
  # Episode IDs via cumsum of is_first (marks start of each episode)
  episode_ids = (jnp.cumsum(is_first) - 1).astype(jnp.int32)  # [T], 0-indexed
  ep_ids_trans = episode_ids[:-1]  # [T-1]

  # Exclude cross-episode transitions: transition t->t+1 is cross-episode
  # if t+1 is the first timestep of a new episode
  not_cross = (1.0 - is_first[1:]).astype(jnp.float32)  # [T-1]
  valid_nondecl = q_not_declining * not_cross  # [T-1]

  # Per-episode aggregation
  nondecl_sum = jax.ops.segment_sum(valid_nondecl, ep_ids_trans, num_segments=T)
  valid_sum = jax.ops.segment_sum(not_cross, ep_ids_trans, num_segments=T)
  proportion = nondecl_sum / jnp.maximum(valid_sum, 1.0)  # [T]

  # Threshold → episode mask → broadcast back to transitions
  ep_mask = (proportion >= threshold).astype(jnp.float32)  # [T]
  return ep_mask[ep_ids_trans]  # [T-1]


def rnn_output_from_state(state):
  """Extract RNN output from LSTM state. state=(carry, hidden) -> hidden."""
  return state[1]


##############################
# Simulation
##############################
def simulate_n_trajectories(
  h_tm1: RnnState,
  x_t: TimeStep,
  online_goal: jax.Array,
  rng: jax.random.PRNGKey,
  network: nn.Module,
  params: Params,
  policy_fn: Callable = None,
  num_steps: int = 5,
  online_task_coeff: float = 1.0,
  simulation_task_coeff: float = 1.0,
  simulation_goal: jax.Array = None,
  place_goal_in_timestep=None,
):
  """Simulate n trajectories from a given state.

  Returns predictions and actions for every time-step including the current one.
  This first applies the model to the current time-step and then simulates T more
  time-steps. Output length is num_steps+1.

  Args:
    h_tm1: RNN state at previous timestep [D] (will be broadcast across sims)
    x_t: current timestep (no batch dim)
    main_task_goal: main-task goal vectors [num_simulations, G]
    rng: random key
    network: agent module
    params: network parameters
    policy_fn: (Predictions, rng) -> actions [num_simulations]
    num_steps: number of simulation steps
    num_simulations: number of parallel simulations
    mainq_coeff: coefficient for main-task Q values
    offtask_coeff: coefficient for off-task Q values
    offtask_goal: off-task goal vectors [num_simulations, G], or None for dyna-only
    place_goal_in_timestep: optional fn to modify timestep for goal
  """

  if simulation_goal is None:
    simulation_goal = online_goal
  assert online_goal.shape[0] == simulation_goal.shape[0]
  num_simulations = online_goal.shape[0]

  def get_q_vals(lstm_out, main_w, off_w):
    main_q = network.apply(params, lstm_out, main_w, method=network.main_task_q_fn)
    offtask_q = network.apply(params, lstm_out, off_w, method=network.off_task_q_fn)
    return online_task_coeff * main_q + simulation_task_coeff * offtask_q

  def initial_predictions(x, prior_h, main_w, sim_w, rng_):
    if place_goal_in_timestep is not None:
      x = place_goal_in_timestep(x, sim_w)
    lstm_state, lstm_out = network.apply(
      params, prior_h, x, rng_, task_dropout=False, method=network.apply_rnn
    )
    preds = Predictions(q_vals=get_q_vals(lstm_out, main_w, sim_w), state=lstm_state)
    return x, lstm_state, preds

  rng, rng_ = jax.random.split(rng)

  # get an x_t for each simulation
  # also, get h_t, preds_t for each simulation
  # [N, ...]
  x_t, h_t, preds_t = jax.vmap(
    initial_predictions, in_axes=(None, None, 0, 0, 0), out_axes=0
  )(
    x_t,
    h_tm1,
    online_goal,
    simulation_goal,
    jax.random.split(rng_, num_simulations),
  )
  # use preds_t to selection a_t (for each simulation)
  a_t = policy_fn(preds_t, rng_)

  def _single_model_step(carry, inputs):
    del inputs
    (timestep, lstm_state, a, rng) = carry

    rng, rng_ = jax.random.split(rng)
    next_timestep = network.apply(params, timestep, a, rng_, method=network.apply_model)

    next_lstm_state, next_rnn_out = network.apply(
      params,
      lstm_state,
      next_timestep,
      rng_,
      task_dropout=False,
      method=network.apply_rnn,
    )

    next_preds = Predictions(
      q_vals=jax.vmap(get_q_vals)(next_rnn_out, online_goal, simulation_goal),
      state=next_lstm_state,
    )
    next_a = policy_fn(next_preds, rng_)
    carry = (next_timestep, next_lstm_state, next_a, rng)
    sim_output = SimulationOutput(
      predictions=next_preds,
      actions=next_a,
    )
    return carry, (next_timestep, sim_output)

  initial_carry = (x_t, h_t, a_t, rng)
  _, (next_timesteps, sim_outputs) = jax.lax.scan(
    f=_single_model_step, init=initial_carry, xs=None, length=num_steps
  )

  # return [preds_t, ..., preds_{t+n}]
  sim_outputs = SimulationOutput(
    predictions=concat_first_rest(preds_t, sim_outputs.predictions),
    actions=concat_first_rest(a_t, sim_outputs.actions),
  )
  all_timesteps = concat_first_rest(x_t, next_timesteps)
  return all_timesteps, sim_outputs


##############################
# Q-Heads
##############################
class DuellingMLP(nn.Module):
  hidden_dim: int
  out_dim: int = 0
  num_layers: int = 1
  norm_type: str = "none"
  activation: str = "relu"
  activate_final: bool = True
  use_bias: bool = False

  @nn.compact
  def __call__(self, x, task, train: bool = False):
    x = jnp.concatenate((x, task), axis=-1)
    value_mlp = MLP(
      hidden_dim=self.hidden_dim,
      num_layers=self.num_layers,
      norm_type=self.norm_type,
      activation=self.activation,
      activate_final=False,
      use_bias=self.use_bias,
      out_dim=1,
    )
    advantage_mlp = MLP(
      hidden_dim=self.hidden_dim,
      num_layers=self.num_layers,
      norm_type=self.norm_type,
      activation=self.activation,
      activate_final=False,
      use_bias=self.use_bias,
      out_dim=self.out_dim,
    )
    assert self.out_dim > 0, "must have at least one action"

    value = value_mlp(x, train)
    advantages = advantage_mlp(x, train)
    advantages -= jnp.mean(advantages, axis=-1, keepdims=True)
    q_values = value + advantages
    return q_values


class DuellingDotMLP(nn.Module):
  hidden_dim: int
  out_dim: int = 0
  num_layers: int = 1
  norm_type: str = "none"
  activation: str = "relu"
  activate_final: bool = True
  use_bias: bool = False

  @nn.compact
  def __call__(self, x, task, train: bool = False):
    task_dim = task.shape[-1]
    x = jnp.concatenate((x, task), axis=-1)
    value_mlp = MLP(
      hidden_dim=self.hidden_dim,
      num_layers=self.num_layers,
      use_bias=self.use_bias,
      out_dim=task_dim,
    )
    advantage_mlp = MLP(
      hidden_dim=self.hidden_dim,
      num_layers=self.num_layers,
      use_bias=self.use_bias,
      out_dim=self.out_dim * task_dim,
    )
    assert self.out_dim > 0, "must have at least one action"

    value = value_mlp(x)  # [C]
    advantages = advantage_mlp(x)  # [A*C]
    advantages = advantages.reshape(self.out_dim, task_dim)  # [A, C]

    # Advantages have zero mean across actions.
    advantages -= jnp.mean(advantages, axis=0, keepdims=True)  # [A, C]

    sf = value[None, :] + advantages  # [A, C]
    q_values = (sf * task[None, :]).sum(-1)  # [A]

    return q_values


##############################
# Loss Function
##############################
@struct.dataclass
class PreplayLossFn:
  """Loss function for multitask preplay.

  Standalone implementation that does not inherit from RecurrentLossFn.
  """

  network: nn.Module
  discount: float = 0.99
  lambda_: float = 0.9
  step_cost: float = 0.001
  max_priority_weight: float = 0.9
  importance_sampling_exponent: float = 0.6
  burn_in_length: int = None

  data_wrapper: flax.struct.PyTreeNode = vbb.AcmeBatchData
  logger: vbb.loggers.Logger = vbb.loggers.Logger

  # Preplay-specific
  num_ontask_simulations: int = 1
  num_offtask_simulations: int = 1
  simulation_length: int = 5
  online_coeff: float = 1.0
  dyna_coeff: float = 1.0
  all_goals_coeff: float = 1.0
  all_goals_lambda: float = 0.3
  num_offtask_goals: int = 5
  offtask_loss_coeff: float = 1.0
  ontask_loss_coeff: float = 1.0
  simulation_task_coeff: float = 1.0
  online_task_coeff: float = 1.0

  augment_dyna_reward: Callable = None
  dyna_epsilon_values: Callable = None
  offtask_dyna_epsilon_values: Callable = None
  sample_preplay_goals: Callable = None
  sample_td_goals: Callable = None
  compute_rewards: Callable = None
  get_main_goal: Callable = None
  place_goal_in_timestep: Callable = None
  make_log_extras: Callable = None
  all_goals_rnn: bool = True
  all_goals_td: str = "retrace"
  retrace_temperature: float = 0.1
  peng_trace_cutting: bool = True
  sim_peng_trace_cutting: bool = False
  dyna_other_only: bool = True  # True=replacement (current), False=additive
  mask_declining_model: str = ""  # "greedy_online", "greedy_target", "target_trend", + "_episode" variants, or "" (off)
  mask_declining_threshold: float = 0.5  # proportion threshold for _episode modes
  mask_declining_keep_last: bool = False  # OR-gate td_loss_mask with is_last
  keep_greedy_aligned: bool = False  # OR with greedy(on)==greedy(off)
  all_goals_dyna_coeff: float = None  # None → use all_goals_coeff
  td_norm_style: str = "none"  # "none", "symlog", "per_goal", "symlog+per_goal"

  def __call__(
    self,
    params: Params,
    target_params: Params,
    batch,
    key_grad: jax.random.PRNGKey,
    steps: int,
  ):
    """Calculate a loss on a single batch of data."""
    unroll = functools.partial(self.network.apply, method=self.network.unroll)

    online_state = batch.experience.extras.get("agent_state")
    online_state = jax.tree_util.tree_map(lambda x: x[:, 0], online_state)
    target_state = online_state

    data = vbb.batch_to_sequence(batch.experience)

    # Maybe burn the core state in.
    if self.burn_in_length:
      burn_data = jax.tree_util.tree_map(lambda x: x[: self.burn_in_length], data)
      key_grad, rng_1, rng_2 = jax.random.split(key_grad, 3)
      _, online_state = unroll(params, online_state, burn_data.timestep, rng_1)
      _, target_state = unroll(target_params, target_state, burn_data.timestep, rng_2)
      data = jax.tree_util.tree_map(lambda seq: seq[self.burn_in_length :], data)

    # Single RNN unroll
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

    # Priority computation + importance weighting
    abs_td_error = jnp.abs(elemwise_error).astype(jnp.float32)
    max_priority = self.max_priority_weight * jnp.max(abs_td_error, axis=0)
    mean_priority = (1 - self.max_priority_weight) * jnp.mean(abs_td_error, axis=0)
    priorities = max_priority + mean_priority

    importance_weights = (1.0 / (batch.priorities + 1e-6)).astype(jnp.float32)
    importance_weights **= self.importance_sampling_exponent
    importance_weights /= jnp.max(importance_weights)
    batch_loss = jnp.mean(importance_weights * batch_loss)

    updates = dict(priorities=priorities)
    return batch_loss, (updates, metrics)

  def loss_fn(
    self,
    timestep,
    online_preds,
    target_preds,
    actions,
    rewards,
    is_last,
    non_terminal,
    loss_mask,
    lambda_override=None,
  ):
    rewards = make_float(rewards)
    rewards = rewards - self.step_cost
    is_last = make_float(is_last)
    discounts = make_float(non_terminal) * self.discount
    effective_lambda = self.lambda_ if lambda_override is None else lambda_override
    lambda_ = jnp.ones_like(non_terminal) * effective_lambda

    has_batch_dim = rewards.ndim == 2
    td_error_fn = losses.q_learning_lambda_td
    if has_batch_dim:
      td_error_fn = jax.vmap(td_error_fn, in_axes=1, out_axes=1)
    selector_actions = jnp.argmax(online_preds.q_vals, axis=-1)

    q_t, target_q_t = td_error_fn(
      online_preds.q_vals[:-1],
      actions[:-1],
      target_preds.q_vals[1:],
      selector_actions[1:],
      rewards[1:],
      discounts[1:],
      is_last[1:],
      lambda_[1:],
    )

    target_q_t = target_q_t * non_terminal[:-1]
    batch_td_error = target_q_t - q_t
    batch_td_error = batch_td_error * loss_mask[:-1]

    batch_loss = 0.5 * jnp.square(batch_td_error)
    batch_loss_mean = (batch_loss * loss_mask[:-1]).sum(0) / (
      loss_mask[:-1].sum(0) + 1e-5
    )

    metrics = {
      "0.q_loss": batch_loss.mean(),
      "0.q_td": jnp.abs(batch_td_error).mean(),
      "1.reward": rewards[1:].mean(),
    }

    log_info = {
      "timesteps": timestep,
      "actions": actions,
      "td_errors": batch_td_error,
      "non_terminal": non_terminal,
      "loss_mask": loss_mask,
      "q_values": online_preds.q_vals,
      "q_loss": batch_loss,
      "q_target": target_q_t,
      "target_q_values": target_preds.q_vals,
      "loss_rewards": rewards,
      "lambda_": lambda_
    }
    return batch_td_error, batch_loss_mean, metrics, log_info

  def error(
    self,
    data,
    online_preds: Predictions,
    online_state,
    target_preds: Predictions,
    target_state,
    params: Params,
    target_params: Params,
    steps: int,
    key_grad: jax.random.PRNGKey,
    **kwargs,
  ):
    loss_mask = is_truncated(data.timestep)

    all_metrics = {}
    all_log_info = {"n_updates": steps}
    T, B = loss_mask.shape[:2]

    # ---- Online loss ----
    if self.online_coeff > 0.0:
      # [T, B, G]
      goal = self.get_main_goal(data.timestep, expand_across_simulations=False)
      rewards = self.compute_rewards(data.timestep, goal, expand_g_across_time=False)

      td_error, batch_loss, metrics, log_info = self.loss_fn(
        timestep=data.timestep,  # [T, B, ...]
        online_preds=online_preds,  # .q_vals: [T, B, A]
        target_preds=target_preds,  # .q_vals: [T, B, A]
        actions=data.action,  # [T, B]
        rewards=rewards,  # [T, B]
        is_last=make_float(data.timestep.last()),  # [T, B]
        non_terminal=data.timestep.discount,  # [T, B]
        loss_mask=loss_mask,  # [T, B]
      )  # returns [T-1, B], [B], dict, dict
      all_metrics.update({f"0.online/{k}": v for k, v in metrics.items()})
      # Swap [T, B, ...] -> [B, T, ...] for consistent first-batch indexing
      swap = lambda x: jnp.swapaxes(x, 0, 1)
      all_log_info["online"] = jax.tree_util.tree_map(swap, log_info)
      if self.make_log_extras is not None:
        extras = self.make_log_extras(data.timestep)  # [T, B, ...]
        all_log_info["online"].update(jax.tree_util.tree_map(swap, extras))
      td_error = jnp.concatenate((td_error, jnp.zeros(B)[None]), 0)
      td_error = jnp.abs(td_error)
    else:
      td_error = jnp.zeros_like(loss_mask)
      batch_loss = td_error.sum(0)

    # ---- All-goals loss (reusing RNN outputs) ----
    if self.all_goals_coeff > 0.0:
      # [B, N, G]
      key_grad, key_grad_ = jax.random.split(key_grad)
      key_grad_ = jax.random.split(key_grad_, B + 1)
      all_goals, _, _ = jax.vmap(self.sample_td_goals, (1, 0), 0)(
        data.timestep, key_grad_[1:]
      )

      N = all_goals.shape[1]
      key_grad, key_grad_ = jax.random.split(key_grad)
      key_grad_ = jax.random.split(key_grad_, B * N).reshape(B, N, 2)

      # vmap over batch
      all_goals_loss = jax.vmap(
        self.all_goals_loss_fn, (0, 1, 1, 1, 1, 0, 0, 0, None, None), 0
      )

      # vmap over goals
      all_goals_loss = jax.vmap(
        all_goals_loss, (1, None, None, None, None, None, None, 1, None, None), 1
      )

      # _, [B, N], _, [B, T, N]
      _, all_goals_batch_loss, all_goals_metrics, all_goals_log_info = all_goals_loss(
        all_goals,  # [B, N, G]
        data.timestep,  # [T, B, ...]
        data.action,  # [T, B]
        online_preds,  # [T, B, ...]
        target_preds,  # [T, B, ...]
        online_state,  # [B, D]
        target_state,  # [B, D]
        key_grad_,  # [B]
        params,
        target_params,
      )
      # all_goals_log_info N dim follows all_tasks ordering:
      # all_tasks = concat(train_objects, test_objects) = concat(groups[:,0], groups[:,1])
      # Default 2 groups: idx 0=microwave, 1=fork, 2=stove, 3=knife
      # Take goal index 2 (stove): [B, T, N, ...] -> [B, T, ...]
      all_log_info["all_goals"] = jax.tree_util.tree_map(
        lambda x: x[:, 2], all_goals_log_info
      )
      # Take goal index 0 (microwave): [B, T, N, ...] -> [B, T, ...]
      all_log_info["all_goals_microwave"] = jax.tree_util.tree_map(
        lambda x: x[:, 0], all_goals_log_info
      )

      batch_loss += all_goals_batch_loss.mean(1)  # coeff applied inside
      all_metrics.update({f"2.all_goals/{k}": v for k, v in all_goals_metrics.items()})

    # ---- Dyna/preplay loss ----
    if self.dyna_coeff > 0.0:
      remove_last = lambda x: jax.tree_util.tree_map(lambda y: y[:-1], x)
      h_tm1_online = concat_first_rest(online_state, remove_last(online_preds.state))
      h_tm1_target = concat_first_rest(target_state, remove_last(target_preds.state))
      x_t = data.timestep

      # For preplay/dyna, also exclude terminal steps: simulating from a terminal
      # state triggers auto-reset, producing a random episode with meaningless loss.
      preplay_loss_mask = loss_mask * (1 - make_float(data.timestep.last()))

      dyna_loss_fn = functools.partial(
        self.preplay_loss_fn, params=params, target_params=target_params
      )
      dyna_loss_fn = jax.vmap(dyna_loss_fn, (1, 1, 1, 1, 1, 0), 0)
      dyna_td_error, dyna_batch_loss, dyna_metrics, dyna_log_info = dyna_loss_fn(
        x_t,
        data.action,
        h_tm1_online,
        h_tm1_target,
        preplay_loss_mask,
        jax.random.split(key_grad, B),
      )
      batch_loss += self.dyna_coeff * dyna_batch_loss
      td_error += self.dyna_coeff * dyna_td_error
      all_metrics.update({f"1.preplay/{k}": v for k, v in dyna_metrics.items()})
      all_log_info["dyna"] = dyna_log_info

    if self.logger.learner_log_extra is not None:
      self.logger.learner_log_extra(all_log_info)

    return td_error, batch_loss, all_metrics

  def all_goals_loss_fn(
    self,
    new_goal,
    timestep,
    actions,
    online_preds,
    target_preds,
    online_state,
    target_state,
    key_grad,
    params,
    target_params,
  ):
    """All-goals Q-learning loss.

    Args:
      new_goal: goal with scalar fields [G]
      timestep: [T, ...] timestep sequence
      actions: [T] actions
      online_preds: Predictions with state [T, ...]
      target_preds: Predictions with state [T, ...]
    """
    dyna_coeff = (
      self.all_goals_dyna_coeff
      if self.all_goals_dyna_coeff is not None
      else self.all_goals_coeff
    )

    unroll = functools.partial(self.network.apply, method=self.network.unroll)

    if self.place_goal_in_timestep:
      place_goal_in_timestep = jax.vmap(self.place_goal_in_timestep, (0, None), 0)
      # [T, D] <- [T, D], [G]
      timestep_w_g = place_goal_in_timestep(timestep, new_goal)

    key_grad, rng_1, rng_2 = jax.random.split(key_grad, 3)
    online_preds_g, _ = unroll(
      params, online_state, timestep_w_g, rng_1, task_dropout=False
    )
    target_preds_g, _ = unroll(
      target_params, target_state, timestep_w_g, rng_2, task_dropout=False
    )

    goal_rewards = self.compute_rewards(timestep_w_g, new_goal)

    # Rewards and discounts
    # NOTE: We use the original episode's discount (timestep.discount) for terminal
    # handling, making on-task terminals absorbing states for off-task goals too.
    # This is correct for JaxMaze where episodes end on ANY object pickup
    # (terminate_with_any=True), so on-task terminal ≡ off-task terminal.
    # Would need off-task-specific discounts if episodes could continue after on-task pickup.
    rewards = make_float(goal_rewards) - self.step_cost
    discounts = timestep.discount * self.discount
    loss_mask = is_truncated(timestep)
    is_last = make_float(timestep.last())

    # Model-based 1-step target: r + γ Q(s', argmax_a' Q_online(s', a'), g)
    # Uses Double DQN: select action with online params, evaluate with target params.
    # All inputs are unbatched (scalar action, single timestep).
    def model_based_target_q(online_h, target_h, timestep_t, action, key_grad):
      key_grad, rng_1, rng_2, rng_3 = jax.random.split(key_grad, 4)
      # Model forward (unsqueeze for apply_model's internal vmap, then squeeze)
      next_timestep = jax.tree_util.tree_map(
        lambda x: x[0],
        self.network.apply(
          params,
          jax.tree_util.tree_map(lambda x: x[None], timestep_t),
          action[None],
          rng_1,
          method=self.network.apply_model,
        ),
      )
      # Reward: timestep_w_g was passed in, so next_timestep has goal context
      reward = self.compute_rewards(next_timestep, new_goal, expand_g_across_time=False)
      reward = reward - self.step_cost
      # RNN forward (online for action selection, target for evaluation)
      _, next_rnn_out_online = self.network.apply(
        params,
        online_h,
        next_timestep,
        rng_2,
        task_dropout=False,
        method=self.network.apply_rnn,
      )
      _, next_rnn_out_target = self.network.apply(
        target_params,
        target_h,
        next_timestep,
        rng_3,
        task_dropout=False,
        method=self.network.apply_rnn,
      )
      # Double DQN: select with online, evaluate with target
      q_online = self.network.apply(
        params,
        next_rnn_out_online,
        new_goal,
        method=self.network.main_task_q_fn,
      )
      selector_action = jnp.argmax(q_online, -1)
      q_target = self.network.apply(
        target_params,
        next_rnn_out_target,
        new_goal,
        method=self.network.main_task_q_fn,
      )
      max_q = rlax.batched_index(q_target, selector_action)
      return reward + next_timestep.discount * self.discount * max_q

    if self.all_goals_td in ("retrace", "tree"):
      # Retrace off-policy correction (Munos et al., 2016)
      # The trajectory was collected under the main-task policy (behavior),
      # but we want to learn Q-values for the off-task goal (target).
      # Retrace uses clipped IS ratios to safely correct for this mismatch.
      temp = self.retrace_temperature

      if self.all_goals_td == "tree":
        mu_t = jnp.ones(goal_rewards.shape[0])
      else:
        # Behavior policy: Boltzmann over on-task Q-values [T, A]
        mu_probs = jax.nn.softmax(online_preds.q_vals / temp, axis=-1)
        # Prob of the taken action under behavior policy: [T]
        mu_t = rlax.batched_index(mu_probs, actions)

      # Target policy: Boltzmann over off-task Q-values [T, A]
      pi_t = jax.nn.softmax(online_preds_g.q_vals / temp, axis=-1)

      # vmap retrace over N (goal dimension = axis 1)
      qa_tm1, target_tm1 = losses.retrace(
        online_preds_g.q_vals[:-1],  # q_tm1:      [T-1, A]
        target_preds_g.q_vals[1:],  # q_t:        [T-1, A]
        actions[:-1],  # a_tm1:      [T-1]
        actions[1:],  # a_t:        [T-1]
        rewards[1:],  # r_t:        [T-1]
        discounts[1:],  # discount_t: [T-1]
        pi_t[1:],  # pi_t:       [T-1, A]
        mu_t[1:],  # mu_t:       [T-1]
        self.all_goals_lambda,  # lambda:     scalar
        is_last[1:],  # is_last_t:  [T-1]
      )
      # Force targets to 0 at episode termination (Q(terminal, a) = 0).
      # Use original episode termination (timestep), not imaginary goal's (timestep_w_g).
      target_tm1 = target_tm1 * timestep.discount[:-1]  # [T-1]
      ag_td_error = target_tm1 - qa_tm1  # [T-1]

    elif self.all_goals_td == "sarsa":
      # sarsa_lambda is a sequence fn ([T, A] q-values), vmap over N only
      qa_tm1, target_tm1 = losses.sarsa_lambda(
        online_preds_g.q_vals[:-1],  # q_tm1: [T-1, A]
        actions[:-1],  # a_tm1: [T-1]
        rewards[1:],  # r_t:   [T-1]
        discounts[1:],  # discount_t: [T-1]
        target_preds_g.q_vals[1:],  # q_t:   [T-1, A]
        actions[1:],  # a_t:   [T-1]
        self.all_goals_lambda,
        is_last[1:],  # is_last_t:  [T-1]
      )
      # Force targets to 0 at episode termination (Q(terminal, a) = 0).
      # Use original episode termination (timestep), not imaginary goal's (timestep_w_g).
      target_tm1 = target_tm1 * timestep.discount[:-1]  # [T-1]
      ag_td_error = target_tm1 - qa_tm1  # [T-1]

    elif self.all_goals_td == "qlearning":
      ag_td_error, ag_batch_loss, ag_metrics, ag_log_info = self.loss_fn(
        timestep=timestep_w_g,  # [T, ...]
        online_preds=online_preds_g,  # .q_vals: [T, A]
        target_preds=target_preds_g,  # .q_vals: [T, A]
        actions=actions,  # [T]
        rewards=goal_rewards,  # [T]
        is_last=make_float(timestep.last()),  # [T]
        non_terminal=timestep.discount,  # [T]
        loss_mask=is_truncated(timestep),  # [T]
        lambda_override=self.all_goals_lambda,
      )  # returns [T-1], [N], dict, dict
      ag_batch_loss = self.all_goals_coeff * ag_batch_loss
      if self.make_log_extras is not None:
        extras = self.make_log_extras(timestep_w_g, goal=new_goal)
        ag_log_info.update(extras)

      return ag_td_error, ag_batch_loss, ag_metrics, ag_log_info

    elif self.all_goals_td == "rmae":
      # RMAE: Retrace for taken action (multi-step) +
      #       model-based 1-step for non-taken actions (action breadth)
      temp = self.retrace_temperature
      T = actions.shape[0]
      num_actions = online_preds_g.q_vals.shape[-1]  # A

      # --- Part A: Retrace target for the taken action [T-1] ---
      # Behavior policy: Boltzmann over on-task Q-values
      mu_probs = jax.nn.softmax(online_preds.q_vals / temp, axis=-1)  # [T, A]
      mu_t = rlax.batched_index(mu_probs, actions)  # [T]
      # Target policy: Boltzmann over off-task Q-values
      pi_t = jax.nn.softmax(online_preds_g.q_vals / temp, axis=-1)  # [T, A]

      _, target_tm1_retrace = losses.retrace(
        online_preds_g.q_vals[:-1],  # q_tm1:      [T-1, A]
        target_preds_g.q_vals[1:],  # q_t:        [T-1, A]
        actions[:-1],  # a_tm1:      [T-1]
        actions[1:],  # a_t:        [T-1]
        rewards[1:],  # r_t:        [T-1]
        discounts[1:],  # discount_t: [T-1]
        pi_t[1:],  # pi_t:       [T-1, A]
        mu_t[1:],  # mu_t:       [T-1]
        self.all_goals_lambda,  # lambda:     scalar
        is_last[1:],  # is_last_t:  [T-1]
      )
      target_tm1_retrace = target_tm1_retrace * timestep.discount[:-1]  # [T-1]

      # --- Part B: Model-based 1-step targets for non-taken actions [T, A] ---
      h_t_online = online_preds_g.state  # [T, ...] — online RNN state after timestep[t]
      h_t_target = target_preds_g.state  # [T, ...] — target RNN state after timestep[t]

      all_actions_tiled = jnp.tile(jnp.arange(num_actions), (T, 1))  # [T, A]
      keys_tiled = jax.random.split(key_grad, T * num_actions).reshape(
        T, num_actions, 2
      )  # [T, A, 2]

      # Model-based targets for all actions: [T, A]
      target_all = jax.vmap(
        jax.vmap(model_based_target_q, in_axes=(None, None, None, 0, 0)),
        in_axes=(0, 0, 0, 0, 0),
      )(h_t_online, h_t_target, timestep_w_g, all_actions_tiled, keys_tiled)  # [T, A]

      target_all_tm1 = target_all[:-1]  # [T-1, A]
      target_all_tm1 = target_all_tm1 * timestep.discount[:-1, None]  # [T-1, A]

      # --- Separate losses: retrace (taken action) + model (all or non-taken) ---
      taken_onehot = jax.nn.one_hot(actions[:-1], num_actions)  # [T-1, A]
      qa_tm1_all = online_preds_g.q_vals[:-1]  # [T-1, A]

      # Retrace loss for taken action [T-1]
      # Divide by num_actions so relative weighting matches original mean(axis=-1)
      qa_tm1_taken = rlax.batched_index(qa_tm1_all, actions[:-1])  # [T-1]
      retrace_td = target_tm1_retrace - qa_tm1_taken  # [T-1]
      # Normalize retrace and model TD-errors by a shared scale
      retrace_td_normed = normalize_td_error(
        retrace_td, loss_mask[:-1], self.td_norm_style, scale_from=retrace_td
      )
      retrace_loss_per_t = (
        0.5 * jnp.square(retrace_td_normed) * loss_mask[:-1] / num_actions
      )  # [T-1]

      # Model loss for actions [T-1, A]
      model_td_all = target_all_tm1 - qa_tm1_all  # [T-1, A]
      model_td_all_normed = normalize_td_error(
        model_td_all, loss_mask[:-1, None], self.td_norm_style, scale_from=retrace_td
      )
      if self.dyna_other_only:
        model_mask = 1 - taken_onehot  # non-taken only [T-1, A]
      else:
        model_mask = jnp.ones_like(taken_onehot)  # ALL actions [T-1, A]
      model_loss_per_action = (
        0.5 * jnp.square(model_td_all_normed) * model_mask
      )  # [T-1, A]
      model_loss_per_t = model_loss_per_action.mean(axis=-1) * loss_mask[:-1]  # [T-1]

      # Combined with separate coefficients
      ag_loss_per_t = (
        self.all_goals_coeff * retrace_loss_per_t + dyna_coeff * model_loss_per_t
      )  # [T-1]
      ag_td_error = retrace_td * loss_mask[:-1]  # [T-1] for logging
      ag_batch_loss = ag_loss_per_t.sum(0) / loss_mask[:-1].sum(0)

      # For logging: use taken action's target (retrace)
      target_tm1 = target_tm1_retrace

      ag_metrics = {
        "0.q_loss": ag_loss_per_t.mean(),
        "0.q_loss_online": retrace_loss_per_t.mean(),
        "0.q_loss_model": model_loss_per_t.mean(),
        "0.q_td": jnp.abs(ag_td_error).mean(),
        "1.reward": rewards.mean(),
      }
      ag_log_info = {
        "timesteps": timestep_w_g,
        "actions": actions,
        "td_errors": ag_td_error,
        "non_terminal": timestep_w_g.discount,
        "loss_mask": loss_mask,
        "q_values": online_preds_g.q_vals,
        "q_loss": ag_loss_per_t,
        "q_target": target_tm1,
        "target_q_values": target_preds_g.q_vals,  # [T, A]
        "behavior_q_values": online_preds.q_vals,
        "loss_rewards": rewards,
      }
      if self.make_log_extras is not None:
        extras = self.make_log_extras(timestep_w_g, goal=new_goal)
        ag_log_info.update(extras)
      return ag_td_error, ag_batch_loss, ag_metrics, ag_log_info

    elif self.all_goals_td == "mb_retrace":
      # Retrace (multi-step) where on-policy, 1-step model where off-policy.
      selector_actions = jnp.argmax(online_preds_g.q_vals, axis=-1)  # [T]
      selector_a_is_online_a = selector_actions == actions  # [T]

      # Retrace targets with IS correction
      temp = self.retrace_temperature
      mu_probs = jax.nn.softmax(online_preds.q_vals / temp, axis=-1)  # [T, A]
      mu_t = rlax.batched_index(mu_probs, actions)  # [T]
      pi_t = jax.nn.softmax(online_preds_g.q_vals / temp, axis=-1)  # [T, A]

      qa_tm1_retrace, target_q_t_online = losses.retrace(
        online_preds_g.q_vals[:-1],  # q_tm1:      [T-1, A]
        target_preds_g.q_vals[1:],  # q_t:        [T-1, A]
        actions[:-1],  # a_tm1:      [T-1]
        actions[1:],  # a_t:        [T-1]
        rewards[1:],  # r_t:        [T-1]
        discounts[1:],  # discount_t: [T-1]
        pi_t[1:],  # pi_t:       [T-1, A]
        mu_t[1:],  # mu_t:       [T-1]
        self.all_goals_lambda,  # lambda:     scalar
        is_last[1:],  # is_last_t:  [T-1]
      )
      target_q_t_online = target_q_t_online * timestep.discount[:-1]  # [T-1]

      # 1-step model-based targets
      T = actions.shape[0]
      h_t_online = online_preds_g.state  # [T, ...]
      h_t_target = target_preds_g.state  # [T, ...]
      target_q_t_model = jax.vmap(model_based_target_q)(
        h_t_online,
        h_t_target,
        timestep_w_g,
        selector_actions,
        jax.random.split(key_grad, T),
      )  # [T]
      target_q_t_model = target_q_t_model[:-1] * timestep.discount[:-1]  # [T-1]

      # Retrace everywhere for Q(s, a_taken);
      # model for Q(s, a*) only where a* != a_taken
      qa_tm1 = rlax.batched_index(
        online_preds_g.q_vals[:-1], selector_actions[:-1]
      )  # [T-1]
      online_td = target_q_t_online - qa_tm1_retrace  # [T-1]
      model_td = target_q_t_model - qa_tm1  # [T-1]

      model_mask = 1.0 - selector_a_is_online_a[:-1].astype(jnp.float32)  # [T-1]

      online_loss = 0.5 * jnp.square(online_td) * loss_mask[:-1]
      model_loss = 0.5 * jnp.square(model_td) * model_mask * loss_mask[:-1]

      ag_loss_per_t = (
        self.all_goals_coeff * online_loss + dyna_coeff * model_loss
      )  # [T-1]
      ag_batch_loss = ag_loss_per_t.sum(0) / loss_mask[:-1].sum(0).clip(min=1)
      ag_td_error = online_td * loss_mask[:-1]  # [T-1]

      target_tm1 = jnp.where(
        selector_a_is_online_a[:-1], target_q_t_online, target_q_t_model
      )  # [T-1]

      ag_metrics = {
        "0.q_loss": ag_loss_per_t.mean(),
        "0.q_loss_online": online_loss.mean(),
        "0.q_loss_model": model_loss.mean(),
        "0.q_td": jnp.abs(ag_td_error).mean(),
        "1.reward": rewards.mean(),
      }
      ag_log_info = {
        "timesteps": timestep_w_g,
        "actions": actions,
        "td_errors": ag_td_error,
        "non_terminal": timestep_w_g.discount,
        "loss_mask": loss_mask,
        "q_values": online_preds_g.q_vals,
        "q_loss": ag_loss_per_t,
        "q_target": target_tm1,
        "target_q_values": target_preds_g.q_vals,  # [T, A]
        "behavior_q_values": online_preds.q_vals,
        "loss_rewards": rewards,
        "target_q_t_lambda": target_q_t_online,
        "target_q_t_model": target_q_t_model,
      }
      if self.make_log_extras is not None:
        extras = self.make_log_extras(timestep_w_g, goal=new_goal)
        ag_log_info.update(extras)
      return ag_td_error, ag_batch_loss, ag_metrics, ag_log_info

    elif "mb_peng_lambda" in self.all_goals_td:
      selector_actions = jnp.argmax(online_preds_g.q_vals, axis=-1)
      selector_a_is_online_a = selector_actions == actions

      # only continue backing up if greedy action is online action
      if self.peng_trace_cutting:
        lambda_ = selector_a_is_online_a * self.all_goals_lambda
      else:
        lambda_ = jnp.ones_like(actions) * self.all_goals_lambda

      # Peng's λ target using goal-conditioned Q-values
      target_q_t_online = losses.q_learning_lambda_target(
        q_t=target_preds_g.q_vals[1:],  # goal-conditioned, not on-task
        r_t=rewards[1:],
        discount_t=discounts[1:],
        is_last_t=is_last[1:],
        a_t=selector_actions[1:],
        lambda_=lambda_[1:],  # NOTE: based on whether online action matches greedy
      )
      target_q_t_online = target_q_t_online * timestep.discount[:-1]
      h_t_online = online_preds_g.state  # [T, ...]
      h_t_target = target_preds_g.state  # [T, ...]

      T = actions.shape[0]
      if self.all_goals_td == "mb_peng_lambda":
        qa_tm1 = rlax.batched_index(
          online_preds_g.q_vals[:-1], selector_actions[:-1]
        )  # [T-1]

        # Full model-based targets for greedy actions: [T]
        target_q_t_model = jax.vmap(model_based_target_q)(
          h_t_online,
          h_t_target,
          timestep_w_g,
          selector_actions,
          jax.random.split(key_grad, T),
        )  # [T] — full targets (r + γ max Q)
        target_q_t_model = (
          target_q_t_model[:-1] * timestep.discount[:-1]
        )  # [T-1] model-based

        # Separate losses: lambda (online) + model
        online_td = target_q_t_online - qa_tm1  # [T-1]
        model_td = target_q_t_model - qa_tm1  # [T-1]

        if self.dyna_other_only:
          # Replacement: lambda for on-policy, model for off-policy
          online_mask = selector_a_is_online_a[:-1].astype(jnp.float32)
          model_mask = 1.0 - online_mask
        else:
          # Additive: both everywhere
          online_mask = jnp.ones_like(loss_mask[:-1])
          model_mask = jnp.ones_like(loss_mask[:-1])

        # Mask out timesteps where off-task Q is declining (trajectory moving away from goal)
        if self.mask_declining_model in ("greedy_online", "greedy_online_episode"):
          greedy_q = jnp.max(online_preds_g.q_vals, axis=-1)  # [T]
          q_not_declining = (greedy_q[1:] >= greedy_q[:-1]).astype(jnp.float32)  # [T-1]
        elif self.mask_declining_model in (
          "greedy_target",
          "greedy_target_episode",
        ):
          greedy_q = jnp.max(target_preds_g.q_vals, axis=-1)  # [T]
          q_not_declining = (greedy_q[1:] >= greedy_q[:-1]).astype(jnp.float32)  # [T-1]
        elif self.mask_declining_model == "target_trend":
          best_target = jnp.maximum(target_q_t_online, target_q_t_model)  # [T-1]
          q_not_declining = (best_target[1:] >= best_target[:-1]).astype(
            jnp.float32
          )  # [T-2]
          q_not_declining = jnp.concatenate(
            [jnp.zeros(1), q_not_declining]
          )  # [T-1], no trend info at first timestep → suppress
        else:
          q_not_declining = None

        # Episode-level aggregation: keep/drop entire episodes based on
        # proportion of non-declining Q transitions within each episode
        if q_not_declining is not None and self.mask_declining_model.endswith(
          "_episode"
        ):
          is_first = make_float(timestep.first())  # [T]
          q_not_declining = _episode_level_mask(
            q_not_declining, is_first, self.mask_declining_threshold
          )

        if q_not_declining is not None:
          td_loss_mask = q_not_declining
          if self.mask_declining_keep_last:
            td_loss_mask = td_loss_mask + is_last[1:]  # [T-1]
          if self.keep_greedy_aligned:
            greedy_ontask = jnp.argmax(online_preds.q_vals, axis=-1)  # [T]
            greedy_aligned = (greedy_ontask == selector_actions)[:-1]  # [T-1]
            td_loss_mask = td_loss_mask + greedy_aligned
          td_loss_mask = (td_loss_mask > 0).astype(jnp.float32)
          td_loss_mask = jax.lax.stop_gradient(td_loss_mask)
        else:
          td_loss_mask = jnp.ones_like(loss_mask[:-1])

        effective_mask = loss_mask[:-1] * td_loss_mask  # [T-1]
        # Normalize online and model TD-errors by a shared scale so we
        # equalize across goals without distorting the online/model ratio.
        combined_td = online_td * online_mask + model_td * model_mask  # [T-1]
        online_td_normed = normalize_td_error(
          online_td, effective_mask, self.td_norm_style, scale_from=combined_td
        )
        model_td_normed = normalize_td_error(
          model_td, effective_mask, self.td_norm_style, scale_from=combined_td
        )
        online_loss = 0.5 * jnp.square(online_td_normed) * online_mask * effective_mask
        model_loss = 0.5 * jnp.square(model_td_normed) * model_mask * effective_mask

        ag_loss_per_t = (
          self.all_goals_coeff * online_loss + dyna_coeff * model_loss
        )  # [T-1]
        ag_batch_loss = ag_loss_per_t.sum(0) / effective_mask.sum(0).clip(min=1)
        ag_td_error = (
          online_td * online_mask + model_td * model_mask
        ) * effective_mask  # [T-1]
        target_tm1 = jnp.where(
          selector_a_is_online_a[:-1],
          target_q_t_online,
          target_q_t_model,
        )  # for logging

        ag_metrics = {
          "0.q_loss": ag_loss_per_t.mean(),
          "0.q_loss_online": online_loss.mean(),
          "0.q_loss_model": model_loss.mean(),
          "0.q_td": jnp.abs(ag_td_error).mean(),
          "0.declining_mask_frac": (1.0 - td_loss_mask).mean(),
          "1.reward": rewards.mean(),
        }
        ag_log_info = {
          "timesteps": timestep_w_g,
          "actions": actions,
          "td_errors": ag_td_error,
          "non_terminal": timestep_w_g.discount,
          "loss_mask": loss_mask,
          "q_values": online_preds_g.q_vals,
          "q_loss": ag_loss_per_t,
          "q_target": target_tm1,
          "target_q_values": target_preds_g.q_vals,  # [T, A]
          "behavior_q_values": online_preds.q_vals,
          "loss_rewards": rewards,
          "target_q_t_lambda": target_q_t_online,
          "target_q_t_model": target_q_t_model,
          "td_mask": td_loss_mask,
        }
        if self.make_log_extras is not None:
          extras = self.make_log_extras(timestep_w_g, goal=new_goal)
          ag_log_info.update(extras)
        return ag_td_error, ag_batch_loss, ag_metrics, ag_log_info

      else:
        raise NotImplementedError

    elif self.all_goals_td == "mb_all":
      # Pure 1-step model-based targets for ALL actions (1-step dyna for goal).
      # Unlike mb_peng_lambda_all/rmae, the taken action also gets a model target,
      # avoiding the conflict where retrace/λ-returns are depressed by on-task
      # terminal boundaries while model targets are not.
      h_t_online = online_preds_g.state  # [T, ...]
      h_t_target = target_preds_g.state  # [T, ...]
      T = actions.shape[0]
      num_actions = online_preds_g.q_vals.shape[-1]  # A

      # Model-based 1-step targets for ALL actions: [T, A]
      all_actions_tiled = jnp.tile(jnp.arange(num_actions), (T, 1))  # [T, A]
      keys_tiled = jax.random.split(key_grad, T * num_actions).reshape(
        T, num_actions, 2
      )  # [T, A, 2]

      # vmap: outer over T, inner over A
      target_all = jax.vmap(
        jax.vmap(model_based_target_q, in_axes=(None, None, None, 0, 0)),
        in_axes=(0, 0, 0, 0, 0),
      )(h_t_online, h_t_target, timestep_w_g, all_actions_tiled, keys_tiled)  # [T, A]

      target_all_tm1 = target_all[:-1]  # [T-1, A]

      # Apply terminal discount
      target_all_tm1 = target_all_tm1 * timestep.discount[:-1, None]  # [T-1, A]

      # Per-action TD errors
      qa_tm1_all = online_preds_g.q_vals[:-1]  # [T-1, A]
      ag_td_error_all = target_all_tm1 - qa_tm1_all  # [T-1, A]

      # Loss: mean over actions, weighted by dyna_coeff
      ag_td_error_all = ag_td_error_all * loss_mask[:-1, None]
      ag_td_error_all_normed = normalize_td_error(
        ag_td_error_all, loss_mask[:-1, None], self.td_norm_style
      )
      ag_loss_per_t = (
        dyna_coeff * 0.5 * jnp.square(ag_td_error_all_normed).mean(axis=-1)
      )  # [T-1]
      ag_td_error = ag_td_error_all.mean(axis=-1)  # [T-1]
      ag_batch_loss = (ag_loss_per_t * loss_mask[:-1]).sum(0) / loss_mask[:-1].sum(0)

      # For logging: use greedy action's target [T-1]
      selector_actions = jnp.argmax(online_preds_g.q_vals, axis=-1)
      target_tm1 = rlax.batched_index(target_all_tm1, selector_actions[:-1])

      ag_metrics = {
        "0.q_loss": ag_loss_per_t.mean(),
        "0.q_td": jnp.abs(ag_td_error).mean(),
        "1.reward": rewards.mean(),
      }
      ag_log_info = {
        "timesteps": timestep_w_g,
        "actions": actions,
        "td_errors": ag_td_error,
        "non_terminal": timestep_w_g.discount,
        "loss_mask": loss_mask,
        "q_values": online_preds_g.q_vals,
        "q_loss": ag_loss_per_t,
        "q_target": target_tm1,
        "target_q_values": target_preds_g.q_vals,
        "behavior_q_values": online_preds.q_vals,
        "loss_rewards": rewards,
      }
      if self.make_log_extras is not None:
        extras = self.make_log_extras(timestep_w_g, goal=new_goal)
        ag_log_info.update(extras)
      return ag_td_error, ag_batch_loss, ag_metrics, ag_log_info

    else:
      raise ValueError(f"Unknown all_goals_td: {self.all_goals_td}")

    # Apply loss mask and compute loss
    ag_td_error = ag_td_error * loss_mask[:-1]
    ag_td_normed = normalize_td_error(ag_td_error, loss_mask[:-1], self.td_norm_style)
    ag_loss_per_t = 0.5 * jnp.square(ag_td_normed)
    # [N] — per-goal mean loss (matches self.loss_fn return shape)
    ag_batch_loss = (
      self.all_goals_coeff
      * (ag_loss_per_t * loss_mask[:-1]).sum(0)
      / loss_mask[:-1].sum(0)
    )

    ag_metrics = {
      "0.q_loss": ag_loss_per_t.mean(),
      "0.q_td": jnp.abs(ag_td_error).mean(),
      "1.reward": rewards.mean(),
    }
    ag_log_info = {
      "timesteps": timestep_w_g,
      "actions": actions,
      "td_errors": ag_td_error,
      "non_terminal": timestep_w_g.discount,
      "loss_mask": loss_mask,
      "q_values": online_preds_g.q_vals,
      "q_loss": ag_loss_per_t,
      "q_target": target_tm1,
      "target_q_values": target_preds_g.q_vals,  # [T, A]
      "behavior_q_values": online_preds.q_vals,
      "loss_rewards": rewards,
    }
    if self.make_log_extras is not None:
      extras = self.make_log_extras(timestep_w_g, goal=new_goal)
      ag_log_info.update(extras)

    return ag_td_error, ag_batch_loss, ag_metrics, ag_log_info

  def preplay_loss_fn(
    self,
    timesteps: TimeStep,
    actions: jax.Array,
    h_online: jax.Array,
    h_target: jax.Array,
    loss_mask: jax.Array,
    rng: jax.random.PRNGKey,
    params,
    target_params,
  ):
    """Preplay loss for a single batch element.

    Vmaps over timesteps directly.

    Args:
      timesteps: [T, ...] sequence of timesteps
      actions: [T] action sequence
      h_online: [T, ...] online RNN states at each timestep
      h_target: [T, ...] target RNN states at each timestep
      loss_mask: [T] per-timestep loss mask
      rng: random key
      params: online network parameters
      target_params: target network parameters
    """
    simulate = partial(
      simulate_n_trajectories,
      network=self.network,
      params=params,
      num_steps=self.simulation_length,
      online_task_coeff=self.online_task_coeff,
      simulation_task_coeff=self.simulation_task_coeff,
      place_goal_in_timestep=self.place_goal_in_timestep,
    )

    def apply_q_head(rnn_out, p, q_method, task):
      """Apply Q-head to rnn outputs.
      rnn_out:[T, N, D],
      task:   [N, G]
      output: [T, N, A]"""

      def q_head(hidden, goal):
        return self.network.apply(p, hidden, goal, method=q_method)

      return jax.vmap(q_head, (0, None))(rnn_out, task)

    def unroll_target_rnn(h_tar_tm1, all_t, num_sims, rng):
      """Unroll target RNN on simulated timesteps (vmap-over-scan).

      Uses original x_t at step 0 (matching what the online RNN saw inside
      simulate_n_trajectories, before place_goal_in_timestep), then the
      model-generated timesteps from all_t[1:] for subsequent steps.

      Args:
        h_tar_t: target RNN state [D]
        all_t: simulated timesteps [S+1, num_sims, ...]
        num_sims: number of simulations
        rng: random key for RNN stochasticity
      Returns:
        target_rnn_out: [S+1, num_sims, D]
      """
      # [num_sims, D]
      h_tar_tm1 = jax.tree_util.tree_map(
        lambda y: jnp.broadcast_to(y, (num_sims,) + y.shape), h_tar_tm1
      )

      def step(carry, x_t):
        h_tml, rng_ = carry
        rng_, step_rng = jax.random.split(rng_)
        h_t, rnn_out = self.network.apply(
          target_params,
          h_tml,
          x_t,
          step_rng,
          task_dropout=False,
          method=self.network.apply_rnn,
        )
        return (h_t, rng_), rnn_out

      # , [T, num_sims, D]
      _, h_tar_out = jax.lax.scan(
        step,
        # [num_dims, D], [2]
        init=(h_tar_tm1, rng),
        xs=all_t,
      )

      return h_tar_out

    def preplay_at_t(x_t, h_on_tm1, h_tar_tm1, l_mask_t, key):
      """Combined on-task + off-task loss at a single timestep.

      Merges on-task and off-task simulations into a single simulate call
      with total_sims = N_on + G_off * N_off.

      Args:
        x_t: single timestep (no time dim)
        h_on_tm1: online RNN state at this timestep
        h_tar_tm1: target RNN state at this timestep
        l_mask_t: scalar loss mask
        key: random key
      """

      # Sample goals
      ontask_goal = self.get_main_goal(x_t)
      N_on, dim_G = ontask_goal.shape
      assert N_on == 1

      G_off = self.num_offtask_goals
      # [G_off, D], , [1]
      offtask_goals, _, offtask_achievable_ = self.sample_preplay_goals(x_t, key, G_off)
      N_off = self.num_offtask_simulations
      total_sims = N_on + G_off * N_off

      # Build unified goals: [total_sims, G]
      offtask_goals = jnp.tile(offtask_goals, (N_off, 1))  # [G_off*N_off, G]
      offtask_achievable = jnp.tile(offtask_achievable_, G_off * N_off)  # [G_off*N_off]
      assert offtask_goals.shape == (G_off * N_off, dim_G)
      all_sim_goals = jnp.concatenate((ontask_goal, offtask_goals), axis=0)
      assert all_sim_goals.shape[0] == total_sims

      # repeat ontask_goals for every simulation
      ontask_goals_tiled = jnp.tile(ontask_goal, (total_sims, 1))

      # Build unified epsilon vector
      key, rng_on, rng_off = jax.random.split(key, 3)
      ontask_epsilon_values = self.dyna_epsilon_values(rng_on)
      offtask_epsilon_values = self.offtask_dyna_epsilon_values(rng_off)
      assert ontask_epsilon_values.shape[0] == ontask_goal.shape[0]
      all_eps = jnp.concatenate(
        (ontask_epsilon_values, jnp.tile(offtask_epsilon_values, G_off))
      )

      def sim_policy(preds, rng):
        q_vals = preds.q_vals  # [total_sims, A]
        rngs = jax.random.split(rng, total_sims)
        return jax.vmap(base_agent.epsilon_greedy_act)(q_vals, all_eps, rngs)

      # all_t: [sim_length+1, total_sims, ...]
      # sim_out.actions: [sim_length+1, total_sims]
      key, key_ = jax.random.split(key)
      all_t, sim_out = simulate(
        h_tm1=h_on_tm1,
        x_t=x_t,
        rng=key_,
        online_goal=ontask_goals_tiled,
        simulation_goal=all_sim_goals,
        policy_fn=sim_policy,
      )

      # all_a: [sim_length+1, total_sims]
      all_t_a = sim_out.actions
      # online_rnn_out: [sim_length+1, total_sims, D]
      all_t_rnn_online = rnn_output_from_state(sim_out.predictions.state)
      # target_rnn_out: [sim_length+1, total_sims, D]
      key, key_tar = jax.random.split(key)
      all_t_rnn_target = unroll_target_rnn(h_tar_tm1, all_t, total_sims, key_tar)

      # all_mask: [sim_length+1, total_sims]
      init_loss_mask = jnp.broadcast_to(l_mask_t, (total_sims,))
      all_t_loss_mask = simulation_finished_mask(init_loss_mask, all_t)
      all_achievable = jnp.concatenate((jnp.ones(N_on), offtask_achievable))
      all_t_loss_mask = all_t_loss_mask * all_achievable[None]  # [S+1, total_sims]

      # === MAIN-TASK LOSS on ALL sims ===
      # all_main_q_on: [sim_length+1, total_sims, A]
      ontask_q_on = apply_q_head(
        all_t_rnn_online, params, self.network.main_task_q_fn, ontask_goals_tiled
      )
      # all_main_q_tar: [sim_length+1, total_sims, A]
      ontask_q_tar = apply_q_head(
        all_t_rnn_target, target_params, self.network.main_task_q_fn, ontask_goals_tiled
      )

      all_t_is_last = make_float(all_t.last())
      ontask_reward = self.compute_rewards(all_t, ontask_goals_tiled)

      # Peng's trace cutting for sim losses:
      # on-policy sims get full λ, off-policy sims cut when greedy ≠ taken
      if self.sim_peng_trace_cutting:
        # main-task: on-task sims on-policy, off-task sims off-policy
        ontask_on_policy = jnp.concatenate(
          [jnp.ones(N_on), jnp.zeros(G_off * N_off)]
        )  # [total_sims]
        ontask_selector = jnp.argmax(ontask_q_on, axis=-1)  # [S+1, total_sims]
        ontask_lambda = jnp.where(
          ontask_on_policy[None],  # [1, total_sims] -> broadcast [S+1, total_sims]
          self.lambda_,  # scalar -> broadcast [S+1, total_sims]
          (ontask_selector == all_t_a).astype(jnp.float32) * self.lambda_,  # [S+1, total_sims]
        )  # [S+1, total_sims]
      else:
        ontask_lambda = None

      ontask_td, ontask_loss, ontask_metrics, _ = self.loss_fn(
        timestep=all_t,  # [S+1, total_sims, ...]
        online_preds=Predictions(ontask_q_on, None),  # .q_vals: [S+1, total_sims, A]
        target_preds=Predictions(ontask_q_tar, None),  # .q_vals: [S+1, total_sims, A]
        actions=all_t_a,  # [S+1, total_sims]
        rewards=ontask_reward,  # [S+1, total_sims]
        is_last=all_t_is_last,  # [S+1, total_sims]
        non_terminal=all_t.discount,  # [S+1, total_sims]
        loss_mask=all_t_loss_mask,  # [S+1, total_sims]
        lambda_override=ontask_lambda,  # [S+1, total_sims] or None
      )  # returns [S, total_sims], [total_sims], dict, dict

      # === OFF-TASK GOAL LOSS (split: different goals + different rewards) ===
      # all_main_q_on: [sim_length+1, total_sims, A]
      offtask_q_on = apply_q_head(
        all_t_rnn_online, params, self.network.off_task_q_fn, all_sim_goals
      )
      # all_main_q_tar: [sim_length+1, total_sims, A]
      offtask_q_tar = apply_q_head(
        all_t_rnn_target, target_params, self.network.off_task_q_fn, all_sim_goals
      )

      offtask_reward = self.compute_rewards(all_t, all_sim_goals)

      if self.sim_peng_trace_cutting:
        # off-task: off-task sims on-policy (for own goal), on-task sims off-policy
        offtask_on_policy = jnp.concatenate(
          [jnp.zeros(N_on), jnp.ones(G_off * N_off)]
        )  # [total_sims]
        offtask_selector = jnp.argmax(offtask_q_on, axis=-1)  # [S+1, total_sims]
        offtask_lambda = jnp.where(
          offtask_on_policy[None],  # [1, total_sims] -> broadcast [S+1, total_sims]
          self.lambda_,  # scalar -> broadcast [S+1, total_sims]
          (offtask_selector == all_t_a).astype(jnp.float32) * self.lambda_,  # [S+1, total_sims]
        )  # [S+1, total_sims]
      else:
        offtask_lambda = None

      off_td, off_loss, offtask_metrics, off_log = self.loss_fn(
        timestep=all_t,  # [S+1, total_sims, ...]
        online_preds=Predictions(offtask_q_on, None),  # .q_vals: [S+1, total_sims, A]
        target_preds=Predictions(offtask_q_tar, None),  # .q_vals: [S+1, total_sims, A]
        actions=all_t_a,  # [S+1, total_sims]
        rewards=offtask_reward,  # [S+1, total_sims]
        is_last=all_t_is_last,  # [S+1, total_sims]
        non_terminal=all_t.discount,  # [S+1, total_sims]
        loss_mask=all_t_loss_mask,  # [S+1, total_sims]
        lambda_override=offtask_lambda,  # [S+1, total_sims] or None
      )  # returns [S, total_sims], [total_sims], dict, dict

      # --------------------
      # zero-out all TDs corresponding to empty goals
      # --------------------
      denominator = 2 * (N_on + offtask_achievable.sum()).astype(jnp.float32)

      # [T, total_sim]
      batch_td_error = (
        self.ontask_loss_coeff * jnp.abs(ontask_td).sum(axis=1)
        + self.offtask_loss_coeff * jnp.abs(off_td).sum(axis=1)
      ) / denominator

      # [total_sim]
      batch_loss_mean = (
        self.ontask_loss_coeff * ontask_loss.sum()
        + self.offtask_loss_coeff * off_loss.sum()
      ) / denominator

      # Metrics
      offtask_metrics[f"all_achievable"] = all_achievable.mean()
      metrics = {
        **{f"{k}/offtask": v for k, v in offtask_metrics.items()},
        **{f"{k}/ontask": v for k, v in ontask_metrics.items()},
      }
      log_info = off_log
      log_info["goal"] = all_sim_goals
      log_info["epsilon_values"] = all_eps
      log_info["simulation_reward"] = offtask_reward
      log_info["sim_ontask_q_values"] = ontask_q_on
      log_info["sim_offtask_q_values"] = offtask_q_on
      return batch_td_error, batch_loss_mean, metrics, log_info

    def dyna_only_at_t(x_t, h_on_t, h_tar_t, l_mask_t, key):
      """On-task only simulation loss (fallback when num_offtask_goals == 0).

      Args:
        x_t: single timestep (no time dim)
        h_on_t: online RNN state at this timestep
        h_tar_t: target RNN state at this timestep
        l_mask_t: scalar loss mask
        key: random key
      """
      num_sims = self.num_ontask_simulations
      # main_goal: [num_sims, G]
      main_goal = self.get_main_goal(x_t)
      # eps: [num_sims]
      key, rng_eps = jax.random.split(key)
      eps = self.dyna_epsilon_values(rng_eps)

      def sim_policy(preds, rng):
        rngs = jax.random.split(rng, num_sims)
        return jax.vmap(base_agent.epsilon_greedy_act)(preds.q_vals, eps, rngs)

      # next_t: [sim_length+1, num_sims, ...]
      # sim_outputs_t.actions: [sim_length+1, num_sims]
      key, key_ = jax.random.split(key)
      next_t, sim_outputs_t = simulate(
        h_tm1=h_on_t,
        x_t=x_t,
        rng=key_,
        online_goal=main_goal,
        policy_fn=sim_policy,
      )

      # all_t: [sim_length+1, num_sims, ...]
      all_t = next_t
      # all_a: [sim_length+1, num_sims]
      all_a = sim_outputs_t.actions

      # online_rnn_out: [sim_length+1, num_sims, D]
      online_rnn_out = rnn_output_from_state(sim_outputs_t.predictions.state)
      # target_rnn_out: [sim_length+1, num_sims, D]
      key, key_tar = jax.random.split(key)
      target_rnn_out = unroll_target_rnn(h_tar_t, all_t, num_sims, key_tar)

      if self.augment_dyna_reward is not None:
        key, key_ = jax.random.split(key)
        reward, main_goal = self.augment_dyna_reward(all_t, main_goal, key_)
      else:
        reward, main_goal = all_t.reward, main_goal

      # main_q_online: [sim_length+1, num_sims, A]
      main_q_online = apply_q_head(
        online_rnn_out, params, self.network.main_task_q_fn, main_goal
      )
      # main_q_target: [sim_length+1, num_sims, A]
      main_q_target = apply_q_head(
        target_rnn_out, target_params, self.network.main_task_q_fn, main_goal
      )

      online_preds = Predictions(q_vals=main_q_online, state=None)
      target_preds = Predictions(q_vals=main_q_target, state=None)

      # all_t_mask: [sim_length+1, num_sims]
      init_mask = jnp.broadcast_to(l_mask_t, (num_sims,))
      all_t_mask = simulation_finished_mask(init_mask, next_t)

      batch_td_error, batch_loss_mean, metrics, log_info = self.loss_fn(
        timestep=all_t,  # [S+1, num_sims, ...]
        online_preds=online_preds,  # .q_vals: [S+1, num_sims, A]
        target_preds=target_preds,  # .q_vals: [S+1, num_sims, A]
        actions=all_a,  # [S+1, num_sims]
        rewards=reward,  # [S+1, num_sims]
        is_last=make_float(all_t.last()),  # [S+1, num_sims]
        non_terminal=all_t.discount,  # [S+1, num_sims]
        loss_mask=all_t_mask,  # [S+1, num_sims]
      )  # returns [S, num_sims], [num_sims], dict, dict
      return batch_td_error, batch_loss_mean, metrics, log_info

    # Vmap over timesteps T
    loss_fn_ = preplay_at_t if self.num_offtask_goals > 0 else dyna_only_at_t

    batch_td_error, batch_loss_mean, metrics, log_info = jax.vmap(loss_fn_)(
      timesteps,  # [T, ...]
      h_online,  # [T, ...]
      h_target,  # [T, ...]
      loss_mask,  # [T]
      jax.random.split(rng, len(actions)),  # [T, 2]
    )

    batch_td_error = batch_td_error.mean()
    batch_loss_mean = batch_loss_mean.mean()

    return batch_td_error, batch_loss_mean, metrics, log_info


##############################
# make_loss_fn_class
##############################
def make_loss_fn_class(config, **kwargs) -> PreplayLossFn:
  return functools.partial(
    PreplayLossFn,
    discount=config["GAMMA"],
    lambda_=config.get("TD_LAMBDA", 0.9),
    online_coeff=config.get("ONLINE_COEFF", 1.0),
    dyna_coeff=config.get("DYNA_COEFF", 1.0),
    num_ontask_simulations=config.get("NUM_ONTASK_SIMULATIONS", 2),
    num_offtask_simulations=config.get("NUM_OFFTASK_SIMULATIONS", 2),
    simulation_length=config.get("SIMULATION_LENGTH", 5),
    importance_sampling_exponent=config.get("IMPORTANCE_SAMPLING_EXPONENT", 0.6),
    max_priority_weight=config.get("MAX_PRIORITY_WEIGHT", 0.9),
    step_cost=config.get("STEP_COST", 0.0),
    offtask_loss_coeff=config.get("OFFTASK_LOSS_COEFF", 1.0),
    ontask_loss_coeff=config.get("ONTASK_LOSS_COEFF", 1.0),
    num_offtask_goals=config.get("NUM_OFFTASK_GOALS", 5),
    simulation_task_coeff=config.get("OFFTASK_COEFF", 2.0),
    online_task_coeff=config.get("MAINQ_COEFF", 1.0),
    all_goals_coeff=config.get("ALL_GOALS_COEFF", 1.0),
    all_goals_lambda=config.get("ALL_GOALS_LAMBDA", 0.9),
    all_goals_rnn=config.get("ALL_GOALS_RNN", True),
    all_goals_td=config.get("ALL_GOALS_TD", "retrace"),
    retrace_temperature=config.get("RETRACE_TEMPERATURE", 0.1),
    peng_trace_cutting=config.get("PENG_TRACE_CUTTING", True),
    sim_peng_trace_cutting=config.get("SIM_PENG_TRACE_CUTTING", False),
    dyna_other_only=config.get("DYNA_OTHER_ONLY", True),
    mask_declining_model=config.get("MASK_DECLINING_MODEL", ""),
    mask_declining_threshold=config.get("MASK_DECLINING_THRESHOLD", 0.5),
    mask_declining_keep_last=config.get("MASK_DECLINING_KEEP_LAST", False),
    keep_greedy_aligned=config.get("KEEP_GREEDY_ALIGNED", False),
    all_goals_dyna_coeff=config.get("ALL_GOALS_DYNA_COEFF", None),
    td_norm_style=config.get("TD_NORM_STYLE", "none"),
    **kwargs,
  )


##############################
# Agent Classes
##############################


# ---- Craftax Single Goal (separate Q-fns) ----
class DynaAgentEnvModel(nn.Module):
  observation_encoder: nn.Module
  rnn: vbb.ScannedRNN
  main_q_head: nn.Module
  off_task_q_head: nn.Module
  env: environment.Environment
  env_params: environment.EnvParams

  def setup(self):
    kernel_init = nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)
    self.task_fn = nn.Dense(128, kernel_init=kernel_init)

  def initialize(self, x: TimeStep):
    rng = jax.random.PRNGKey(0)
    batch_dims = (x.reward.shape[0],)
    rnn_state = self.initialize_carry(rng, batch_dims)
    _, rnn_state = self.__call__(rnn_state, x, rng)
    task = jnp.zeros_like(x.observation.achievable)
    self.off_task_q_fn(rnn_state[1], task)

  def initialize_carry(self, *args, **kwargs):
    return self.rnn.initialize_carry(*args, **kwargs)

  def apply_rnn(self, rnn_state, x: TimeStep, rng: jax.random.PRNGKey):
    embedding = self.observation_encoder(x.observation)
    rnn_in = vbb.RNNInput(obs=embedding, reset=x.first())
    rng, _rng = jax.random.split(rng)
    return self.rnn(rnn_state, rnn_in, _rng)

  def __call__(
    self, rnn_state, x: TimeStep, rng: jax.random.PRNGKey
  ) -> Tuple[Predictions, RnnState]:
    new_rnn_state, rnn_out = self.apply_rnn(rnn_state, x, rng)
    task = jnp.zeros_like(x.observation.achievements)
    q_vals = self.main_task_q_fn(rnn_out, task)
    predictions = Predictions(q_vals=q_vals, state=new_rnn_state)
    return predictions, new_rnn_state

  def main_task_q_fn(self, rnn_out, task):
    assert task.ndim == rnn_out.ndim  # [D] or [B, D]
    assert task.ndim < 3
    q_head = self.main_q_head
    if task.ndim == 2:
      q_head = jax.vmap(q_head)
    task = self.task_fn(task)
    return q_head(rnn_out, task)

  def off_task_q_fn(self, rnn_out, task):
    assert task.ndim == rnn_out.ndim  # [D] or [B, D]
    assert task.ndim < 3
    q_head = self.off_task_q_head
    if task.ndim == 2:
      q_head = jax.vmap(q_head)
    task = self.task_fn(task)
    return q_head(rnn_out, task)

  def unroll(
    self, rnn_state, xs: TimeStep, rng: jax.random.PRNGKey
  ) -> Tuple[Predictions, RnnState]:
    embedding = jax.vmap(self.observation_encoder)(xs.observation)
    rnn_in = vbb.RNNInput(obs=embedding, reset=xs.first())
    rng, _rng = jax.random.split(rng)
    new_rnn_state, new_rnn_states = self.rnn.unroll(rnn_state, rnn_in, _rng)

    task = jnp.zeros_like(xs.observation.achievements)
    rnn_out = self.rnn.output_from_state(new_rnn_states)
    if embedding.ndim == 3:
      q_vals = jax.vmap(self.main_task_q_fn)(rnn_out, task)
    else:
      q_vals = self.main_task_q_fn(rnn_out, task)
    predictions = Predictions(q_vals=q_vals, state=new_rnn_states)
    return predictions, new_rnn_state

  def apply_model(self, state, action, rng):
    B = action.shape[0]

    def env_step(s, a, rng_):
      return self.env.step(rng_, s, a, self.env_params)

    return jax.vmap(env_step)(state, action, jax.random.split(rng, B))

  def compute_reward(self, timestep, task):
    return self.env.compute_reward(timestep, task, self.env_params)


# ---- Craftax Multigoal (shared Q-fn) ----
class DynaAgentEnvModelMultigoal(nn.Module):
  observation_encoder: nn.Module
  rnn: vbb.ScannedRNN
  main_q_head: nn.Module
  env: environment.Environment
  env_params: environment.EnvParams

  def setup(self):
    kernel_init = nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)
    self.task_fn = nn.Dense(128, kernel_init=kernel_init)

  def initialize(self, x: TimeStep):
    rng = jax.random.PRNGKey(0)
    batch_dims = (x.reward.shape[0],)
    rnn_state = self.initialize_carry(rng, batch_dims)
    _, rnn_state = self.__call__(rnn_state, x, rng)
    dummy_task = jnp.zeros_like(x.observation.state_features)
    self.off_task_q_fn(rnn_state[1], dummy_task)

  def initialize_carry(self, *args, **kwargs):
    return self.rnn.initialize_carry(*args, **kwargs)

  def apply_rnn(self, rnn_state, x: TimeStep, rng: jax.random.PRNGKey):
    embedding = self.observation_encoder(x.observation)
    rnn_in = vbb.RNNInput(obs=embedding, reset=x.first())
    rng, _rng = jax.random.split(rng)
    return self.rnn(rnn_state, rnn_in, _rng)

  def __call__(
    self, rnn_state, x: TimeStep, rng: jax.random.PRNGKey
  ) -> Tuple[Predictions, RnnState]:
    new_rnn_state, rnn_out = self.apply_rnn(rnn_state, x, rng)
    dummy_task = jnp.zeros_like(x.observation.state_features)
    q_vals = self.main_task_q_fn(rnn_out, dummy_task)
    predictions = Predictions(q_vals=q_vals, state=new_rnn_state)
    return predictions, new_rnn_state

  def main_task_q_fn(self, rnn_out, task):
    assert task.ndim == rnn_out.ndim  # [D] or [B, D]
    assert task.ndim < 3
    q_head = self.main_q_head
    if task.ndim == 2:
      q_head = jax.vmap(q_head)
    task = self.task_fn(task)
    return q_head(rnn_out, task)

  def off_task_q_fn(self, rnn_out, task):
    return self.main_task_q_fn(rnn_out, task)

  def unroll(
    self, rnn_state, xs: TimeStep, rng: jax.random.PRNGKey
  ) -> Tuple[Predictions, RnnState]:
    embedding = jax.vmap(self.observation_encoder)(xs.observation)
    rnn_in = vbb.RNNInput(obs=embedding, reset=xs.first())
    rng, _rng = jax.random.split(rng)
    new_rnn_state, new_rnn_states = self.rnn.unroll(rnn_state, rnn_in, _rng)

    dummy_task = jnp.zeros_like(xs.observation.state_features)
    rnn_out = self.rnn.output_from_state(new_rnn_states)
    if embedding.ndim == 3:
      q_vals = jax.vmap(self.main_task_q_fn)(rnn_out, dummy_task)
    else:
      q_vals = self.main_task_q_fn(rnn_out, dummy_task)
    predictions = Predictions(q_vals=q_vals, state=new_rnn_states)
    return predictions, new_rnn_state

  def apply_model(self, state, action, rng):
    B = action.shape[0]

    def env_step(s, a, rng_):
      return self.env.step(rng_, s, a, self.env_params)

    return jax.vmap(env_step)(state, action, jax.random.split(rng, B))

  def compute_reward(self, timestep, task):
    return self.env.compute_reward(timestep, task, self.env_params)


# ---- JaxMaze Multigoal (shared Q-fn) ----
class DynaAgentEnvModelMultigoalJaxMaze(nn.Module):
  """JaxMaze agent with shared Q-head for on-task and off-task goals.

  task_dropout_rate: Randomly zero out task_w in observations during training.
    This prevents over-reliance on task embedding in RNN states, enabling better
    generalization for off-task Q-value estimation (e.g., in all_goals loss).
    Inspired by classifier-free guidance (Ho & Salimans, 2022):
    https://arxiv.org/abs/2207.12598
  """

  observation_encoder: nn.Module
  rnn: vbb.ScannedRNN
  main_q_head: nn.Module
  env: environment.Environment
  env_params: environment.EnvParams
  task_dropout_rate: float = 0.0  # 0.0 = disabled, 0.1-0.2 recommended

  def setup(self):
    kernel_init = nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)
    self.task_fn = nn.Dense(128, kernel_init=kernel_init)
    self.off_task_q_head = self.main_q_head

  def _maybe_dropout_task(self, obs, rng):
    """Zero out entire task_w with probability task_dropout_rate."""
    if self.task_dropout_rate <= 0.0:
      return obs
    keep_prob = 1.0 - self.task_dropout_rate
    keep_shape = obs.task_w.shape[:-1]  # [B] or () depending on input
    keep = jax.random.bernoulli(rng, p=keep_prob, shape=keep_shape)
    keep = keep[..., None]  # [B, 1] or [1] for broadcasting with [B, G] or [G]
    new_task_w = obs.task_w * keep
    return obs.replace(task_w=new_task_w)

  def initialize(self, x: TimeStep):
    rng = jax.random.PRNGKey(0)
    batch_dims = (x.reward.shape[0],)
    rnn_state = self.initialize_carry(rng, batch_dims)
    _, rnn_state = self.__call__(rnn_state, x, rng)
    task = x.state.task_w
    self.off_task_q_fn(rnn_state[1], task)

  def initialize_carry(self, *args, **kwargs):
    return self.rnn.initialize_carry(*args, **kwargs)

  def apply_rnn(
    self, rnn_state, x: TimeStep, rng: jax.random.PRNGKey, task_dropout: bool = True
  ):
    rng, dropout_rng, rnn_rng = jax.random.split(rng, 3)
    obs = (
      self._maybe_dropout_task(x.observation, dropout_rng)
      if task_dropout
      else x.observation
    )
    embedding = self.observation_encoder(obs)
    assert embedding.ndim in (1, 2)  # [D] or [B, D]
    rnn_in = vbb.RNNInput(obs=embedding, reset=x.first())
    return self.rnn(rnn_state, rnn_in, rnn_rng)

  def __call__(
    self, rnn_state, x: TimeStep, rng: jax.random.PRNGKey
  ) -> Tuple[Predictions, RnnState]:
    """
    rnn_state: [B]
    x: [B]
    """
    new_rnn_state, rnn_out = self.apply_rnn(rnn_state, x, rng, task_dropout=False)
    task = x.observation.task_w
    q_vals = self.main_task_q_fn(rnn_out, task)
    predictions = Predictions(q_vals=q_vals, state=new_rnn_state)
    return predictions, new_rnn_state

  def unroll(
    self, rnn_state, xs: TimeStep, rng: jax.random.PRNGKey, task_dropout: bool = True
  ) -> Tuple[Predictions, RnnState]:
    """
    rnn_state: [B]
    x: [T, B]
    """
    rng, dropout_rng, rnn_rng = jax.random.split(rng, 3)
    T = xs.observation.task_w.shape[0]
    if task_dropout:
      dropout_keys = jax.random.split(dropout_rng, T)
      obs = jax.vmap(self._maybe_dropout_task)(xs.observation, dropout_keys)
    else:
      obs = xs.observation
    embedding = jax.vmap(self.observation_encoder)(obs)
    assert embedding.ndim in (2, 3)

    rnn_in = vbb.RNNInput(obs=embedding, reset=xs.first())
    new_rnn_state, new_rnn_states = self.rnn.unroll(rnn_state, rnn_in, rnn_rng)
    task = xs.observation.task_w
    rnn_out = self.rnn.output_from_state(new_rnn_states)

    if embedding.ndim == 3:
      # [T, B D]
      q_vals = jax.vmap(self.main_task_q_fn)(rnn_out, task)
    else:
      # [T, D]
      q_vals = self.main_task_q_fn(rnn_out, task)
    predictions = Predictions(q_vals=q_vals, state=new_rnn_states)
    return predictions, new_rnn_state

  def main_task_q_fn(self, rnn_out, task):
    assert task.ndim == rnn_out.ndim  # [D] or [B, D]
    assert task.ndim < 3
    q_head = self.main_q_head
    if task.ndim == 2:
      q_head = jax.vmap(q_head)
    task = self.task_fn(task)
    return q_head(rnn_out, task)

  def off_task_q_fn(self, rnn_out, task):
    assert task.ndim == rnn_out.ndim  # [D] or [B, D]
    assert task.ndim < 3
    q_head = self.off_task_q_head
    if task.ndim == 2:
      q_head = jax.vmap(q_head)
    task = self.task_fn(task)
    return q_head(rnn_out, task)

  def apply_model(self, state, action, rng):
    B = action.shape[0]

    def env_step(s, a, rng_):
      return self.env.step(rng_, s, a, self.env_params)

    return jax.vmap(env_step)(state, action, jax.random.split(rng, B))

  def compute_reward(self, timestep, task):
    return self.env.compute_reward(timestep, task, self.env_params)


##############################
# make_*_agent functions
##############################
def make_craftax_singlegoal_agent(
  config: dict,
  env: environment.Environment,
  env_params: environment.EnvParams,
  example_timestep: TimeStep,
  rng: jax.random.PRNGKey,
  model_env: Optional[environment.Environment] = None,
  model_env_params: Optional[environment.EnvParams] = None,
) -> Tuple[nn.Module, Params, vbb.AgentResetFn]:
  model_env_params = model_env_params or env_params
  rnn = vbb.ScannedRNN(
    hidden_dim=config.get("AGENT_RNN_DIM", 256),
    cell_type=config.get("RNN_CELL_TYPE", "OptimizedLSTMCell"),
    unroll_output_state=True,
  )

  qhead_type = config.get("QHEAD_TYPE", "dot")
  if qhead_type == "dot":
    QFnCls = DuellingDotMLP
  else:
    QFnCls = DuellingMLP

  agent = DynaAgentEnvModel(
    observation_encoder=CraftaxObsEncoder(
      hidden_dim=config["MLP_HIDDEN_DIM"],
      num_layers=config["NUM_MLP_LAYERS"],
      activation=config["ACTIVATION"],
      norm_type=config.get("NORM_TYPE", "none"),
      structured_inputs=config.get("STRUCTURED_INPUTS", False),
      use_bias=config.get("USE_BIAS", True),
      include_achievable=config.get("INCLUDE_ACHIEVABLE", True),
      action_dim=env.action_space(env_params).n,
    ),
    rnn=rnn,
    main_q_head=QFnCls(
      hidden_dim=config.get("Q_HIDDEN_DIM", 512),
      num_layers=config.get("NUM_Q_LAYERS", 1),
      activation=config["ACTIVATION"],
      activate_final=False,
      use_bias=config.get("USE_BIAS", True),
      out_dim=env.action_space(env_params).n,
    ),
    off_task_q_head=QFnCls(
      hidden_dim=config.get("Q_HIDDEN_DIM", 512),
      num_layers=config.get("NUM_AUX_LAYERS", 0),
      activation=config["ACTIVATION"],
      activate_final=False,
      use_bias=config.get("USE_BIAS", True),
      out_dim=env.action_space(env_params).n,
    ),
    env=model_env,
    env_params=model_env_params,
  )

  rng, _rng = jax.random.split(rng)
  network_params = agent.init(_rng, example_timestep, method=agent.initialize)

  def reset_fn(params, example_timestep, reset_rng):
    batch_dims = example_timestep.reward.shape
    return agent.apply(
      params, batch_dims=batch_dims, rng=reset_rng, method=agent.initialize_carry
    )

  return agent, network_params, reset_fn


def make_craftax_multigoal_agent(
  config: dict,
  env: environment.Environment,
  env_params: environment.EnvParams,
  example_timestep: TimeStep,
  rng: jax.random.PRNGKey,
  model_env: Optional[environment.Environment] = None,
  model_env_params: Optional[environment.EnvParams] = None,
) -> Tuple[nn.Module, Params, vbb.AgentResetFn]:
  model_env_params = model_env_params or env_params
  rnn = vbb.ScannedRNN(
    hidden_dim=config.get("AGENT_RNN_DIM", 256),
    cell_type=config.get("RNN_CELL_TYPE", "OptimizedLSTMCell"),
    unroll_output_state=True,
  )

  qhead_type = config.get("QHEAD_TYPE", "dot")
  if qhead_type == "dot":
    QFnCls = DuellingDotMLP
  else:
    QFnCls = DuellingMLP

  agent = DynaAgentEnvModelMultigoal(
    observation_encoder=CraftaxMultiGoalObsEncoder(
      hidden_dim=config["MLP_HIDDEN_DIM"],
      num_layers=config["NUM_MLP_LAYERS"],
      activation=config["ACTIVATION"],
      norm_type=config.get("NORM_TYPE", "none"),
      use_bias=config.get("USE_BIAS", True),
      include_goal=config.get("OBS_INCLUDE_GOAL", False),
      action_dim=env.action_space(env_params).n,
    ),
    rnn=rnn,
    main_q_head=QFnCls(
      hidden_dim=config.get("Q_HIDDEN_DIM", 512),
      num_layers=config.get("NUM_PRED_LAYERS", 1),
      activation=config["ACTIVATION"],
      activate_final=False,
      use_bias=config.get("USE_BIAS", True),
      out_dim=env.action_space(env_params).n,
    ),
    env=model_env,
    env_params=model_env_params,
  )

  rng, _rng = jax.random.split(rng)
  network_params = agent.init(_rng, example_timestep, method=agent.initialize)

  def reset_fn(params, example_timestep, reset_rng):
    batch_dims = example_timestep.reward.shape
    return agent.apply(
      params, batch_dims=batch_dims, rng=reset_rng, method=agent.initialize_carry
    )

  return agent, network_params, reset_fn


def make_jaxmaze_multigoal_agent(
  config: dict,
  env: environment.Environment,
  env_params: environment.EnvParams,
  example_timestep: TimeStep,
  rng: jax.random.PRNGKey,
  model_env: Optional[environment.Environment] = None,
  model_env_params: Optional[environment.EnvParams] = None,
) -> Tuple[nn.Module, Params, vbb.AgentResetFn]:
  model_env_params = model_env_params or env_params
  rnn = vbb.ScannedRNN(
    hidden_dim=config.get("AGENT_RNN_DIM", 256),
    cell_type=config.get("RNN_CELL_TYPE", "OptimizedLSTMCell"),
    unroll_output_state=True,
  )

  qhead_type = config.get("QHEAD_TYPE", "duelling")
  if qhead_type == "dot":
    QFnCls = DuellingDotMLP
  else:
    QFnCls = DuellingMLP

  observation_encoder = CategoricalJaxmazeObsEncoder(
    num_categories=max(10_000, env.total_categories(env_params)),
    embed_hidden_dim=config["EMBED_HIDDEN_DIM"],
    mlp_hidden_dim=config["MLP_HIDDEN_DIM"],
    num_embed_layers=config["NUM_EMBED_LAYERS"],
    num_mlp_layers=config["NUM_MLP_LAYERS"],
    activation=config["ACTIVATION"],
    norm_type=config.get("NORM_TYPE", "none"),
    include_task=config.get("OBS_INCLUDE_GOAL", True),
  )

  agent = DynaAgentEnvModelMultigoalJaxMaze(
    observation_encoder=observation_encoder,
    rnn=rnn,
    main_q_head=QFnCls(
      hidden_dim=config.get("Q_HIDDEN_DIM", 512),
      num_layers=config.get("NUM_PRED_LAYERS", 1),
      activation=config["ACTIVATION"],
      activate_final=False,
      use_bias=config.get("USE_BIAS", False),
      out_dim=env.num_actions(env_params),
    ),
    env=model_env,
    env_params=model_env_params,
    task_dropout_rate=config.get("TASK_DROPOUT_RATE", 0.0),
  )

  rng, _rng = jax.random.split(rng)
  network_params = agent.init(_rng, example_timestep, method=agent.initialize)

  def reset_fn(params, example_timestep, reset_rng):
    batch_dims = example_timestep.reward.shape
    return agent.apply(
      params, batch_dims=batch_dims, rng=reset_rng, method=agent.initialize_carry
    )

  return agent, network_params, reset_fn


##############################
# make_train_* functions
##############################
def make_train_craftax_singlegoal(**kwargs):
  config = kwargs["config"]
  # ACME default epsilon schedule - in range of ~(0, .1)
  vals = np.logspace(num=256, start=1, stop=3, base=0.1)

  num_offtask_simulations = config["NUM_OFFTASK_SIMULATIONS"]
  num_ontask_simulations = config.get("NUM_ONTASK_SIMULATIONS", 1)
  assert num_ontask_simulations == 1

  def dyna_epsilon_values(rng):
    """Returns: epsilons: [num_ontask_simulations] all greedy."""
    return jnp.zeros(num_ontask_simulations)  # [num_ontask_simulations]

  def offtask_dyna_epsilon_values(rng):
    """Returns: epsilons: [num_offtask_simulations] with first greedy."""
    epsilons = jax.random.choice(
      rng, vals, shape=(num_offtask_simulations - 1,)
    )  # [num_offtask_simulations-1]
    return jnp.concatenate((jnp.array([0.0]), epsilons))  # [num_offtask_simulations]

  def sample_visible_goals(timestep, key, num_offtask_goals):
    """Sample goals from currently achievable achievements.

    Args:
      timestep: single timestep (no batch/time dims)
      key: random key
      num_offtask_goals: number of goals to sample (N)
    Returns:
      goals: [N, G] one-hot goal vectors
      achievable: [G] achievability scores
      any_achievable: scalar bool
    """
    achievable = timestep.observation.achievable.astype(jnp.float32)  # [G]
    any_achievable = achievable.sum() > 0.1
    achieve_poss = achievable + 1e-5  # [G]
    achieve_poss = achieve_poss / achieve_poss.sum()  # [G]
    key, key_ = jax.random.split(key)
    goals = distrax.Categorical(probs=achieve_poss).sample(
      seed=key_, sample_shape=(num_offtask_goals)
    )  # [N]
    num_classes = timestep.observation.achievable.shape[-1]
    goals = jax.nn.one_hot(goals, num_classes=num_classes)  # [N, G]
    return goals, achievable, any_achievable

  num_random_goals = num_offtask_simulations * config["NUM_OFFTASK_GOALS"]

  def sample_random_goals(timestep, key):
    """Sample uniformly random goals.

    Args:
      timestep: single timestep (no batch/time dims)
      key: random key
      num_offtask_goals: number of goals to sample (N)
    Returns:
      goals: [N, G] one-hot goal vectors
      achievable: [G] uniform achievability
      any_achievable: scalar bool (always True)
    """
    num_classes = timestep.observation.achievable.shape[-1]
    key, key_ = jax.random.split(key)
    goals = jax.random.randint(
      key_, shape=(num_random_goals,), minval=0, maxval=num_classes
    )  # [N]
    goals = jax.nn.one_hot(goals, num_classes=num_classes)  # [N, G]
    achievable = jnp.ones(num_classes) / num_classes  # [G]
    any_achievable = jnp.bool_(True)
    return goals, achievable, any_achievable

  def compute_rewards(timesteps, goal_onehot):
    """Compute rewards for given goals across timesteps.

    Args:
      timesteps: [T, num_sims, ...] or [T, ...] batch of timesteps
      goal_onehot: [num_sims, G] or [G] one-hot goal vectors
    Returns:
      offtask_reward: [T, num_sims] or [T] reward per timestep
    """
    achievements = timesteps.observation.achievements.astype(jnp.float32)  # [T, ..., G]
    achievement_coefficients = timesteps.observation.task_w.astype(
      jnp.float32
    )  # [T, ..., G]
    assert goal_onehot.ndim == achievements.ndim - 1
    achievement_coefficients = achievement_coefficients[..., : goal_onehot.shape[-1]]
    coeff_mask = goal_onehot.astype(jnp.float32)  # [num_sims, G] or [G]
    # coeff_mask[None]: [1, num_sims, G] broadcasts with [T, num_sims, G]
    offtask_reward = (achievements * achievement_coefficients * coeff_mask[None]).sum(
      -1
    )  # [T, num_sims] or [T]
    return offtask_reward

  num_ontask_simulations = config.get("NUM_ONTASK_SIMULATIONS", 2)

  def get_main_goal(timestep):
    """Get on-task goal tiled for each simulation.

    Args:
      timestep: single timestep (no batch/time dims)
    Returns:
      goals: [num_ontask_simulations, G] zero goals (singlegoal uses env reward)
    """
    return jnp.zeros(
      (num_ontask_simulations, timestep.observation.achievements.shape[-1])
    )  # [num_ontask_simulations, G]

  return vbb.make_train(
    make_agent=partial(
      make_craftax_singlegoal_agent, model_env=kwargs.pop("model_env")
    ),
    make_loss_fn_class=functools.partial(
      make_loss_fn_class,
      dyna_epsilon_values=dyna_epsilon_values,
      offtask_dyna_epsilon_values=offtask_dyna_epsilon_values,
      sample_preplay_goals=sample_visible_goals,
      sample_td_goals=sample_random_goals,
      compute_rewards=compute_rewards,
      get_main_goal=get_main_goal,
    ),
    make_optimizer=base_agent.make_optimizer,
    make_actor=make_actor,
    **kwargs,
  )


def make_train_craftax_singlegoal_dyna_multigoal(**kwargs):
  config = kwargs["config"]
  config["ALL_GOALS_COEFF"] = 0.0
  config["DYNA_COEFF"] = 1.0
  config["OFFTASK_COEFF"] = 0.0
  config["OFFTASK_LOSS_COEFF"] = 0.0

  # ACME default epsilon schedule
  vals = np.logspace(num=256, start=1, stop=3, base=0.1)

  num_offtask_goals = config["NUM_OFFTASK_GOALS"]
  num_offtask_simulations = config["NUM_OFFTASK_SIMULATIONS"]
  num_ontask_simulations = config.get("NUM_ONTASK_SIMULATIONS", 1)
  num_simulations = num_ontask_simulations + num_offtask_simulations

  def dyna_epsilon_values(rng):
    """Sample epsilon values for all simulations.

    Returns:
      epsilons: [num_simulations] with first always greedy (0.0)
    """
    epsilons = jax.random.choice(
      rng, vals, shape=(num_simulations - 1,)
    )  # [num_simulations-1]
    return jnp.concatenate((jnp.array([0.0]), epsilons))  # [num_simulations]

  def sample_random_goals(timestep, key, num):
    """Sample uniformly random goals.

    Args:
      timestep: single timestep (no batch/time dims)
      key: random key
      num_offtask_goals: number of goals to sample (N)
    Returns:
      goals: [N, G] one-hot goal vectors
      achievable: [G] uniform achievability
      any_achievable: scalar bool (always True)
    """
    num_classes = timestep.observation.achievable.shape[-1]
    key, key_ = jax.random.split(key)
    goals = jax.random.randint(key_, shape=(num,), minval=0, maxval=num_classes)  # [N]
    goals = jax.nn.one_hot(goals, num_classes=num_classes)  # [N, G]
    achievable = jnp.ones(num_classes) / num_classes  # [G]
    any_achievable = jnp.bool_(True)
    return goals, achievable, any_achievable

  def compute_rewards(timesteps, goal_onehot):
    """Compute rewards for given goals across timesteps.

    Args:
      timesteps: [T, num_sims, ...] batch of timesteps
      goal_onehot: [num_sims, G] one-hot goal vectors
    Returns:
      offtask_reward: [T, num_sims] reward per timestep
    """
    achievements = timesteps.observation.achievements.astype(
      jnp.float32
    )  # [T, num_sims, G]
    achievement_coefficients = timesteps.observation.task_w.astype(
      jnp.float32
    )  # [T, num_sims, G]
    assert goal_onehot.ndim == achievements.ndim - 1
    achievement_coefficients = achievement_coefficients[..., : goal_onehot.shape[-1]]
    coeff_mask = goal_onehot.astype(jnp.float32)  # [num_sims, G]
    # coeff_mask[None]: [1, num_sims, G] broadcasts with [T, num_sims, G]
    offtask_reward = (achievements * achievement_coefficients * coeff_mask[None]).sum(
      -1
    )  # [T, num_sims]
    return offtask_reward

  def augment_dyna_reward(all_t, original_goal, rng):
    """Swap goals for half the simulations and recompute rewards.

    Args:
        all_t: [sim_length+1, num_sims, ...]
        original_goal: [num_sims, G]
        rng: random key
    Returns:
        new_rewards: [sim_length+1, num_sims]
        new_goals: [num_sims, G]
    """
    num_sims = original_goal.shape[0]
    n_keep_goal = num_sims // 2
    n_change_goal = num_sims - n_keep_goal

    # sample random goals for the change portion
    # use last timestep for achievable info
    last_t = jax.tree_util.tree_map(lambda x: x[-1, 0], all_t)  # single timestep
    random_goals, _, _ = sample_random_goals(
      last_t, rng, n_change_goal
    )  # [n_change_goal, G]

    # new_goals: [num_sims, G] = concat([n_keep_goal, G], [n_change_goal, G])
    new_goals = jnp.concatenate((original_goal[:n_keep_goal], random_goals), axis=0)
    # new_rewards: [sim_length+1, num_sims]
    new_rewards = compute_rewards(all_t, new_goals)
    return new_rewards, new_goals

  def get_main_goal(timestep):
    """Tile the current task goal for all simulations.

    Args:
      timestep: single timestep (no batch/time dims), task_w is [G]
    Returns:
      goals: [num_simulations, G]
    """
    goal = timestep.observation.task_w[None]  # [1, G]
    return jnp.tile(goal, (num_simulations, 1))  # [num_simulations, G]

  return vbb.make_train(
    make_agent=partial(
      make_craftax_singlegoal_agent, model_env=kwargs.pop("model_env")
    ),
    make_loss_fn_class=functools.partial(
      make_loss_fn_class,
      dyna_epsilon_values=dyna_epsilon_values,
      get_main_goal=get_main_goal,
      augment_dyna_reward=augment_dyna_reward,
    ),
    make_optimizer=base_agent.make_optimizer,
    make_actor=make_actor,
    **kwargs,
  )


def make_train_craftax_multigoal(**kwargs):
  config = kwargs["config"]
  vals = np.logspace(num=256, start=1, stop=3, base=0.1)

  num_offtask_simulations = config["NUM_OFFTASK_SIMULATIONS"]
  num_ontask_simulations = config.get("NUM_ONTASK_SIMULATIONS", 1)
  assert num_ontask_simulations == 1

  def dyna_epsilon_values(rng):
    """Returns: epsilons: [num_ontask_simulations] all greedy."""
    return jnp.zeros(num_ontask_simulations)  # [num_ontask_simulations]

  def offtask_dyna_epsilon_values(rng):
    """Returns: epsilons: [num_offtask_simulations] with first greedy."""
    epsilons = jax.random.choice(
      rng, vals, shape=(num_offtask_simulations - 1,)
    )  # [num_offtask_simulations-1]
    return jnp.concatenate((jnp.array([0.0]), epsilons))  # [num_offtask_simulations]

  def sample_all_tasks(timestep, key):
    """Return all tasks as one-hot identity matrix.

    Args:
      timestep: single timestep (no batch/time dims), task_w is [1, G]
      key: random key (unused)
    Returns:
      goals: [G, G] identity matrix (all one-hot goal vectors)
      achievable: [G] uniform achievability
      any_achievable: scalar bool
    """
    num_goals = timestep.observation.task_w.shape[-1]
    goals = jnp.eye(num_goals)  # [G, G] identity = all one-hots
    achievable = jnp.ones(num_goals)
    any_achievable = jnp.bool_(True)
    return goals, achievable, any_achievable

  def sample_nontask_visible_goals(timestep, key, num_offtask_goals):
    """Sample goals from nearby non-task objects.

    Args:
      timestep: single timestep (no batch/time dims)
      key: random key
      num_offtask_goals: number of goals to sample (N)
    Returns:
      goals: [N, G] one-hot goal vectors
      achievable: [G] achievability scores (nearby minus task objects)
      any_achievable: scalar bool
    """
    is_object_nearby = timestep.observation.nearby_objects.astype(jnp.int32)  # [G, ...]
    is_object_nearby = is_object_nearby.sum(-1)  # [G]
    task_object = timestep.observation.task_w.astype(jnp.int32)  # [G]
    achievable = nn.relu((is_object_nearby - task_object).astype(jnp.float32))  # [G]
    achievable_ = achievable + 1e-5  # [G]
    probabilities = achievable_ / achievable_.sum(-1)  # [G]
    key, key_ = jax.random.split(key)
    goals = distrax.Categorical(probs=probabilities).sample(
      seed=key_, sample_shape=(num_offtask_goals)
    )  # [N]
    num_classes = len(is_object_nearby)
    goals = jax.nn.one_hot(
      goals, num_classes=num_classes, dtype=timestep.observation.task_w.dtype
    )  # [N, G]
    any_achievable = achievable.sum() > 0.1
    return goals, achievable, any_achievable

  def compute_rewards(timesteps, goal_onehot):
    """Compute rewards using state features and goal vectors.

    Args:
      timesteps: [T, num_sims, ...] batch of timesteps
      goal_onehot: [num_sims, G] one-hot goal vectors
    Returns:
      offtask_reward: [T, num_sims] reward per timestep
    """
    state_features = timesteps.observation.state_features.astype(
      jnp.float32
    )  # [T, num_sims, G]
    # goal_onehot[None]: [1, num_sims, G] broadcasts with [T, num_sims, G]
    offtask_reward = (state_features * goal_onehot.astype(jnp.float32)[None]).sum(
      -1
    )  # [T, num_sims]
    return offtask_reward

  num_ontask_simulations = config.get("NUM_ONTASK_SIMULATIONS", 2)

  def get_main_goal(timestep):
    """Tile the current task goal for on-task simulations.

    Args:
      timestep: single timestep (no batch/time dims), task_w is [G]
    Returns:
      goals: [num_ontask_simulations, G]
    """
    goal = timestep.observation.task_w[None]  # [G] -> [1, G]
    return jnp.tile(goal, (num_ontask_simulations, 1))  # [num_ontask_simulations, G]

  def place_goal_in_timestep(timestep, goal):
    """Create a new timestep with the given goal swapped in.

    Args:
      timestep: single timestep (no batch/time dims)
      goal: [G] one-hot goal vector
    Returns:
      new timestep with goal set in state and observation
    """
    achievement_goal = jax.lax.dynamic_index_in_dim(
      craftax_web_env.IDX_to_Achievement, goal.argmax(-1), keepdims=False
    )
    new_state = timestep.state.replace(
      env_state=timestep.state.env_state.replace(
        current_goal=achievement_goal,
      ),
    )
    new_observation = timestep.observation.replace(task_w=goal)
    return timestep.replace(state=new_state, observation=new_observation)

  def make_log_extras(timestep, goal=None):
    """Env-specific: extract goal reward, task vector, achievements from timestep."""
    state_features = timestep.observation.state_features.astype(jnp.float32)
    task_w = timestep.observation.task_w.astype(jnp.float32)
    goal_vec = goal if goal is not None else task_w
    goal_vec = jnp.broadcast_to(goal_vec, state_features.shape)
    goal_reward = (goal_vec * state_features).sum(-1)
    return {
      "goal_reward": goal_reward,
      "goal_task_vector": goal_vec,
      "goal_achievements": state_features,
    }

  return vbb.make_train(
    make_agent=partial(make_craftax_multigoal_agent, model_env=kwargs.pop("model_env")),
    make_loss_fn_class=functools.partial(
      make_loss_fn_class,
      dyna_epsilon_values=dyna_epsilon_values,
      offtask_dyna_epsilon_values=offtask_dyna_epsilon_values,
      sample_preplay_goals=sample_nontask_visible_goals,
      sample_td_goals=sample_all_tasks,
      compute_rewards=compute_rewards,
      get_main_goal=get_main_goal,
      place_goal_in_timestep=place_goal_in_timestep,
      make_log_extras=make_log_extras,
    ),
    make_optimizer=base_agent.make_optimizer,
    make_actor=make_actor,
    **kwargs,
  )


def make_train_jaxmaze_multigoal(**kwargs):
  config = kwargs["config"]
  all_tasks = jnp.array(kwargs.pop("all_tasks"))
  known_offtask_goal = config.get("KNOWN_OFFTASK_GOAL", False)
  num_offtask_goals = 1 if known_offtask_goal else config["NUM_OFFTASK_GOALS"]
  config["NUM_OFFTASK_GOALS"] = num_offtask_goals
  distance_coeff = config.get("DISTANCE_COEFF", 0.0)
  epsilon_setting = config.get("SIM_EPSILON_SETTING", 1)

  if epsilon_setting == 1:
    # ACME default
    # range of ~(0.001, .1)
    vals = np.logspace(num=256, start=1, stop=3, base=0.1)
  elif epsilon_setting == 2:
    # range of ~(.9,.1)
    vals = np.logspace(num=256, start=0.05, stop=0.9, base=0.1)
  elif epsilon_setting == 3:
    # very random
    vals = np.ones(256) * 0.9

  num_offtask_simulations = config["NUM_OFFTASK_SIMULATIONS"]
  num_ontask_simulations = config["NUM_ONTASK_SIMULATIONS"] = 1

  def dyna_epsilon_values(rng):
    """Returns: epsilons: [num_ontask_simulations] all greedy."""
    return jnp.zeros(num_ontask_simulations)  # [num_ontask_simulations]

  def offtask_dyna_epsilon_values(rng):
    """Returns: epsilons: [num_offtask_simulations] with first greedy."""
    epsilons = jax.random.choice(
      rng, vals, shape=(num_offtask_simulations - 1,)
    )  # [num_offtask_simulations-1]
    return jnp.concatenate((jnp.array([0.0]), epsilons))  # [num_offtask_simulations]

  def sample_nontask_visible_goals(timestep, key, num):
    """Sample goals from nearby non-task objects.

    Args:
      timestep: single timestep (no batch/time dims)
      key: random key
      num_offtask_goals: number of goals to sample (N)
    Returns:
      goals: [N, G] one-hot goal vectors
      achievable: [G] achievability scores (nearby minus task objects)
      any_achievable: scalar bool
    """

    is_object_nearby = timestep.observation.nearby_objects.astype(jnp.int32)  # [G]
    task_object = timestep.observation.task_w.astype(jnp.int32)  # [G]
    achievable = nn.relu((is_object_nearby - task_object).astype(jnp.float32))  # [G]
    achievable_ = achievable + 1e-5  # [G]
    probabilities = achievable_ / achievable_.sum(-1)  # [G]
    key, key_ = jax.random.split(key)
    goals = distrax.Categorical(probs=probabilities).sample(
      seed=key_, sample_shape=(num)
    )  # [N]
    num_classes = len(is_object_nearby)
    goals = jax.nn.one_hot(
      goals, num_classes=num_classes, dtype=timestep.observation.task_w.dtype
    )  # [N, G]
    any_achievable = achievable.sum() > 0.1
    return goals, achievable, any_achievable

  def sample_all_tasks(timestep, key):
    """Return all tasks directly.

    Args:
      timestep: single timestep (no batch/time dims)
      key: random key (unused)
    Returns:
      goals: [G, G] all task vectors
      achievable: [G] uniform achievability
      any_achievable: scalar bool
    """
    goals = all_tasks.astype(timestep.state.offtask_w.dtype)
    achievable = jnp.ones(len(goals))
    any_achievable = jnp.bool_(True)
    return goals, achievable, any_achievable

  def compute_goal_distance_reward(timestep):
    # get agent position [T, B, 2] --> [T, B, 1, 2]
    player_position = jnp.expand_dims(timestep.observation.player_position, -2).astype(
      jnp.float32
    )

    # get object position # [T, B, N, 2]
    obj_pos = timestep.observation.object_positions.astype(jnp.float32)

    # compute distance [T, B, N]
    distance = jnp.sqrt(jnp.square(player_position - obj_pos).sum(-1))
    return -distance
    # assert distance.shape == obj_pos.shape[:-1]

    ## [T-1, B, G]
    # improvements = distance[1:] - distance[:-1]

    ## [T, B, G], should be safe since last step is always ignored
    # improvements = jnp.concatenate((improvements, jnp.zeros_like(improvements[:1])))

    ## mask out improvements for objects already collected
    ## object_positions == -1 when collected (see craftax_web_env.py:766)
    # collected = (timestep.observation.object_positions == -1).all(axis=-1)  # [T, B, N]
    # improvements = improvements * (~collected).astype(improvements.dtype)

    # return improvements

  def compute_rewards(timesteps, goal_onehot, expand_g_across_time=True):
    """Compute rewards using state features and goal vectors.

    Args:
      timesteps: [T, B, ...] batch of timesteps
      goal_onehot: [B, G] or [T, B, G] one-hot goal vectors
    Returns:
      offtask_reward: [T, B] reward per timestep
    """
    # [T, B, G]
    state_features = timesteps.state.task_state.features.astype(jnp.float32)
    goal_onehot = goal_onehot.astype(jnp.float32)
    if expand_g_across_time:
      assert goal_onehot.ndim == state_features.ndim - 1
      # goal_onehot[None]: [1, B, G] broadcasts with [T, B, G]
      goal_onehot = goal_onehot[None]

      # [T, B]
      reward = (state_features * goal_onehot).sum(-1)

    else:
      assert goal_onehot.ndim == state_features.ndim
      # [T, B]
      reward = (state_features * goal_onehot.astype(jnp.float32)).sum(-1)

    # [T, B, G]
    if distance_coeff > 0:
      distance_reward = compute_goal_distance_reward(timesteps)

      # [T, B]
      distance_penalty = (distance_reward * goal_onehot).sum(-1)

      reward = reward + distance_coeff * distance_penalty

    return reward

  def get_main_goal(timestep, expand_across_simulations: bool = True):
    """Tile the current task goal for on-task simulations.

    Args:
      timestep: single timestep (no batch/time dims), task_w is [G]
    Returns:
      goals: [num_ontask_simulations, G]
    """
    if not expand_across_simulations:
      return timestep.state.task_w
    goal = timestep.state.task_w[None]  # [G] -> [1, G]

    return jnp.tile(goal, (num_ontask_simulations, 1))  # [num_ontask_simulations, G]

  env = kwargs["model_env"]
  task_objects = kwargs.pop("task_objects")

  def place_goal_in_timestep(timestep, goal):
    """Create a new timestep with the given goal swapped in.

    Args:
      timestep: single timestep (no batch/time dims)
      goal: [G] goal vector
    Returns:
      new timestep with goal set in state and observation
    """
    task_object = (task_objects * goal).sum(-1)  # scalar
    task_object = task_object.astype(jnp.int32)
    new_state = timestep.state.replace(
      # step_num=jnp.zeros_like(timestep.state.step_num),
      task_w=goal.astype(timestep.state.task_w.dtype),
      task_object=task_object,
    )
    new_observation = env.make_observation(
      new_state, timestep.observation.prev_action_raw
    )

    return timestep.replace(
      state=new_state,
      observation=new_observation,
      # reward=jnp.zeros_like(timestep.reward),
      # discount=jnp.ones_like(timestep.discount),
      # step_type=jnp.ones_like(timestep.step_type),
    )

  def make_log_extras(timestep, goal=None):
    """Env-specific: extract goal reward, task vector, achievements from timestep."""
    state_features = timestep.observation.state_features.astype(jnp.float32)
    task_w = timestep.observation.task_w.astype(jnp.float32)
    goal_vec = goal if goal is not None else task_w
    goal_vec = jnp.broadcast_to(goal_vec, state_features.shape)
    goal_reward = (goal_vec * state_features).sum(-1)
    return {
      "goal_reward": goal_reward,
      "goal_task_vector": goal_vec,
      "goal_achievements": state_features,
    }

  return vbb.make_train(
    make_agent=partial(
      make_jaxmaze_multigoal_agent,
      model_env=kwargs.pop("model_env"),
    ),
    make_loss_fn_class=functools.partial(
      make_loss_fn_class,
      dyna_epsilon_values=dyna_epsilon_values,
      offtask_dyna_epsilon_values=offtask_dyna_epsilon_values,
      sample_preplay_goals=sample_nontask_visible_goals,
      sample_td_goals=sample_all_tasks,
      compute_rewards=compute_rewards,
      get_main_goal=get_main_goal,
      place_goal_in_timestep=place_goal_in_timestep,
      make_log_extras=make_log_extras,
    ),
    make_optimizer=base_agent.make_optimizer,
    make_actor=make_actor,
    **kwargs,
  )


##############################
# learner_log_extra (jaxmaze)
##############################
def jaxmaze_learner_log_extra(
  data: dict,
  config: dict,
  action_names: dict,
  render_fn: Callable,
  extract_task_info: Callable[[TimeStep], flax.struct.PyTreeNode] = lambda t: t,
  get_task_name: Callable = lambda t: "Task",
  image_keys: list = None,
  sim_idx: int = 0,
  all_tasks=None,
):
  from math import ceil

  # ---- Pre-index data outside callback (JAX land) ----
  callback_data = {}

  # Online: [B, T, ...] -> first batch -> [T, ...]
  if "online" in data:
    callback_data["online"] = jax.tree_util.tree_map(lambda x: x[0], data["online"])
    # Keep full batch for episode grid plot
    callback_data["online_all_batches"] = {
      "timesteps": data["online"]["timesteps"],
      "actions": data["online"]["actions"],
    }

  # All goals: [B, T, ...] -> first batch -> [T, ...]
  if "all_goals" in data:
    callback_data["all_goals"] = jax.tree_util.tree_map(
      lambda x: x[0], data["all_goals"]
    )

  if "all_goals_microwave" in data:
    callback_data["all_goals_microwave"] = jax.tree_util.tree_map(
      lambda x: x[0], data["all_goals_microwave"]
    )

  def task_object_to_name(task_object_id):
    return image_keys[int(task_object_id)]

  def task_w__to__object(task_objects, task_w):
    task_idx = jnp.argmax(task_w)
    object_idx = task_objects[task_idx]
    return image_keys[int(object_idx)]

  # Dyna: [B, T, sim_len, sims, ...] -> batch=0, T=0, all sim_len, sim=0/1
  if "dyna" in data:
    dyna = data["dyna"]
    dyna.pop("any_achievable", None)

    # first batch & time
    dyna = jax.tree_util.tree_map(lambda y: y[0, 0], dyna)

    DYNA_IDX = 0
    PREPLAY_IDX = 1
    simulation_goals = dyna.pop("goal")
    ontask_q_values = dyna.pop("sim_ontask_q_values")
    offtask_q_values = dyna.pop("sim_offtask_q_values")
    epsilon_values = dyna.pop("epsilon_values")

    callback_data["dyna"] = jax.tree_util.tree_map(lambda x: x[:, DYNA_IDX], dyna)
    callback_data["dyna"]["ontask_q_values"] = ontask_q_values[:, DYNA_IDX]
    callback_data["dyna"]["offtask_q_values"] = offtask_q_values[:, DYNA_IDX]
    callback_data["dyna"]["goal"] = simulation_goals[DYNA_IDX]
    callback_data["dyna"]["epsilon"] = epsilon_values[DYNA_IDX]

    callback_data["preplay"] = jax.tree_util.tree_map(lambda x: x[:, PREPLAY_IDX], dyna)
    callback_data["preplay"]["ontask_q_values"] = ontask_q_values[:, PREPLAY_IDX]
    callback_data["preplay"]["offtask_q_values"] = offtask_q_values[:, PREPLAY_IDX]
    callback_data["preplay"]["goal"] = simulation_goals[PREPLAY_IDX]
    callback_data["preplay"]["epsilon"] = epsilon_values[PREPLAY_IDX]

  def plot_q_heatmap(
    ax,
    q_values,
    actions,
    title,
    nT,
    action_names=None,
    show_numbers=True,
    vmin=None,
    vmax=None,
  ):
    """Draw Q-value heatmap [A, T] with argmax (red) and taken (white) rectangles."""
    from matplotlib.patches import Rectangle

    q_T = np.array(q_values[:nT]).T  # [A, T]
    im = ax.imshow(
      q_T, aspect="auto", cmap="coolwarm", interpolation="nearest", vmin=vmin, vmax=vmax
    )
    n_actions = q_T.shape[0]
    for t in range(nT):
      # Text annotations with 2 sig figs
      if show_numbers:
        for a in range(n_actions):
          ax.text(
            t,
            a,
            f"{q_T[a, t]:.1f}",
            ha="center",
            va="center",
            fontsize=10,
            color="black",
          )
      # Black rectangle: argmax action (slightly larger)
      best_a = int(np.argmax(q_values[t]))
      ax.add_patch(
        Rectangle(
          (t - 0.45, best_a - 0.45),
          0.9,
          0.9,
          linewidth=2,
          edgecolor="black",
          facecolor="none",
        )
      )
      # Cyan rectangle: taken action (slightly smaller)
      taken_a = int(actions[t])
      ax.add_patch(
        Rectangle(
          (t - 0.35, taken_a - 0.35),
          0.7,
          0.7,
          linewidth=3,
          edgecolor="cyan",
          facecolor="none",
        )
      )
    ax.set_title(title, fontsize=22)
    ax.set_yticks(range(q_T.shape[0]))
    if action_names:
      ax.set_yticklabels([action_names.get(i, str(i)) for i in range(q_T.shape[0])])
    ax.set_xticks(range(nT))
    return im

  def plot_unified(d):
    """Unified figure: up to 4 columns (online, all_goals, preplay, dyna) x 5 rows."""
    column_order = ["online", "all_goals", "preplay", "dyna"]
    col_names = [c for c in column_order if c in d]
    if not col_names:
      return None

    col_data = {c: d[c] for c in col_names}
    n_cols = len(col_names)
    n_rows = 5

    fig, axes = plt.subplots(
      n_rows,
      n_cols,
      figsize=(10.4 * n_cols, 5.2 * n_rows),
      gridspec_kw={"hspace": 0.4, "wspace": 0.3},
    )
    fig.set_dpi(150)
    if n_cols == 1:
      axes = axes[:, None]
    if n_rows == 1:
      axes = axes[None, :]

    # Set tick label size on all axes
    for ax_row in axes:
      for ax in ax_row:
        ax.tick_params(labelsize=18)

    # Detect episodes from online timesteps (for trajectory rendering)
    online_timesteps = None
    online_episode_ranges = []
    online_episode_starts = []
    if "online" in col_data:
      online_timesteps = col_data["online"]["timesteps"]
      nT_online = len(online_timesteps.reward)
      is_first = online_timesteps.first()
      online_episode_starts = list(jnp.where(is_first)[0])
      if not online_episode_starts or online_episode_starts[0] != 0:
        online_episode_starts = [0] + online_episode_starts
      for i, start in enumerate(online_episode_starts):
        end = (
          online_episode_starts[i + 1]
          if i + 1 < len(online_episode_starts)
          else nT_online
        )
        online_episode_ranges.append((int(start), int(end)))

    # --- Row 0: Q-values line plot ---
    for ci, col_name in enumerate(col_names):
      ax = axes[0, ci]
      cd = col_data[col_name]
      actions = cd["actions"]
      q_values = cd["q_values"]
      q_target = cd["q_target"]
      nT = len(actions)
      q_values_taken = rlax.batched_index(q_values, actions)

      if col_name in ("online", "all_goals"):
        loss_rewards = cd.get("loss_rewards")
        if loss_rewards is not None:
          ax.plot(loss_rewards, label="Loss Reward", color="tab:cyan")
        # Show microwave reward timesteps as dashed black vertical lines
        if col_name == "all_goals" and "all_goals_microwave" in d:
          mw_rewards = d["all_goals_microwave"].get("loss_rewards")
          if mw_rewards is not None:
            mw_ts = np.where(np.array(mw_rewards) > 0)[0]
            if len(mw_ts) > 0:
              ax.vlines(
                mw_ts,
                0,
                np.array(mw_rewards)[mw_ts],
                linestyles="--",
                color="black",
                label="Reward (microwave)",
              )
        ax.plot(q_values_taken, label="Q-Values", color="tab:orange")
        ax.plot(q_values.max(axis=-1), label="Q-Max", color="tab:green")
        ax.plot(q_target, label="Q-Targets", color="tab:red")
        tql = cd.get("target_q_t_lambda")
        if tql is not None:
          ax.plot(tql, label="Q-Target (λ)", color="blue")
        tqm = cd.get("target_q_t_model")
        if tqm is not None:
          ax.plot(tqm, label="Q-Target (model)", color="grey")
        qnd = cd.get("td_mask")
        if qnd is not None:
          ax.plot(qnd * 0.1, label="TD Mask", linestyle="--", color="purple")
      else:
        # preplay/dyna
        simulation_reward = cd.get("simulation_reward")
        if simulation_reward is not None:
          ax.plot(simulation_reward, label="Sim rewards", color="tab:blue")
        ax.plot(q_values_taken, label="Q-Values", color="tab:orange")
        ax.plot(q_target, label="Q-Targets", color="tab:red")
        loss_mask = cd.get("loss_mask")
        if loss_mask is not None:
          ax.plot(loss_mask * 0.5, label="Loss Mask", linestyle="--", color="black")

      ax.set_title(f"{col_name} — Q-Values", fontsize=22)
      ax.grid(True)
      ax.set_xticks(range(nT))

    # --- Row 1: Environment path image ---
    for ci, col_name in enumerate(col_names):
      ax = axes[1, ci]
      cd = col_data[col_name]

      if col_name == "online" and online_timesteps is not None:
        # Render episode 0 (first episode)
        if len(online_episode_ranges) >= 1:
          ep_start, ep_end = online_episode_ranges[0]
          maze_height, maze_width, _ = online_timesteps.state.grid[0].shape
          initial_state = jax.tree_util.tree_map(
            lambda x: x[ep_start], online_timesteps.state
          )
          img = render_fn(initial_state)
          ep_positions = jax.tree_util.tree_map(
            lambda x: x[ep_start : ep_end - 1], online_timesteps.state.agent_pos
          )
          ep_actions = col_data["online"]["actions"][ep_start : ep_end - 1]
          renderer.place_arrows_on_image(
            img, ep_positions, ep_actions, maze_height, maze_width, arrow_scale=5, ax=ax
          )
          ep_ts = jax.tree_util.tree_map(lambda x: x[ep_start], online_timesteps)
          goal_name = task_object_to_name(ep_ts.state.task_object)
          total_reward = float(online_timesteps.reward[ep_start:ep_end].sum())
          ax.set_title(
            f"online — Ep 0 ({goal_name}, r={total_reward:.1f})", fontsize=22
          )
          ax.axis("off")
        else:
          ax.set_visible(False)

      elif col_name == "all_goals":
        if "all_goals_microwave" in d:
          cd2 = d["all_goals_microwave"]
          actions2 = cd2["actions"]
          q_values2 = cd2["q_values"]
          q_target2 = cd2["q_target"]
          nT2 = len(actions2)
          q_values_taken2 = rlax.batched_index(q_values2, actions2)
          loss_rewards2 = cd2.get("loss_rewards")
          if loss_rewards2 is not None:
            reward_ts = np.where(np.array(loss_rewards2) > 0)[0]
            if len(reward_ts) > 0:
              ax.vlines(
                reward_ts,
                0,
                np.array(loss_rewards2)[reward_ts],
                label="Loss Reward",
                color="tab:cyan",
              )
          ax.plot(q_values_taken2, label="Q-Values", color="tab:orange")
          ax.plot(q_values2.max(axis=-1), label="Q-Max", color="tab:green")
          ax.plot(q_target2, label="Q-Targets", color="tab:red")
          tql2 = cd2.get("target_q_t_lambda")
          if tql2 is not None:
            ax.plot(tql2, label="Q-Target (λ)", color="darkblue")
          tqm2 = cd2.get("target_q_t_model")
          if tqm2 is not None:
            ax.plot(tqm2, label="Q-Target (model)", color="grey")
          qnd2 = cd2.get("td_mask")
          if qnd2 is not None:
            ax.plot(qnd2 * 0.1, label="TD Mask", linestyle="--", color="purple")
          ax.set_title("all_goals(microwave) — Q-Values", fontsize=22)
          ax.legend(fontsize=15)
          ax.grid(True)
          ax.set_xticks(range(nT2))
        else:
          ax.set_visible(False)

      elif col_name in ("preplay", "dyna"):
        # Simulation trajectory rendering
        timesteps = cd["timesteps"]
        actions = cd["actions"]
        goal = cd.get("goal")
        maze_height, maze_width, _ = timesteps.state.grid[0].shape
        initial_state = jax.tree_util.tree_map(lambda x: x[0], timesteps.state)
        img = render_fn(initial_state)
        in_episode = get_in_episode(timesteps)
        ep_actions = actions[in_episode][:-1]
        ep_positions = jax.tree_util.tree_map(
          lambda x: x[in_episode][:-1], timesteps.state.agent_pos
        )
        renderer.place_arrows_on_image(
          img, ep_positions, ep_actions, maze_height, maze_width, arrow_scale=5, ax=ax
        )
        initial_timestep = jax.tree_util.tree_map(lambda x: x[0], timesteps)
        t_goal_name = task_object_to_name(initial_timestep.state.task_object)
        if goal is not None:
          task_objects = initial_timestep.state.objects
          sim_goal_name = task_w__to__object(task_objects, goal)
          ax.set_title(
            f"{col_name} — Traj (sim={sim_goal_name}, t={t_goal_name})", fontsize=22
          )
        else:
          ax.set_title(f"{col_name} — Trajectory ({t_goal_name})", fontsize=22)
        ax.axis("off")
      else:
        ax.set_visible(False)

    # --- Shared color scale for Q-value rows ---
    all_q = []
    for col_name in col_names:
      cd = col_data[col_name]
      nT = len(cd["actions"])
      all_q.append(np.array(cd["q_values"][:nT]))
      target_q = cd.get("target_q_values")
      if target_q is not None:
        all_q.append(np.array(target_q[:nT]))
    all_q_flat = np.concatenate([q.ravel() for q in all_q])
    q_vmin, q_vmax = float(all_q_flat.min()), float(all_q_flat.max())

    # --- Row 2: Q-value heatmap [A, T] ---
    im_q = None
    for ci, col_name in enumerate(col_names):
      ax = axes[2, ci]
      cd = col_data[col_name]
      actions = cd["actions"]
      q_values = cd["q_values"]
      nT = len(actions)
      if col_name in ("online", "all_goals"):
        nT = min(nT, 20)
      im_q = plot_q_heatmap(
        ax,
        q_values,
        actions,
        f"{col_name} — Q-values",
        nT,
        action_names=action_names,
        show_numbers=col_name in ("preplay", "dyna"),
        vmin=q_vmin,
        vmax=q_vmax,
      )
    if im_q is not None:
      fig.colorbar(im_q, ax=axes[2, :].tolist(), shrink=0.8, pad=0.02)

    # --- Row 3: Target-net Q-value heatmap [A, T] ---
    im_tq = None
    for ci, col_name in enumerate(col_names):
      ax = axes[3, ci]
      cd = col_data[col_name]
      # For all_goals column: show microwave regular Q-values instead of stove target Q
      if col_name == "all_goals" and "all_goals_microwave" in d:
        cd2 = d["all_goals_microwave"]
        actions2 = cd2["actions"]
        q_values2 = cd2["q_values"]
        nT2 = min(len(actions2), 20)
        im_tq = plot_q_heatmap(
          ax,
          q_values2,
          actions2,
          "all_goals (microwave) — Q-values",
          nT2,
          action_names=action_names,
          show_numbers=False,
          vmin=q_vmin,
          vmax=q_vmax,
        )
      else:
        target_q = cd.get("target_q_values")
        if target_q is not None:
          actions = cd["actions"]
          nT = len(actions)
          if col_name in ("online", "all_goals"):
            nT = min(nT, 20)
          im_tq = plot_q_heatmap(
            ax,
            target_q,
            actions,
            f"{col_name} — Target Q",
            nT,
            action_names=action_names,
            show_numbers=col_name in ("preplay", "dyna"),
            vmin=q_vmin,
            vmax=q_vmax,
          )
        else:
          ax.set_visible(False)
    if im_tq is not None:
      fig.colorbar(im_tq, ax=axes[3, :].tolist(), shrink=0.8, pad=0.02)

    # --- Row 4: Goal vectors ---
    for ci, col_name in enumerate(col_names):
      ax = axes[4, ci]
      cd = col_data[col_name]
      nT = len(cd["actions"])

      if col_name in ("online", "all_goals"):
        goal_tv = cd.get("goal_task_vector")
        goal_ach = cd.get("goal_achievements")
        if goal_tv is not None and goal_ach is not None:
          combined = (goal_tv + goal_ach).T  # [D, T]
          ax.imshow(
            combined,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            vmin=0,
            vmax=2,
          )
          ax.set_title(f"{col_name} — Task+Ach", fontsize=22)
          ax.set_ylabel("Dims", fontsize=18)
          ax.set_yticks(range(combined.shape[0]))
          ax.set_xticks(range(nT))
          ax.grid(True, axis="x", alpha=0.3, color="white")
          loss_mask = cd.get("loss_mask")
          if loss_mask is not None:
            mask_diff = np.diff(np.concatenate([[0], loss_mask, [0]]))
            starts = np.where(mask_diff == 1)[0]
            ends = np.where(mask_diff == -1)[0] - 1
            for s in starts:
              ax.axvline(s - 0.5, color="red", linestyle="-", linewidth=1.5)
            for e in ends:
              ax.axvline(e + 0.5, color="red", linestyle="-", linewidth=1.5)
          # Draw white vertical lines at episode boundaries
          if col_name == "online" and online_episode_starts:
            for ep_start in online_episode_starts[1:]:
              ax.axvline(ep_start - 0.5, color="white", linestyle="-", linewidth=2)
        else:
          ax.set_visible(False)

      elif col_name in ("preplay", "dyna"):
        goal = cd.get("goal")
        if goal is not None:
          goal_2d = jnp.tile(goal[None, :], (nT, 1)).T  # [D, T]
          ax.imshow(
            goal_2d,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            vmin=0,
            vmax=1,
          )
          ax.set_title(f"{col_name} — Goal Vector", fontsize=22)
          ax.set_ylabel("Dims", fontsize=18)
          ax.set_yticks(range(goal_2d.shape[0]))
          ax.set_xticks(range(nT))
          ax.grid(True, axis="x", alpha=0.3, color="white")
        else:
          ax.set_visible(False)
      else:
        ax.set_visible(False)

    fig.tight_layout()

    return fig

  def plot_episode_grid(d):
    """Grid of all batch episodes showing maze + trajectory arrows."""
    if "online_all_batches" not in d:
      return None
    all_timesteps = d["online_all_batches"]["timesteps"]
    all_actions = d["online_all_batches"]["actions"]
    B = len(all_timesteps.reward)

    n_cols = ceil(B**0.5)
    n_rows = ceil(B / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
      axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
      axes = axes.reshape(n_rows, n_cols)

    for b in range(B):
      row, col = divmod(b, n_cols)
      ax = axes[row, col]

      ts_b = jax.tree_util.tree_map(lambda x: x[b], all_timesteps)
      actions_b = np.array(all_actions[b])
      nT = len(ts_b.reward)

      # Detect first episode
      is_first = np.array(ts_b.first())
      ep_starts = list(np.where(is_first)[0])
      if not ep_starts or ep_starts[0] != 0:
        ep_starts = [0] + ep_starts
      ep_start = ep_starts[0]
      ep_end = ep_starts[1] if len(ep_starts) > 1 else nT

      maze_height, maze_width, _ = ts_b.state.grid[0].shape
      initial_state = jax.tree_util.tree_map(lambda x: x[ep_start], ts_b.state)
      img = render_fn(initial_state)
      ep_positions = jax.tree_util.tree_map(
        lambda x: x[ep_start : ep_end - 1], ts_b.state.agent_pos
      )
      ep_actions = actions_b[ep_start : ep_end - 1]
      renderer.place_arrows_on_image(
        img, ep_positions, ep_actions, maze_height, maze_width, arrow_scale=5, ax=ax
      )
      goal_name = task_object_to_name(ts_b.state.task_object[ep_start])
      total_reward = float(np.array(ts_b.reward[ep_start:ep_end]).sum())
      ax.set_title(f"b{b} {goal_name} r={total_reward:.1f}", fontsize=9)
      ax.axis("off")

    # Hide unused axes
    for idx in range(B, n_rows * n_cols):
      row, col = divmod(idx, n_cols)
      axes[row, col].set_visible(False)

    fig.tight_layout()
    return fig

  def callback(d):
    unified_fig = plot_unified(d)
    grid_fig = plot_episode_grid(d)
    if wandb.run is not None:
      log_data = {}
      if unified_fig is not None:
        log_data["learner_example/unified"] = wandb.Image(unified_fig)
      if grid_fig is not None:
        log_data["learner_example/episode_grid"] = wandb.Image(grid_fig)
      if log_data:
        wandb.log(log_data)
    if unified_fig is not None:
      plt.close(unified_fig)
    if grid_fig is not None:
      plt.close(grid_fig)

  n_updates = data["n_updates"] + 1
  if config["LEARNER_EXTRA_LOG_PERIOD"] > 0:
    is_log_time = n_updates % config["LEARNER_EXTRA_LOG_PERIOD"] == 0
    jax.lax.cond(
      is_log_time,
      lambda d: jax.debug.callback(callback, d),
      lambda d: None,
      callback_data,
    )


##############################
# __main__ test block
##############################
if __name__ == "__main__":
  import os

  os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
  os.environ["JAX_DEBUG_NANS"] = "True"

  from jaxmaze.human_dyna import experiments as jaxmaze_experiments
  from jaxmaze.human_dyna import multitask_env_fast as multitask_env
  from jaxmaze import utils as jaxmaze_utils
  import jaxmaze_observer as humansf_observers

  config = {
    "SEED": 42,
    "NUM_ENVS": 4,
    "TOTAL_TIMESTEPS": 2000,
    "TRAINING_INTERVAL": 5,
    "BUFFER_SIZE": 1000,
    "BUFFER_BATCH_SIZE": 2,
    "TOTAL_BATCH_SIZE": 20,
    "LEARNING_STARTS": 50,
    "SAMPLE_LENGTH": 10,
    # Small network dims
    "EMBED_HIDDEN_DIM": 16,
    "MLP_HIDDEN_DIM": 32,
    "NUM_MLP_LAYERS": 0,
    "NUM_EMBED_LAYERS": 0,
    "AGENT_RNN_DIM": 32,
    "Q_HIDDEN_DIM": 64,
    "NUM_PRED_LAYERS": 1,
    "ACTIVATION": "leaky_relu",
    "QHEAD_TYPE": "duelling",
    "USE_BIAS": False,
    # Preplay params
    "SIMULATION_LENGTH": 3,
    "NUM_OFFTASK_SIMULATIONS": 1,
    "NUM_ONTASK_SIMULATIONS": 1,
    "NUM_OFFTASK_GOALS": 1,
    "ONLINE_COEFF": 1.0,
    "DYNA_COEFF": 1.0,
    "ALL_GOALS_COEFF": 1.0,
    "MAINQ_COEFF": 1e-2,
    "OFFTASK_COEFF": 2.0,
    # Optimizer
    "LR": 0.001,
    "MAX_GRAD_NORM": 80,
    "EPS_ADAM": 1e-5,
    "LR_LINEAR_DECAY": False,
    "GAMMA": 0.99,
    "TD_LAMBDA": 0.9,
    "ALL_GOALS_LAMBDA": 0.6,
    "ALL_GOALS_TD": "mb_all",
    "STEP_COST": 0.0,
    "TARGET_UPDATE_INTERVAL": 10,
    # Epsilon
    "FIXED_EPSILON": 0,
    "EPSILON_START": 1.0,
    "EPSILON_FINISH": 0.1,
    # Logging
    "LEARNER_LOG_PERIOD": 1,
    "LEARNER_EXTRA_LOG_PERIOD": 0,
    "EVAL_LOG_PERIOD": 1,
    "GRADIENT_LOG_PERIOD": 0,
    "MAX_EPISODE_LOG_LEN": 10,
    "EVAL_EPISODES": 1,
    "EVAL_STEPS": 10,
    "ALG": "preplay",
    "PROJECT": "test",
    "ENV_NAME": "jaxmaze",
    "ENV": "jaxmaze",
    "KNOWN_OFFTASK_GOAL": True,
    "FAST_ENV": True,
  }
  config["NUM_UPDATES"] = 1
  config["rlenv"] = {"ENV_KWARGS": {"exp": "exp4"}}

  # Load env
  exp = config["rlenv"]["ENV_KWARGS"].pop("exp")
  exp_fn = getattr(jaxmaze_experiments, exp, None)
  env_params, test_env_params, task_objects, idx2maze = exp_fn(config)

  task_runner = multitask_env.TaskRunner(task_objects=task_objects)
  env = multitask_env.HouseMaze(
    task_runner=task_runner,
    num_categories=200,
  )
  env = jaxmaze_utils.AutoResetWrapper(env)

  train_objects = env_params.reset_params.train_objects[0]
  test_objects = env_params.reset_params.test_objects[0]
  train_tasks = jnp.array([env.task_runner.task_vector(o) for o in train_objects])
  test_tasks = jnp.array([env.task_runner.task_vector(o) for o in test_objects])
  all_tasks_arr = jnp.concatenate((train_tasks, test_tasks), axis=0)

  train_fn = make_train_jaxmaze_multigoal(
    config=config,
    env=env,
    save_path="/tmp/multitask_preplay_test",
    train_env_params=env_params,
    test_env_params=test_env_params,
    ObserverCls=functools.partial(
      humansf_observers.TaskObserver,
      action_names={},
      extract_task_info=lambda t: t,
    ),
    initial_params=None,
    model_env=env,
    task_objects=task_objects,
    all_tasks=all_tasks_arr,
  )

  rng = jax.random.PRNGKey(config["SEED"])
  train_fn = jax.jit(train_fn)
  print("JIT compiling...")
  outs = train_fn(rng)
  print("Jaxmaze test passed!")

  ## ---- Craftax Multigoal Test ----
  # import craftax_simulation_configs
  # import craftax_observer
  # from jaxneurorl.wrappers import TimestepWrapper
  # from craftax_web_env import CraftaxMultiGoalSymbolicWebEnvNoAutoReset

  # static_env_params = (
  #  CraftaxMultiGoalSymbolicWebEnvNoAutoReset.default_static_params().replace(
  #    landmark_features=False
  #  )
  # )
  # craftax_base_env = CraftaxMultiGoalSymbolicWebEnvNoAutoReset(
  #  static_env_params=static_env_params
  # )
  # craftax_env = TimestepWrapper(LogWrapper(craftax_base_env), autoreset=True)

  # craftax_env_params = craftax_simulation_configs.default_params.replace(
  #  task_configs=craftax_simulation_configs.TRAIN_CONFIGS
  # )
  # craftax_test_env_params = craftax_simulation_configs.default_params.replace(
  #  task_configs=craftax_simulation_configs.TEST_CONFIGS
  # )

  # craftax_config = {
  #  "SEED": 42,
  #  "NUM_ENVS": 4,
  #  "TOTAL_TIMESTEPS": 2000,
  #  "TRAINING_INTERVAL": 5,
  #  "BUFFER_SIZE": 1000,
  #  "BUFFER_BATCH_SIZE": 2,
  #  "TOTAL_BATCH_SIZE": 20,
  #  "LEARNING_STARTS": 50,
  #  "SAMPLE_LENGTH": 10,
  #  # Small network dims
  #  "MLP_HIDDEN_DIM": 32,
  #  "NUM_MLP_LAYERS": 0,
  #  "AGENT_RNN_DIM": 32,
  #  "Q_HIDDEN_DIM": 64,
  #  "NUM_PRED_LAYERS": 1,
  #  "ACTIVATION": "leaky_relu",
  #  "QHEAD_TYPE": "duelling",
  #  "USE_BIAS": False,
  #  "OBS_INCLUDE_GOAL": False,
  #  # Preplay params
  #  "SIMULATION_LENGTH": 3,
  #  "NUM_OFFTASK_SIMULATIONS": 1,
  #  "NUM_ONTASK_SIMULATIONS": 1,
  #  "NUM_OFFTASK_GOALS": 1,
  #  "ONLINE_COEFF": 1.0,
  #  "DYNA_COEFF": 1.0,
  #  "ALL_GOALS_COEFF": 1.0,
  #  "MAINQ_COEFF": 1e-2,
  #  "SUBTASK_COEFF": 2.0,
  #  # Optimizer
  #  "LR": 0.001,
  #  "MAX_GRAD_NORM": 80,
  #  "EPS_ADAM": 1e-5,
  #  "LR_LINEAR_DECAY": False,
  #  "GAMMA": 0.99,
  #  "TD_LAMBDA": 0.9,
  #  "ALL_GOALS_LAMBDA": 0.6,
  #  "STEP_COST": 0.0,
  #  "TARGET_UPDATE_INTERVAL": 10,
  #  # Epsilon
  #  "FIXED_EPSILON": 2,
  #  "EPSILON_START": 1.0,
  #  "EPSILON_FINISH": 0.1,
  #  # Logging
  #  "LEARNER_LOG_PERIOD": 0,
  #  "LEARNER_EXTRA_LOG_PERIOD": 0,
  #  "EVAL_LOG_PERIOD": 0,
  #  "GRADIENT_LOG_PERIOD": 0,
  #  "MAX_EPISODE_LOG_LEN": 10,
  #  "EVAL_EPISODES": 1,
  #  "EVAL_STEPS": 10,
  #  "ALG": "preplay",
  #  "PROJECT": "test",
  #  "ENV_NAME": "craftax",
  #  "ENV": "craftax",
  #  "NUM_UPDATES": 1,
  # }

  # from jaxneurorl import loggers as jnrl_loggers

  # def noop_experience_logger(*args, **kwargs):
  #  pass

  # def craftax_make_logger(config, env, env_params, learner_log_extra=None):
  #  return jnrl_loggers.Logger(
  #    gradient_logger=jnrl_loggers.default_gradient_logger,
  #    learner_logger=jnrl_loggers.default_learner_logger,
  #    experience_logger=noop_experience_logger,
  #  )

  # craftax_train_fn = make_train_craftax_multigoal(
  #  config=craftax_config,
  #  env=craftax_env,
  #  save_path="/tmp/multitask_preplay_craftax_test",
  #  model_env=craftax_env,
  #  train_env_params=craftax_env_params,
  #  test_env_params=craftax_test_env_params,
  #  ObserverCls=craftax_observer.Observer,
  #  make_logger=craftax_make_logger,
  # )

  # rng = jax.random.PRNGKey(craftax_config["SEED"])
  # craftax_train_fn = jax.jit(craftax_train_fn)
  # print("JIT compiling Craftax multigoal test...")
  # outs = craftax_train_fn(rng)
  # print("Craftax multigoal test passed!")
