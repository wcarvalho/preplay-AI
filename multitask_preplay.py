"""
Simplified multitask preplay.

Simplification of multitask_preplay_craftax_v2.py:
- Removed dead code branches (rolling_window, combine_real_sim, backtracking=False,
  terminate_offtask, beta logic, cell_type=="none", SIM_EPSILON_SETTING 2/3)
- Avoided redundant RNN computation: extract rnn_out from simulation states
  instead of re-running apply_rnn_and_q
- Standalone DynaLossFn (not inheriting RecurrentLossFn)
- Renamed subtask_coeff -> offtask_coeff, reg_q_fn -> main_task_q_fn,
  subtask_q_fn -> off_task_q_fn
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

from jaxneurorl import losses
from jaxneurorl.agents.basics import TimeStep
from jaxneurorl.agents import value_based_basics as vbb
from jaxneurorl.agents import qlearning as base_agent
from networks import MLP, CraftaxObsEncoder, CraftaxMultiGoalObsEncoder
from networks import CategoricalJaxmazeObsEncoder

from visualizer import plot_frames
from jaxmaze import renderer


def info(x):
  return jax.tree_util.tree_map(lambda y: (y.shape, y.dtype), x)


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
  return jnp.concatenate((initial_mask[None], loss_mask_t), axis=0)


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


def rnn_output_from_state(state):
  """Extract RNN output from LSTM state. state=(carry, hidden) -> hidden."""
  return state[1]


##############################
# Simulation
##############################
def simulate_n_trajectories(
  h_tm1: RnnState,
  x_t: TimeStep,
  main_task_goal: jax.Array,
  rng: jax.random.PRNGKey,
  network: nn.Module,
  params: Params,
  policy_fn: Callable = None,
  num_steps: int = 5,
  num_simulations: int = 5,
  mainq_coeff: float = 1.0,
  offtask_coeff: float = 1.0,
  offtask_goal: jax.Array = None,
  make_init_goal_timestep=None,
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
    make_init_goal_timestep: optional fn to modify timestep for goal
  """

  if offtask_goal is not None:
    # Preplay: blend both Q-heads
    def get_q_vals(lstm_out, main_w, off_w):
      main_q = network.apply(params, lstm_out, main_w, method=network.main_task_q_fn)
      offtask_q = network.apply(params, lstm_out, off_w, method=network.off_task_q_fn)
      return mainq_coeff * main_q + offtask_coeff * offtask_q

    goal_for_timestep = offtask_goal
  else:
    # Dyna-only: use only main-task Q-head
    offtask_goal = main_task_goal  # dummy so arrays have matching shape
    goal_for_timestep = main_task_goal

    def get_q_vals(lstm_out, main_w, off_w):
      del off_w
      return network.apply(params, lstm_out, main_w, method=network.main_task_q_fn)

  def initial_predictions(x, prior_h, main_w, off_w, rng_):
    lstm_state, lstm_out = network.apply(
      params, prior_h, x, rng_, method=network.apply_rnn
    )
    preds = Predictions(q_vals=get_q_vals(lstm_out, main_w, off_w), state=lstm_state)
    if make_init_goal_timestep is not None:
      x = make_init_goal_timestep(x, off_w)
    return x, lstm_state, preds

  rng, rng_ = jax.random.split(rng)

  x_t, h_t, preds_t = jax.vmap(
    initial_predictions, in_axes=(None, None, 0, 0, 0), out_axes=0
  )(
    x_t,
    h_tm1,
    main_task_goal,
    goal_for_timestep,
    jax.random.split(rng_, num_simulations),
  )
  a_t = policy_fn(preds_t, rng_)

  def _single_model_step(carry, inputs):
    del inputs
    (timestep, lstm_state, a, rng) = carry

    rng, rng_ = jax.random.split(rng)
    next_timestep = network.apply(params, timestep, a, rng_, method=network.apply_model)

    next_lstm_state, next_rnn_out = network.apply(
      params, lstm_state, next_timestep, rng_, method=network.apply_rnn
    )

    next_preds = Predictions(
      q_vals=jax.vmap(get_q_vals)(next_rnn_out, main_task_goal, goal_for_timestep),
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
  offtask_coeff: float = 1.0
  mainq_coeff: float = 1.0

  augment_dyna_reward: Callable = None
  dyna_epsilon_values: Callable = None
  offtask_dyna_epsilon_values: Callable = None
  sample_preplay_goals: Callable = None
  sample_td_goals: Callable = None
  compute_rewards: Callable = None
  get_main_goal: Callable = None
  make_init_goal_timestep: Callable = None
  make_log_extras: Callable = None

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

    importance_weights = (1.0 / (batch.probabilities + 1e-6)).astype(jnp.float32)
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

    batch_td_error_fn = jax.vmap(losses.q_learning_lambda_td, in_axes=1, out_axes=1)
    selector_actions = jnp.argmax(online_preds.q_vals, axis=-1)

    q_t, target_q_t = batch_td_error_fn(
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
    batch_loss_mean = (batch_loss * loss_mask[:-1]).sum(0) / loss_mask[:-1].sum(0)

    metrics = {
      "0.q_loss": batch_loss.mean(),
      "0.q_td": jnp.abs(batch_td_error).mean(),
      "1.reward": rewards[1:].mean(),
      "z.q_mean": online_preds.q_vals.mean(),
      "z.q_var": online_preds.q_vals.var(),
    }

    log_info = {
      "timesteps": timestep,
      "actions": actions,
      "td_errors": batch_td_error,
      "loss_mask": loss_mask,
      "q_values": online_preds.q_vals,
      "q_loss": batch_loss,
      "q_target": target_q_t,
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
      td_error, batch_loss, metrics, log_info = self.loss_fn(
        timestep=data.timestep,
        online_preds=online_preds,
        target_preds=target_preds,
        actions=data.action,
        rewards=data.reward,
        is_last=make_float(data.timestep.last()),
        non_terminal=data.timestep.discount,
        loss_mask=loss_mask,
      )
      all_metrics.update({f"0.online{k}": v for k, v in metrics.items()})
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

      def all_goals_loss_single(
        timestep_b, action_b, h_on_b, h_tar_b, loss_mask_b, key
      ):
        """Off-task Q-learning loss on real trajectory for a single batch element.

        Samples N random goals and applies off_task_q_fn to existing RNN outputs
        (no simulation or RNN re-run needed).

        Args:
          timestep_b: [T, ...] timestep sequence for one batch element
          action_b: [T] actions
          h_on_b: [T, ...] online RNN states
          h_tar_b: [T, ...] target RNN states
          loss_mask_b: [T] loss mask
          key: random key
        """
        N = self.num_offtask_goals
        goals, _, _ = self.sample_td_goals(timestep_b, key, N)

        # Extract rnn_out from online/target states (no RNN re-run)
        rnn_out_online = rnn_output_from_state(h_on_b)  # [T, D]
        rnn_out_target = rnn_output_from_state(h_tar_b)  # [T, D]

        # Broadcast to [T, N, ...]
        rnn_out_online_rep = jnp.broadcast_to(
          rnn_out_online[:, None, :],
          (rnn_out_online.shape[0], N, rnn_out_online.shape[1]),
        )
        rnn_out_target_rep = jnp.broadcast_to(
          rnn_out_target[:, None, :],
          (rnn_out_target.shape[0], N, rnn_out_target.shape[1]),
        )
        action_repeated = jnp.broadcast_to(action_b[:, None], (action_b.shape[0], N))
        loss_mask_repeated = jnp.broadcast_to(
          loss_mask_b[:, None], (loss_mask_b.shape[0], N)
        )
        timestep_repeated = jax.tree_util.tree_map(
          lambda x: jnp.broadcast_to(x[:, None, ...], (x.shape[0], N) + x.shape[1:]),
          timestep_b,
        )

        # Apply off_task_q_fn to rnn outputs for each goal [T, N, A]
        def apply_q(params_, rnn_out_rep, q_method):
          """Apply Q-head to broadcast rnn outputs for each goal.

          Args:
            params_: network parameters
            rnn_out_rep: [T, N, D] broadcast RNN outputs
            q_method: network method (off_task_q_fn or main_task_q_fn)

          Returns:
            [T, N, A] Q-values
          """
          q_fn = lambda h, g: self.network.apply(params_, h, g, method=q_method)
          return jax.vmap(jax.vmap(q_fn, (0, 0)), (0, None))(rnn_out_rep, goals)

        online_q = apply_q(params, rnn_out_online_rep, self.network.off_task_q_fn)
        target_q = apply_q(
          target_params, rnn_out_target_rep, self.network.off_task_q_fn
        )

        online_preds_g = Predictions(q_vals=online_q, state=None)
        target_preds_g = Predictions(q_vals=target_q, state=None)

        goal_rewards = self.compute_rewards(timestep_repeated, goals)

        ag_td_error, ag_batch_loss, ag_metrics, ag_log_info = self.loss_fn(
          timestep=timestep_repeated,
          online_preds=online_preds_g,
          target_preds=target_preds_g,
          actions=action_repeated,
          rewards=goal_rewards,
          is_last=make_float(timestep_repeated.last()),
          non_terminal=timestep_repeated.discount,
          loss_mask=loss_mask_repeated,
          lambda_override=self.all_goals_lambda,
        )
        if self.make_log_extras is not None:
          extras = self.make_log_extras(timestep_repeated, goal=goals)
          ag_log_info.update(extras)
        ag_log_info["sampled_goal"] = goals  # [N, G]
        return ag_td_error, ag_batch_loss, ag_metrics, ag_log_info

      # Pass per-timestep RNN states (not initial state)
      # online_preds.state has shape [T, B, ...], target_preds.state likewise
      all_goals_fn = jax.vmap(all_goals_loss_single, (1, 1, 1, 1, 1, 0), 0)
      _, all_goals_batch_loss, all_goals_metrics, all_goals_log_info = all_goals_fn(
        data.timestep,
        data.action,
        online_preds.state,
        target_preds.state,
        loss_mask,
        jax.random.split(key_grad, B),
      )
      # Take first goal: [B, T, N, ...] -> [B, T, ...]
      all_log_info["all_goals"] = jax.tree_util.tree_map(
        lambda x: x[:, :, 0], all_goals_log_info
      )
      batch_loss += self.all_goals_coeff * all_goals_batch_loss.mean(1)
      all_metrics.update({f"2.all_goals/{k}": v for k, v in all_goals_metrics.items()})

    # ---- Dyna/preplay loss ----
    if self.dyna_coeff > 0.0:
      remove_last = lambda x: jax.tree_util.tree_map(lambda y: y[:-1], x)
      h_tm1_online = concat_first_rest(online_state, remove_last(online_preds.state))
      h_tm1_target = concat_first_rest(target_state, remove_last(target_preds.state))
      x_t = data.timestep

      dyna_loss_fn = functools.partial(
        self.preplay_loss_fn, params=params, target_params=target_params
      )
      dyna_loss_fn = jax.vmap(dyna_loss_fn, (1, 1, 1, 1, 1, 0), 0)
      _, dyna_batch_loss, dyna_metrics, dyna_log_info = dyna_loss_fn(
        x_t,
        data.action,
        h_tm1_online,
        h_tm1_target,
        loss_mask,
        jax.random.split(key_grad, B),
      )
      batch_loss += self.dyna_coeff * dyna_batch_loss
      all_metrics.update({f"1.preplay/{k}": v for k, v in dyna_metrics.items()})
      all_log_info["dyna"] = dyna_log_info

    if self.logger.learner_log_extra is not None:
      self.logger.learner_log_extra(all_log_info)

    return td_error, batch_loss, all_metrics

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
      mainq_coeff=self.mainq_coeff,
      offtask_coeff=self.offtask_coeff,
      make_init_goal_timestep=self.make_init_goal_timestep,
    )

    def apply_q_head(rnn_out, p, q_method, task):
      """Apply Q-head to rnn outputs. rnn_out: [T, N, D], task: [N, G] -> [T, N, A]"""

      def at_t(h_t):
        return jax.vmap(lambda h, g: self.network.apply(p, h, g, method=q_method))(
          h_t, task
        )

      return jax.vmap(at_t)(rnn_out)

    def unroll_target_rnn(x_t, h_tar_t, all_t, num_sims, rng):
      """Unroll target RNN on simulated timesteps (vmap-over-scan).

      Uses original x_t at step 0 (matching what the online RNN saw inside
      simulate_n_trajectories, before make_init_goal_timestep), then the
      model-generated timesteps from all_t[1:] for subsequent steps.

      Args:
        x_t: original timestep (no sim dim)
        h_tar_t: target RNN state (no sim dim)
        all_t: simulated timesteps [S+1, num_sims, ...]
        num_sims: number of simulations
        rng: random key for RNN stochasticity
      Returns:
        target_rnn_out: [S+1, num_sims, D]
      """
      # Build input: original x_t at step 0, model-generated for the rest
      x_t_broadcast = jax.tree_util.tree_map(
        lambda y: jnp.broadcast_to(y, (num_sims,) + y.shape), x_t
      )
      target_input = jax.tree_util.tree_map(
        lambda x0, xrest: jnp.concatenate((x0[None], xrest), axis=0),
        x_t_broadcast,
        jax.tree_util.tree_map(lambda y: y[1:], all_t),
      )  # [S+1, num_sims, ...]

      h_tar_broadcast = jax.tree_util.tree_map(
        lambda y: jnp.broadcast_to(y, (num_sims,) + y.shape), h_tar_t
      )

      def single_sim_unroll(h_init, xs, sim_rng):
        """Unroll target RNN for a single simulation."""

        def step(carry, x_step):
          h, rng_ = carry
          rng_, step_rng = jax.random.split(rng_)
          new_h, rnn_out = self.network.apply(
            target_params,
            h,
            x_step,
            step_rng,
            method=self.network.apply_rnn,
          )
          return (new_h, rng_), rnn_out

        _, rnn_outs = jax.lax.scan(step, (h_init, sim_rng), xs)
        return rnn_outs  # [S+1, D]

      # vmap over sim dimension: input [num_sims, S+1, ...] -> [num_sims, S+1, D]
      # Swap axes: [S+1, num_sims, ...] -> [num_sims, S+1, ...]
      target_input_t = jax.tree_util.tree_map(
        lambda y: jnp.swapaxes(y, 0, 1), target_input
      )
      sim_rngs = jax.random.split(rng, num_sims)
      target_rnn_out = jax.vmap(single_sim_unroll)(
        h_tar_broadcast, target_input_t, sim_rngs
      )  # [num_sims, S+1, D]
      return jnp.swapaxes(target_rnn_out, 0, 1)  # [S+1, num_sims, D]

    def preplay_at_t(x_t, h_on_t, h_tar_t, l_mask_t, key):
      """Combined off-task + on-task loss at a single timestep.

      Merges on-task and off-task simulations into a single simulate call
      with total_sims = N_on + G_off * N_off.

      Args:
        x_t: single timestep (no time dim)
        h_on_t: online RNN state at this timestep
        h_tar_t: target RNN state at this timestep
        l_mask_t: scalar loss mask
        key: random key
      """
      N_on = self.num_ontask_simulations
      N_off = self.num_offtask_simulations
      G_off = self.num_offtask_goals
      total_sims = N_on + G_off * N_off

      # Sample off-task goals
      goals, achievable, any_achievable = self.sample_preplay_goals(x_t, key, G_off)
      main_goal = self.get_main_goal(x_t)  # [N_on, G]

      # Build unified goals: [total_sims, G]
      offtask_goals = jnp.repeat(goals, N_off, axis=0)  # [G_off*N_off, G]
      all_goals = jnp.concatenate((main_goal, offtask_goals), axis=0)

      # Build unified epsilon vector
      key, rng_on, rng_off = jax.random.split(key, 3)
      all_eps = jnp.concatenate(
        (
          self.dyna_epsilon_values(rng_on),
          jnp.tile(self.offtask_dyna_epsilon_values(rng_off), G_off),
        )
      )  # [total_sims]

      def sim_policy(preds, rng):
        q_vals = preds.q_vals  # [total_sims, A]
        rngs = jax.random.split(rng, total_sims)
        return jax.vmap(base_agent.epsilon_greedy_act)(q_vals, all_eps, rngs)

      # all_t: [sim_length+1, total_sims, ...]
      # sim_out.actions: [sim_length+1, total_sims]
      key, key_ = jax.random.split(key)
      all_t, sim_out = simulate(
        h_tm1=h_on_t,
        x_t=x_t,
        rng=key_,
        main_task_goal=jnp.tile(main_goal[0:1], (total_sims, 1)),
        offtask_goal=all_goals,
        num_simulations=total_sims,
        policy_fn=sim_policy,
      )
      # all_a: [sim_length+1, total_sims]
      all_a = sim_out.actions
      # online_rnn_out: [sim_length+1, total_sims, D]
      online_rnn_out = rnn_output_from_state(sim_out.predictions.state)
      # target_rnn_out: [sim_length+1, total_sims, D]
      key, key_tar = jax.random.split(key)
      target_rnn_out = unroll_target_rnn(x_t, h_tar_t, all_t, total_sims, key_tar)

      # all_mask: [sim_length+1, total_sims]
      init_mask = jnp.broadcast_to(l_mask_t, (total_sims,))
      all_mask = simulation_finished_mask(init_mask, all_t)

      # === MAIN-TASK LOSS on ALL sims ===
      # all_main_goal: [total_sims, G]
      all_main_goal = jnp.tile(main_goal, (total_sims, 1))
      # all_main_q_on: [sim_length+1, total_sims, A]
      all_main_q_on = apply_q_head(
        online_rnn_out, params, self.network.main_task_q_fn, all_main_goal
      )
      # all_main_q_tar: [sim_length+1, total_sims, A]
      all_main_q_tar = apply_q_head(
        target_rnn_out, target_params, self.network.main_task_q_fn, all_main_goal
      )
      all_main_td, all_main_loss, main_metrics, main_log = self.loss_fn(
        all_t,
        Predictions(all_main_q_on, None),
        Predictions(all_main_q_tar, None),
        all_a,
        all_t.reward,
        make_float(all_t.last()),
        all_t.discount,
        all_mask,
      )

      # === OFF-TASK GOAL LOSS (split: different goals + different rewards) ===
      # off_*: [sim_length+1, G_off*N_off, ...]
      off_t = jax.tree_util.tree_map(lambda y: y[:, N_on:], all_t)
      off_online_rnn = online_rnn_out[:, N_on:]
      off_target_rnn = target_rnn_out[:, N_on:]
      off_a = all_a[:, N_on:]
      off_mask = all_mask[:, N_on:]
      # off_g: [G_off*N_off, G]
      off_g = all_goals[N_on:]

      # offtask_reward: [sim_length+1, G_off*N_off]
      offtask_reward = self.compute_rewards(off_t, off_g)
      # off_q_on: [sim_length+1, G_off*N_off, A]
      off_q_on = apply_q_head(off_online_rnn, params, self.network.off_task_q_fn, off_g)
      # off_q_tar: [sim_length+1, G_off*N_off, A]
      off_q_tar = apply_q_head(
        off_target_rnn, target_params, self.network.off_task_q_fn, off_g
      )
      off_td, off_loss, off_sub_metrics, off_log = self.loss_fn(
        off_t,
        Predictions(off_q_on, None),
        Predictions(off_q_tar, None),
        off_a,
        offtask_reward,
        make_float(off_t.last()),
        off_t.discount,
        off_mask,
      )

      # Scale off-task goal loss by achievability
      any_achievable_f = any_achievable.astype(jnp.float32)
      off_td = off_td * any_achievable_f
      off_loss = off_loss * any_achievable_f

      # === Combine ===
      denominator = G_off * N_off + self.ontask_loss_coeff * N_on

      batch_td_error = (
        self.offtask_loss_coeff * jnp.abs(off_td).sum(axis=1)
        + self.ontask_loss_coeff * jnp.abs(all_main_td).sum(axis=1)
      ) / denominator

      batch_loss_mean = (
        self.offtask_loss_coeff * off_loss.sum()
        + self.ontask_loss_coeff * all_main_loss.sum()
      ) / denominator

      # Metrics
      achieve_poss = achievable + 1e-5
      achieve_poss = achieve_poss / achieve_poss.sum()
      entropy = -jnp.sum(
        achieve_poss * jnp.log(achieve_poss + 1e-5), axis=-1
      ) / jnp.log(achieve_poss.shape[-1])
      metrics = {
        **{f"{k}/offtask-goal": v for k, v in off_sub_metrics.items()},
        **{f"{k}/main-q": v for k, v in main_metrics.items()},
        "2.achievable_entropy": entropy,
        "2.any_achievable": any_achievable_f,
      }
      log_info = off_log
      log_info["main_q_values"] = main_log["q_values"]
      log_info["main_q_target"] = main_log["q_target"]
      log_info["goal"] = off_g
      log_info["any_achievable"] = any_achievable_f
      log_info["offtask_reward"] = offtask_reward

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
        main_task_goal=main_goal,
        offtask_goal=None,
        num_simulations=num_sims,
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
      target_rnn_out = unroll_target_rnn(x_t, h_tar_t, all_t, num_sims, key_tar)

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
        timestep=all_t,
        online_preds=online_preds,
        target_preds=target_preds,
        actions=all_a,
        rewards=reward,
        is_last=make_float(all_t.last()),
        non_terminal=all_t.discount,
        loss_mask=all_t_mask,
      )
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
    step_cost=config.get("STEP_COST", 0.001),
    offtask_loss_coeff=config.get("OFFTASK_LOSS_COEFF", 1.0),
    ontask_loss_coeff=config.get("ONTASK_LOSS_COEFF", 1.0),
    num_offtask_goals=config.get("NUM_OFFTASK_GOALS", 5),
    offtask_coeff=config.get("OFFTASK_COEFF", 2.0),
    mainq_coeff=config.get("MAINQ_COEFF", 1.0),
    all_goals_coeff=config.get("ALL_GOALS_COEFF", 1.0),
    all_goals_lambda=config.get("ALL_GOALS_LAMBDA", 0.6),
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
    task = self.task_fn(task)
    return self.main_q_head(rnn_out, task)

  def off_task_q_fn(self, rnn_out, task):
    task = self.task_fn(task)
    return self.off_task_q_head(rnn_out, task)

  def unroll(
    self, rnn_state, xs: TimeStep, rng: jax.random.PRNGKey
  ) -> Tuple[Predictions, RnnState]:
    embedding = jax.vmap(self.observation_encoder)(xs.observation)
    rnn_in = vbb.RNNInput(obs=embedding, reset=xs.first())
    rng, _rng = jax.random.split(rng)
    new_rnn_state, new_rnn_states = self.rnn.unroll(rnn_state, rnn_in, _rng)

    task = jnp.zeros_like(xs.observation.achievements)
    rnn_out = self.rnn.output_from_state(new_rnn_states)
    q_vals = nn.BatchApply(self.main_task_q_fn)(rnn_out, task)
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
  off_task_q_head: nn.Module  # ignored when shared
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
    task = self.task_fn(task)
    return self.main_q_head(rnn_out, task)

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
    q_vals = nn.BatchApply(self.main_task_q_fn)(rnn_out, dummy_task)
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
    task = x.state.task_w
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
    task = x.observation.task_w
    q_vals = self.main_task_q_fn(rnn_out, task)
    predictions = Predictions(q_vals=q_vals, state=new_rnn_state)
    return predictions, new_rnn_state

  def main_task_q_fn(self, rnn_out, task):
    task = self.task_fn(task)
    return self.main_q_head(rnn_out, task)

  def off_task_q_fn(self, rnn_out, task):
    return self.main_task_q_fn(rnn_out, task)

  def unroll(
    self, rnn_state, xs: TimeStep, rng: jax.random.PRNGKey
  ) -> Tuple[Predictions, RnnState]:
    embedding = jax.vmap(self.observation_encoder)(xs.observation)
    rnn_in = vbb.RNNInput(obs=embedding, reset=xs.first())
    rng, _rng = jax.random.split(rng)
    new_rnn_state, new_rnn_states = self.rnn.unroll(rnn_state, rnn_in, _rng)

    task = xs.observation.task_w
    rnn_out = self.rnn.output_from_state(new_rnn_states)
    q_vals = nn.BatchApply(self.main_task_q_fn)(rnn_out, task)
    predictions = Predictions(q_vals=q_vals, state=new_rnn_states)
    return predictions, new_rnn_state

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
    off_task_q_head=QFnCls(
      hidden_dim=config.get("Q_HIDDEN_DIM", 512),
      num_layers=config.get("NUM_PRED_LAYERS", 0),
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

  qhead_type = config.get("QHEAD_TYPE", "dot")
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
  # ACME default epsilon schedule
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

  def sample_random_goals(timestep, key, num_offtask_goals):
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
      key_, shape=(num_offtask_goals,), minval=0, maxval=num_classes
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

  def sample_random_goals(timestep, key, num_offtask_goals):
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
      key_, shape=(num_offtask_goals,), minval=0, maxval=num_classes
    )  # [N]
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

  def sample_nontask_random_goals(timestep, key, num_offtask_goals):
    """Sample random goals excluding the current task goal.

    Args:
      timestep: single timestep (no batch/time dims), task_w is [1, G]
      key: random key
      num_offtask_goals: number of goals to sample (N)
    Returns:
      goals: [N, G] one-hot goal vectors
      achievable: [G] uniform achievability
      any_achievable: scalar bool
    """
    current_goal = timestep.observation.task_w[0]  # [G]
    other_goals = 1.0 - current_goal  # [G]
    other_goal_probs = other_goals / other_goals.sum()  # [G]
    achievable = jnp.ones_like(other_goal_probs)  # [G]
    any_achievable = achievable.sum() > 0.1
    key, key_ = jax.random.split(key)
    goals = distrax.Categorical(probs=other_goal_probs).sample(
      seed=key_, sample_shape=(num_offtask_goals)
    )  # [N]
    num_goals = timestep.observation.task_w.shape[-1]
    goals = jax.nn.one_hot(goals, num_classes=num_goals)  # [N, G]
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

  def make_init_goal_timestep(timestep, goal):
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

  return vbb.make_train(
    make_agent=partial(make_craftax_multigoal_agent, model_env=kwargs.pop("model_env")),
    make_loss_fn_class=functools.partial(
      make_loss_fn_class,
      dyna_epsilon_values=dyna_epsilon_values,
      offtask_dyna_epsilon_values=offtask_dyna_epsilon_values,
      sample_preplay_goals=sample_nontask_visible_goals,
      sample_td_goals=sample_nontask_random_goals,
      compute_rewards=compute_rewards,
      get_main_goal=get_main_goal,
      make_init_goal_timestep=make_init_goal_timestep,
    ),
    make_optimizer=base_agent.make_optimizer,
    make_actor=make_actor,
    **kwargs,
  )


def make_train_jaxmaze_multigoal(**kwargs):
  config = kwargs["config"]
  all_tasks = jnp.array(kwargs.pop("all_tasks"))
  known_offtask_goal = config.get("KNOWN_OFFTASK_GOAL", True)
  num_offtask_goals = 1 if known_offtask_goal else config["NUM_OFFTASK_GOALS"]
  config["NUM_OFFTASK_GOALS"] = num_offtask_goals

  vals = np.logspace(num=256, start=1, stop=3, base=0.1)

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

    is_object_nearby = timestep.observation.nearby_objects.astype(jnp.int32)  # [G]
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

  def sample_nontask_random_goals(timestep, key, num_offtask_goals):
    """Sample random off-task goals (known or from all_tasks pool).

    Args:
      timestep: single timestep (no batch/time dims)
      key: random key
      num_offtask_goals: number of goals to sample (N)
    Returns:
      goals: [N, G] goal vectors
      achievable: [N] uniform achievability
      any_achievable: scalar bool
    """
    if known_offtask_goal:
      goals = timestep.state.offtask_w[0][None]  # [1, G]
    else:
      key, key_ = jax.random.split(key)
      idxs = jax.random.choice(
        key_, len(all_tasks), shape=(num_offtask_goals,), replace=False
      )  # [N]
      index_into_tasks = jax.vmap(
        lambda i: jax.lax.dynamic_index_in_dim(all_tasks, i, keepdims=False)
      )
      goals = index_into_tasks(idxs)  # [N, G]
      goals = goals.astype(timestep.state.offtask_w.dtype)
    achievable = jnp.ones(len(goals))  # [N]
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
    state_features = timesteps.state.task_state.features.astype(
      jnp.float32
    )  # [T, num_sims, G]
    # goal_onehot[None]: [1, num_sims, G] broadcasts with [T, num_sims, G]
    offtask_reward = (state_features * goal_onehot.astype(jnp.float32)[None]).sum(
      -1
    )  # [T, num_sims]

    return offtask_reward

  def get_main_goal(timestep):
    """Tile the current task goal for on-task simulations.

    Args:
      timestep: single timestep (no batch/time dims), task_w is [G]
    Returns:
      goals: [num_ontask_simulations, G]
    """
    goal = timestep.state.task_w[None]  # [G] -> [1, G]

    return jnp.tile(goal, (num_ontask_simulations, 1))  # [num_ontask_simulations, G]

  env = kwargs["model_env"]
  task_objects = kwargs.pop("task_objects")

  def make_init_goal_timestep(timestep, goal):
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
      step_num=jnp.zeros_like(timestep.state.step_num),
      task_w=goal.astype(timestep.state.task_w.dtype),
      task_object=task_object,
    )
    new_observation = env.make_observation(new_state, timestep.observation.prev_action)

    return timestep.replace(
      state=new_state,
      observation=new_observation,
      reward=jnp.zeros_like(timestep.reward),
      discount=jnp.ones_like(timestep.discount),
      step_type=jnp.ones_like(timestep.step_type),
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
      sample_td_goals=sample_nontask_random_goals,
      compute_rewards=compute_rewards,
      get_main_goal=get_main_goal,
      make_init_goal_timestep=make_init_goal_timestep,
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
  sim_idx: int = 0,
):
  from math import ceil

  # ---- Pre-index data outside callback (JAX land) ----
  callback_data = {}

  # Online: [B, T, ...] -> first batch -> [T, ...]
  if "online" in data:
    callback_data["online"] = jax.tree_util.tree_map(lambda x: x[0], data["online"])

  # All goals: [B, T, ...] -> first batch -> [T, ...]
  if "all_goals" in data:
    callback_data["all_goals"] = jax.tree_util.tree_map(
      lambda x: x[0], data["all_goals"]
    )

  # Dyna: [B, T, sim_len, sims, ...] -> batch=0, T=0, all sim_len, sim=0/1
  if "dyna" in data:
    dyna = data["dyna"]
    dyna.pop("any_achievable", None)
    callback_data["dyna"] = jax.tree_util.tree_map(lambda x: x[0, 0, :, 0], dyna)
    callback_data["preplay"] = jax.tree_util.tree_map(lambda x: x[0, 0, :, 1], dyna)

  def plot_unified(d):
    """Plot online + all_goals columns (mimics her.py plot_unified)."""
    columns = ["online", "all_goals"]
    col_data = {c: d[c] for c in columns if c in d}

    if not col_data:
      return

    n_cols = len(col_data)
    col_names = list(col_data.keys())

    # Use first available column to get dimensions
    first_d = col_data[col_names[0]]
    timesteps = first_d["timesteps"]
    nT = len(timesteps.reward)
    maze_height, maze_width, _ = timesteps.state.grid[0].shape

    # Detect episodes in T dimension
    is_first = timesteps.first()
    episode_starts = list(jnp.where(is_first)[0])
    if not episode_starts or episode_starts[0] != 0:
      episode_starts = [0] + episode_starts
    episode_ranges = []
    for i, start in enumerate(episode_starts):
      end = episode_starts[i + 1] if i + 1 < len(episode_starts) else nT
      episode_ranges.append((int(start), int(end)))

    n_episodes = len(episode_ranges)
    n_traj_rows = ceil(n_episodes / n_cols)

    # Layout: 2 data rows + n_traj_rows trajectory rows
    n_rows = 2 + n_traj_rows
    fig, axes = plt.subplots(
      n_rows,
      n_cols,
      figsize=(8 * n_cols, 4 * n_rows),
      gridspec_kw={"hspace": 0.4, "wspace": 0.3},
    )
    if n_cols == 1:
      axes = axes[:, None]

    # Row 0: Reward, Q-values, Q-targets
    for ci, col_name in enumerate(col_names):
      ax = axes[0, ci]
      cd = col_data[col_name]
      loss_mask = cd.get("loss_mask")
      goal_reward = cd.get("goal_reward")

      if goal_reward is not None:
        ax.plot(goal_reward, label="Goal Reward")

      actions = cd["actions"]
      q_values = cd["q_values"]
      q_target = cd["q_target"]
      q_values_taken = rlax.batched_index(q_values, actions)
      ax.plot(q_values_taken, label="Q-Values")
      ax.plot(q_target, label="Q-Targets")
      if loss_mask is not None:
        ax.plot(loss_mask * 0.5, label="Loss Mask", linestyle="--", color="black")
      ax.set_title(f"{col_name} — Rewards and Q-Values", fontsize=13.5)
      if ci == 0:
        ax.legend(fontsize=10)
      ax.grid(True)
      ax.set_xticks(range(nT))
      ax.set_ylim(-0.1, 1.5)

    # Row 1: Combined heatmap (task_vector + achievements)
    for ci, col_name in enumerate(col_names):
      ax = axes[1, ci]
      cd = col_data[col_name]

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
        ax.set_title(f"{col_name} — Task+Ach", fontsize=13.5)
        ax.set_ylabel("Dims")
        ax.set_yticks(range(combined.shape[0]))
        ax.set_xticks(range(nT))
        ax.grid(True, axis="x", alpha=0.3, color="white")
        # Draw white vertical lines at episode boundaries
        for ep_start in episode_starts[1:]:
          ax.axvline(ep_start - 0.5, color="white", linestyle="-", linewidth=2)
      else:
        ax.set_visible(False)

    # Rows 2+: Trajectory renders with arrows
    for ep_idx, (ep_start, ep_end) in enumerate(episode_ranges):
      row_idx = 2 + ep_idx // n_cols
      col_idx = ep_idx % n_cols
      ax = axes[row_idx, col_idx]

      initial_state = jax.tree_util.tree_map(lambda x: x[ep_start], timesteps.state)
      img = render_fn(initial_state)

      ep_positions = jax.tree_util.tree_map(
        lambda x: x[ep_start : ep_end - 1], timesteps.state.agent_pos
      )
      ep_actions = first_d["actions"][ep_start : ep_end - 1]

      renderer.place_arrows_on_image(
        img, ep_positions, ep_actions, maze_height, maze_width, arrow_scale=5, ax=ax
      )
      total_reward = float(timesteps.reward[ep_start:ep_end].sum())
      ep_timestep = jax.tree_util.tree_map(lambda x: x[ep_start], timesteps)
      goal_name = get_task_name(extract_task_info(ep_timestep))
      ax.set_title(f"Ep {ep_idx} ({goal_name}, r={total_reward:.1f})", fontsize=8)
      ax.axis("off")

    # Hide unused trajectory subplots
    total_traj_slots = n_traj_rows * n_cols
    for slot_idx in range(n_episodes, total_traj_slots):
      row_idx = 2 + slot_idx // n_cols
      col_idx = slot_idx % n_cols
      axes[row_idx, col_idx].set_visible(False)

    fig.tight_layout()

    if wandb.run is not None:
      wandb.log({"learner_example/online": wandb.Image(fig)})
    plt.close(fig)

  def plot_dyna(d):
    """Plot dyna + preplay columns."""
    columns = ["dyna", "preplay"]
    col_data = {c: d[c] for c in columns if c in d}

    if not col_data:
      return

    n_cols = len(col_data)
    col_names = list(col_data.keys())

    n_rows = 3
    fig, axes = plt.subplots(
      n_rows,
      n_cols,
      figsize=(8 * n_cols, 4 * n_rows),
      gridspec_kw={"hspace": 0.4, "wspace": 0.3},
    )
    if n_cols == 1:
      axes = axes[:, None]

    for ci, col_name in enumerate(col_names):
      cd = col_data[col_name]
      timesteps = cd["timesteps"]
      actions = cd["actions"]
      q_values = cd["q_values"]
      q_target = cd["q_target"]
      td_errors = cd["td_errors"]
      loss_mask = cd.get("loss_mask")
      rewards = timesteps.reward
      nT = len(rewards)
      q_values_taken = rlax.batched_index(q_values, actions)

      # Row 0: Rewards, Q-values, Q-targets
      ax = axes[0, ci]
      ax.plot(rewards, label="Rewards")
      ax.plot(q_values_taken, label="Q-Values")
      ax.plot(q_target, label="Q-Targets")
      if loss_mask is not None:
        ax.plot(loss_mask * 0.5, label="Loss Mask", linestyle="--", color="black")
      ax.set_title(f"{col_name} — Rewards and Q-Values")
      if ci == 0:
        ax.legend(fontsize=10)
      ax.grid(True)
      ax.set_xticks(range(nT))

      # Row 1: TD errors
      ax = axes[1, ci]
      ax.plot(td_errors)
      ax.set_title(f"{col_name} — TD Errors")
      ax.grid(True)
      ax.set_xticks(range(len(td_errors)))

      # Row 2: Environment image + path
      ax = axes[2, ci]
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
      ax.set_title(f"{col_name} — Trajectory")
      ax.axis("off")

    fig.tight_layout()

    if wandb.run is not None:
      wandb.log({"learner_example/preplay": wandb.Image(fig)})
    plt.close(fig)

  def callback(d):
    try:
      plot_unified(d)
    except Exception as e:
      print(f"plot_unified error: {e}")
    try:
      plot_dyna(d)
    except Exception as e:
      print(f"plot_dyna error: {e}")

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
    "STEP_COST": 0.0,
    "TARGET_UPDATE_INTERVAL": 10,
    # Epsilon
    "FIXED_EPSILON": 2,
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

  # ---- Craftax Multigoal Test ----
  import craftax_simulation_configs
  import craftax_observer
  from jaxneurorl.wrappers import TimestepWrapper
  from craftax_web_env import CraftaxMultiGoalSymbolicWebEnvNoAutoReset

  static_env_params = (
    CraftaxMultiGoalSymbolicWebEnvNoAutoReset.default_static_params().replace(
      landmark_features=False
    )
  )
  craftax_base_env = CraftaxMultiGoalSymbolicWebEnvNoAutoReset(
    static_env_params=static_env_params
  )
  craftax_env = TimestepWrapper(LogWrapper(craftax_base_env), autoreset=True)

  craftax_env_params = craftax_simulation_configs.default_params.replace(
    task_configs=craftax_simulation_configs.TRAIN_CONFIGS
  )
  craftax_test_env_params = craftax_simulation_configs.default_params.replace(
    task_configs=craftax_simulation_configs.TEST_CONFIGS
  )

  craftax_config = {
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
    "MLP_HIDDEN_DIM": 32,
    "NUM_MLP_LAYERS": 0,
    "AGENT_RNN_DIM": 32,
    "Q_HIDDEN_DIM": 64,
    "NUM_PRED_LAYERS": 1,
    "ACTIVATION": "leaky_relu",
    "QHEAD_TYPE": "duelling",
    "USE_BIAS": False,
    "OBS_INCLUDE_GOAL": False,
    # Preplay params
    "SIMULATION_LENGTH": 3,
    "NUM_OFFTASK_SIMULATIONS": 1,
    "NUM_ONTASK_SIMULATIONS": 1,
    "NUM_OFFTASK_GOALS": 1,
    "ONLINE_COEFF": 1.0,
    "DYNA_COEFF": 1.0,
    "ALL_GOALS_COEFF": 1.0,
    "MAINQ_COEFF": 1e-2,
    "SUBTASK_COEFF": 2.0,
    # Optimizer
    "LR": 0.001,
    "MAX_GRAD_NORM": 80,
    "EPS_ADAM": 1e-5,
    "LR_LINEAR_DECAY": False,
    "GAMMA": 0.99,
    "TD_LAMBDA": 0.9,
    "ALL_GOALS_LAMBDA": 0.6,
    "STEP_COST": 0.0,
    "TARGET_UPDATE_INTERVAL": 10,
    # Epsilon
    "FIXED_EPSILON": 2,
    "EPSILON_START": 1.0,
    "EPSILON_FINISH": 0.1,
    # Logging
    "LEARNER_LOG_PERIOD": 0,
    "LEARNER_EXTRA_LOG_PERIOD": 0,
    "EVAL_LOG_PERIOD": 0,
    "GRADIENT_LOG_PERIOD": 0,
    "MAX_EPISODE_LOG_LEN": 10,
    "EVAL_EPISODES": 1,
    "EVAL_STEPS": 10,
    "ALG": "preplay",
    "PROJECT": "test",
    "ENV_NAME": "craftax",
    "ENV": "craftax",
    "NUM_UPDATES": 1,
  }

  from jaxneurorl import loggers as jnrl_loggers

  def noop_experience_logger(*args, **kwargs):
    pass

  def craftax_make_logger(config, env, env_params, learner_log_extra=None):
    return jnrl_loggers.Logger(
      gradient_logger=jnrl_loggers.default_gradient_logger,
      learner_logger=jnrl_loggers.default_learner_logger,
      experience_logger=noop_experience_logger,
    )

  craftax_train_fn = make_train_craftax_multigoal(
    config=craftax_config,
    env=craftax_env,
    save_path="/tmp/multitask_preplay_craftax_test",
    model_env=craftax_env,
    train_env_params=craftax_env_params,
    test_env_params=craftax_test_env_params,
    ObserverCls=craftax_observer.Observer,
    make_logger=craftax_make_logger,
  )

  rng = jax.random.PRNGKey(craftax_config["SEED"])
  craftax_train_fn = jax.jit(craftax_train_fn)
  print("JIT compiling Craftax multigoal test...")
  outs = craftax_train_fn(rng)
  print("Craftax multigoal test passed!")
