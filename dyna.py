"""
Dyna
"""

from typing import Tuple, Optional, Callable
import functools
from functools import partial

import jax
import jax.numpy as jnp
import flax
from flax import struct
import optax
import flax.linen as nn
from gymnax.environments import environment
import numpy as np
import rlax

import matplotlib.pyplot as plt
import wandb

import losses
from jaxneurorl.agents.basics import TimeStep
from jaxneurorl.agents import value_based_basics as vbb
from jaxneurorl.agents import qlearning as base_agent
from networks import (
  MLP,
  CraftaxObsEncoder,
  CraftaxMultiGoalObsEncoder,
  CategoricalJaxmazeObsEncoder,
)

from jaxmaze import renderer

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


##############################
# Simulation
##############################
def simulate_n_trajectories(
  h_tm1: RnnState,
  x_t: TimeStep,
  rng: jax.random.PRNGKey,
  network: nn.Module,
  params: Params,
  policy_fn: Callable = None,
  q_fn: Callable = None,
  num_steps: int = 5,
  num_simulations: int = 5,
):
  """Simulate n trajectories from a given state.

  Returns predictions and actions for every time-step including the current one.
  This first applies the model to the current time-step and then simulates T more
  time-steps. Output length is num_steps+1.

  Args:
    h_tm1: RNN state at previous timestep [D] (will be broadcast across sims)
    x_t: current timestep (no batch dim)
    rng: random key
    network: agent module
    params: network parameters
    policy_fn: (Predictions, rng) -> actions [num_simulations]
    q_fn: Q-function method on network
    num_steps: number of simulation steps
    num_simulations: number of parallel simulations
  """

  def initial_predictions(x, prior_h, rng_):
    lstm_state, lstm_out = network.apply(
      params, prior_h, x, rng_, method=network.apply_rnn
    )
    preds = Predictions(
      q_vals=network.apply(params, lstm_out, None, method=q_fn), state=lstm_state
    )
    return x, lstm_state, preds

  rng, rng_ = jax.random.split(rng)

  x_t, h_t, preds_t = jax.vmap(
    initial_predictions, in_axes=(None, None, 0), out_axes=0
  )(
    x_t,
    h_tm1,
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
      q_vals=network.apply(params, next_rnn_out, None, method=q_fn),
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


def apply_rnn_and_q(
  h_tm1: RnnState,
  timesteps: TimeStep,
  task,
  rng: jax.random.PRNGKey,
  network: nn.Module,
  params: Params,
  q_fn: Callable = None,
):
  """Sequentially applies RNN and Q-function to a sequence of timesteps using scan.

  Args:
    h_tm1: Initial RNN state [N, D]
    timesteps: Sequence of timesteps [T, N, ...]
    task: Task specification (unused for single-task Dyna, pass None)
    rng: Random key
    network: Network module
    params: Network parameters
    q_fn: Q-function method to apply
  """

  def _single_step(carry, x_t):
    lstm_state, rng = carry
    rng, rng_ = jax.random.split(rng)

    next_lstm_state, next_rnn_out = network.apply(
      params, lstm_state, x_t, rng_, method=network.apply_rnn
    )

    next_preds = Predictions(
      q_vals=network.apply(params, next_rnn_out, task, method=q_fn),
      state=next_lstm_state,
    )

    return (next_lstm_state, rng), next_preds

  initial_carry = (h_tm1, rng)
  _, preds = jax.lax.scan(f=_single_step, init=initial_carry, xs=timesteps)

  return preds


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
  def __call__(self, x, task=None, train: bool = False):
    del task
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


##############################
# Loss Function
##############################
@struct.dataclass
class DynaLossFn:
  """Standalone Dyna loss function.

  Simulates from every timestep directly (no rolling windows or backtracking).
  """

  network: nn.Module
  discount: float = 0.99
  lambda_: float = 0.9
  step_cost: float = 0.0
  max_priority_weight: float = 0.9
  importance_sampling_exponent: float = 0.6
  burn_in_length: int = None

  data_wrapper: flax.struct.PyTreeNode = vbb.AcmeBatchData
  logger: vbb.loggers.Logger = vbb.loggers.Logger

  # Dyna-specific
  num_simulations: int = 2
  simulation_length: int = 5
  online_coeff: float = 1.0
  dyna_coeff: float = 1.0
  dyna_epsilon_values: Callable = None

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

    if self.burn_in_length:
      burn_data = jax.tree_util.tree_map(lambda x: x[: self.burn_in_length], data)
      key_grad, rng_1, rng_2 = jax.random.split(key_grad, 3)
      _, online_state = unroll(params, online_state, burn_data.timestep, rng_1)
      _, target_state = unroll(target_params, target_state, burn_data.timestep, rng_2)
      data = jax.tree_util.tree_map(lambda seq: seq[self.burn_in_length :], data)

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
  ):
    rewards = make_float(rewards)
    rewards = rewards - self.step_cost
    is_last = make_float(is_last)
    discounts = make_float(non_terminal) * self.discount
    lambda_ = jnp.ones_like(non_terminal) * self.lambda_

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
      all_metrics.update({f"{k}/online": v for k, v in metrics.items()})
      all_log_info["online"] = log_info
      td_error = jnp.concatenate((td_error, jnp.zeros(B)[None]), 0)
      td_error = jnp.abs(td_error)
    else:
      td_error = jnp.zeros_like(loss_mask)
      batch_loss = td_error.sum(0)

    # ---- Dyna loss ----
    if self.dyna_coeff > 0.0:
      remove_last = lambda x: jax.tree_util.tree_map(lambda y: y[:-1], x)
      h_tm1_online = concat_first_rest(online_state, remove_last(online_preds.state))
      h_tm1_target = concat_first_rest(target_state, remove_last(target_preds.state))
      x_t = data.timestep

      dyna_loss_fn = functools.partial(
        self.dyna_loss_fn, params=params, target_params=target_params
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
      all_metrics.update({f"{k}/dyna": v for k, v in dyna_metrics.items()})
      all_log_info["dyna"] = dyna_log_info

    if self.logger.learner_log_extra is not None:
      self.logger.learner_log_extra(all_log_info)

    return td_error, batch_loss, all_metrics

  def dyna_loss_fn(
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
    """Dyna loss for a single batch element. Vmaps over timesteps directly."""
    simulate = partial(
      simulate_n_trajectories,
      network=self.network,
      params=params,
      num_steps=self.simulation_length,
      num_simulations=self.num_simulations,
      q_fn=self.network.reg_q_fn,
    )

    def dyna_at_t(x_t, h_on_t, h_tar_t, l_mask_t, key):
      """Dyna loss at a single timestep."""
      num_sims = self.num_simulations

      # eps: [num_sims]
      key, rng_eps = jax.random.split(key)
      eps = self.dyna_epsilon_values(rng_eps)

      def sim_policy(preds, rng):
        rngs = jax.random.split(rng, num_sims)
        return jax.vmap(base_agent.epsilon_greedy_act)(preds.q_vals, eps, rngs)

      # all_t: [sim_length+1, num_sims, ...]
      # sim_outputs_t.actions: [sim_length+1, num_sims]
      key, key_ = jax.random.split(key)
      next_t, sim_outputs_t = simulate(
        h_tm1=h_on_t,
        x_t=x_t,
        rng=key_,
        policy_fn=sim_policy,
      )

      # all_t: [sim_length+1, num_sims, ...]
      all_t = next_t
      # all_a: [sim_length+1, num_sims]
      all_a = sim_outputs_t.actions

      # Re-run RNN + Q with online params
      # h_on_rep: [num_sims, D]
      key, key_ = jax.random.split(key)
      h_on_rep = repeat(h_on_t, num_sims)
      # online_preds.q_vals: [sim_length+1, num_sims, A]
      online_preds = apply_rnn_and_q(
        h_tm1=h_on_rep,
        timesteps=all_t,
        task=None,
        rng=key_,
        network=self.network,
        params=params,
        q_fn=self.network.reg_q_fn,
      )

      # Re-run RNN + Q with target params
      # h_tar_rep: [num_sims, D]
      key, key_ = jax.random.split(key)
      h_tar_rep = repeat(h_tar_t, num_sims)
      # target_preds.q_vals: [sim_length+1, num_sims, A]
      target_preds = apply_rnn_and_q(
        h_tm1=h_tar_rep,
        timesteps=all_t,
        task=None,
        rng=key_,
        network=self.network,
        params=target_params,
        q_fn=self.network.reg_q_fn,
      )

      # all_t_mask: [sim_length+1, num_sims]
      init_mask = jnp.broadcast_to(l_mask_t, (num_sims,))
      all_t_mask = simulation_finished_mask(init_mask, next_t)

      batch_td_error, batch_loss_mean, metrics, log_info = self.loss_fn(
        timestep=all_t,
        online_preds=online_preds,
        target_preds=target_preds,
        actions=all_a,
        rewards=all_t.reward,
        is_last=make_float(all_t.last()),
        non_terminal=all_t.discount,
        loss_mask=all_t_mask,
      )
      return batch_td_error, batch_loss_mean, metrics, log_info

    # Vmap over timesteps T
    batch_td_error, batch_loss_mean, metrics, log_info = jax.vmap(dyna_at_t)(
      timesteps,
      h_online,
      h_target,
      loss_mask,
      jax.random.split(rng, len(actions)),
    )

    batch_td_error = batch_td_error.mean()
    batch_loss_mean = batch_loss_mean.mean()

    return batch_td_error, batch_loss_mean, metrics, log_info


##############################
# make_loss_fn_class
##############################
def make_loss_fn_class(config, **kwargs) -> DynaLossFn:
  return functools.partial(
    DynaLossFn,
    discount=config["GAMMA"],
    lambda_=config.get("TD_LAMBDA", 0.9),
    online_coeff=config.get("ONLINE_COEFF", 1.0),
    dyna_coeff=config.get("DYNA_COEFF", 1.0),
    num_simulations=config.get("NUM_SIMULATIONS", 2),
    simulation_length=config.get("SIMULATION_LENGTH", 5),
    importance_sampling_exponent=config.get("IMPORTANCE_SAMPLING_EXPONENT", 0.6),
    max_priority_weight=config.get("MAX_PRIORITY_WEIGHT", 0.9),
    step_cost=config.get("STEP_COST", 0.0),
    **kwargs,
  )


##############################
# Agent Classes
##############################
class DynaAgentEnvModel(nn.Module):
  observation_encoder: nn.Module
  rnn: vbb.ScannedRNN
  q_fn: nn.Module
  env: environment.Environment
  env_params: environment.EnvParams

  def initialize(self, x: TimeStep):
    rng = jax.random.PRNGKey(0)
    batch_dims = (x.reward.shape[0],)
    rnn_state = self.initialize_carry(rng, batch_dims)
    self.__call__(rnn_state, x, rng)

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
    q_vals = self.q_fn(rnn_out)
    predictions = Predictions(q_vals=q_vals, state=new_rnn_state)
    return predictions, new_rnn_state

  def reg_q_fn(self, rnn_out, task):
    del task
    return self.q_fn(rnn_out)

  def unroll(
    self, rnn_state, xs: TimeStep, rng: jax.random.PRNGKey
  ) -> Tuple[Predictions, RnnState]:
    embedding = jax.vmap(self.observation_encoder)(xs.observation)
    rnn_in = vbb.RNNInput(obs=embedding, reset=xs.first())
    rng, _rng = jax.random.split(rng)
    new_rnn_state, new_rnn_states = self.rnn.unroll(rnn_state, rnn_in, _rng)
    rnn_out = self.rnn.output_from_state(new_rnn_states)
    q_vals = nn.BatchApply(self.q_fn)(rnn_out)
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
def make_craftax_agent(
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
  if config.get("MULTIGOAL", False):
    observation_encoder = CraftaxMultiGoalObsEncoder(
      hidden_dim=config["MLP_HIDDEN_DIM"],
      num_layers=config["NUM_MLP_LAYERS"],
      activation=config["ACTIVATION"],
      norm_type=config.get("NORM_TYPE", "none"),
    )
  else:
    observation_encoder = CraftaxObsEncoder(
      hidden_dim=config["MLP_HIDDEN_DIM"],
      num_layers=config["NUM_MLP_LAYERS"],
      activation=config["ACTIVATION"],
      norm_type=config.get("NORM_TYPE", "none"),
      structured_inputs=config.get("STRUCTURED_INPUTS", False),
      use_bias=config.get("USE_BIAS", True),
      action_dim=env.action_space(env_params).n,
    )
  agent = DynaAgentEnvModel(
    observation_encoder=observation_encoder,
    rnn=rnn,
    q_fn=DuellingMLP(
      hidden_dim=config.get("Q_HIDDEN_DIM", 512),
      num_layers=config.get("NUM_Q_LAYERS", 1),
      out_dim=env.action_space(env_params).n,
      activation=config["ACTIVATION"],
      activate_final=False,
      use_bias=config.get("USE_BIAS", True),
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


def make_jaxmaze_agent(
  config: dict,
  env: environment.Environment,
  env_params: environment.EnvParams,
  example_timestep: TimeStep,
  rng: jax.random.PRNGKey,
  model_env: Optional[environment.Environment] = None,
  model_env_params: Optional[environment.EnvParams] = None,
) -> Tuple[nn.Module, Params, vbb.AgentResetFn]:
  model_env = model_env or env
  model_env_params = model_env_params or env_params
  rnn = vbb.ScannedRNN(
    hidden_dim=config.get("AGENT_RNN_DIM", 256),
    cell_type=config.get("RNN_CELL_TYPE", "OptimizedLSTMCell"),
    unroll_output_state=True,
  )
  observation_encoder = CategoricalJaxmazeObsEncoder(
    num_categories=max(10_000, env.total_categories(env_params)),
    embed_hidden_dim=config["EMBED_HIDDEN_DIM"],
    mlp_hidden_dim=config["MLP_HIDDEN_DIM"],
    num_embed_layers=config["NUM_EMBED_LAYERS"],
    num_mlp_layers=config["NUM_MLP_LAYERS"],
    activation=config["ACTIVATION"],
    norm_type=config.get("NORM_TYPE", "none"),
  )
  agent = DynaAgentEnvModel(
    observation_encoder=observation_encoder,
    rnn=rnn,
    q_fn=DuellingMLP(
      hidden_dim=config.get("Q_HIDDEN_DIM", 512),
      num_layers=config.get("NUM_Q_LAYERS", 1),
      out_dim=env.num_actions(env_params),
      activation=config["ACTIVATION"],
      activate_final=False,
      use_bias=config.get("USE_BIAS", True),
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
# make_train
##############################
def make_train(**kwargs):
  config = kwargs["config"]
  vals = np.logspace(num=256, start=1, stop=3, base=0.1)
  num_simulations = config["NUM_SIMULATIONS"]

  def dyna_epsilon_values(rng):
    epsilons = jax.random.choice(rng, vals, shape=(num_simulations - 1,))
    return jnp.concatenate((jnp.array([0.0]), epsilons))

  make_agent = kwargs.pop("make_agent", None)
  if make_agent is None:
    make_agent = partial(make_craftax_agent, model_env=kwargs.pop("model_env", None))

  return vbb.make_train(
    make_agent=make_agent,
    make_loss_fn_class=functools.partial(
      make_loss_fn_class,
      dyna_epsilon_values=dyna_epsilon_values,
    ),
    make_optimizer=base_agent.make_optimizer,
    make_actor=make_actor,
    **kwargs,
  )


##############################
# learner_log_extra
##############################
def learner_log_extra(
  data: dict,
  config: dict,
):
  def log_data(
    key: str,
    timesteps: TimeStep,
    actions: np.array,
    td_errors: np.array,
    loss_mask: np.array,
    q_values: np.array,
    q_loss: np.array,
    q_target: np.array,
  ):
    discounts = timesteps.discount
    rewards = timesteps.reward
    q_values_taken = rlax.batched_index(q_values, actions)

    width = 0.3
    nT = len(rewards)
    width = max(int(width * nT), 10)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(width, 20))

    def format(ax):
      ax.set_xlabel("Time")
      ax.grid(True)
      ax.set_xticks(range(0, len(rewards), 1))

    ax1.plot(rewards, label="Rewards")
    ax1.plot(q_values_taken, label="Q-Values")
    ax1.plot(q_target, label="Q-Targets")
    ax1.set_title("Rewards and Q-Values")
    format(ax1)
    ax1.legend()

    ax2.plot(td_errors)
    format(ax2)
    ax2.set_title("TD Errors")

    ax3.plot(q_loss)
    format(ax3)
    ax3.set_title("Q-Loss")

    is_last = timesteps.last()
    ax4.plot(discounts, label="Discounts")
    ax4.plot(loss_mask, label="mask")
    ax4.plot(is_last, label="is_last")
    format(ax4)
    ax4.set_title("Episode markers")
    ax4.legend()

    if wandb.run is not None:
      wandb.log({f"learner_example/{key}/q-values": wandb.Image(fig)})
    plt.close(fig)

  def callback(d):
    log_data(**d, key="dyna")

  n_updates = data["n_updates"] + 1
  if config["LEARNER_EXTRA_LOG_PERIOD"] > 0:
    is_log_time = n_updates % config["LEARNER_EXTRA_LOG_PERIOD"] == 0

    if "dyna" in data:
      # [Batch, Env Time, Sim Time, Num Simulations]
      dyna_data = jax.tree_util.tree_map(lambda x: x[0, 0, :, 0], data["dyna"])

      jax.lax.cond(
        is_log_time,
        lambda d: jax.debug.callback(callback, d),
        lambda d: None,
        dyna_data,
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
    "NUM_Q_LAYERS": 1,
    "ACTIVATION": "leaky_relu",
    "USE_BIAS": False,
    # Dyna params
    "SIMULATION_LENGTH": 3,
    "NUM_SIMULATIONS": 2,
    "ONLINE_COEFF": 1.0,
    "DYNA_COEFF": 1.0,
    # Optimizer
    "LR": 0.001,
    "MAX_GRAD_NORM": 80,
    "EPS_ADAM": 1e-5,
    "LR_LINEAR_DECAY": False,
    "GAMMA": 0.99,
    "TD_LAMBDA": 0.9,
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
    "EVAL_EPISODES": 1,
    "EVAL_STEPS": 10,
    "ALG": "dyna",
    "PROJECT": "test",
    "ENV_NAME": "jaxmaze",
    "ENV": "jaxmaze",
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

  train_fn = make_train(
    config=config,
    env=env,
    save_path="/tmp/dyna_test",
    train_env_params=env_params,
    test_env_params=test_env_params,
    ObserverCls=functools.partial(
      humansf_observers.TaskObserver,
      action_names={},
      extract_task_info=lambda t: t,
    ),
    initial_params=None,
    make_agent=partial(make_jaxmaze_agent, model_env=env),
  )

  rng = jax.random.PRNGKey(config["SEED"])
  train_fn = jax.jit(train_fn)
  print("JIT compiling...")
  outs = train_fn(rng)
  print("Test passed!")
