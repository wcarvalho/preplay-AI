"""
Hindsight experience replay

This is a self-contained module that only depends on:
1. base_algorithm.py (consolidated jaxneurorl components)
2. networks.py (CraftaxObsEncoder, CraftaxMultiGoalObsEncoder)
"""

from distrax import Categorical
from networks import (
  CategoricalJaxmazeObsEncoder,
  CraftaxMultiGoalObsEncoder,
  CraftaxObsEncoder,
)
from base_algorithm import TimeStep
from visualizer import plot_frames
from craftax.craftax.renderer import render_craftax_pixels
from craftax.craftax.constants import BLOCK_PIXEL_SIZE_IMG, Achievement, Action
import functools
from functools import partial
from typing import Callable, NamedTuple, Union, Optional

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend before importing pyplot

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import rlax
import wandb
from flax import struct
from gymnax.environments import environment
import base_algorithm as base

make_optimizer = base.make_optimizer
make_actor = base.make_actor

MAX_REWARD = 1.0

Agent = nn.Module
Params = flax.core.FrozenDict

Goal = flax.struct.PyTreeNode
AgentState = flax.struct.PyTreeNode
RNNInput = base.RNNInput


ENVIRONMENT_TO_GOAL_FNS = {
  "jaxmaze": (
    lambda t: t.observation.task_w,
    lambda t: t.observation.state_features,
    lambda t: t.observation.position,  # jaxmaze uses .position
  ),
  "craftax-multigoal": (
    lambda t: t.observation.task_w,
    lambda t: t.observation.state_features,
    lambda t: t.observation.player_position,
  ),
  "craftax-gen": (
    lambda t: jnp.ones_like(t.observation.achievements),
    lambda t: t.observation.achievements,
    lambda t: t.observation.player_position,
  ),
}


class Predictions(NamedTuple):
  q_vals: jax.Array
  rnn_states: jax.Array


class GoalPosition(NamedTuple):
  goal: jax.Array
  position: jax.Array


class RnnAgent(nn.Module):
  """_summary_

  - observation encoder: CNN
  Args:
      nn (_type_): _description_

  Returns:
      _type_: _description_
  """

  observation_encoder: nn.Module
  rnn: base.ScannedRNN
  q_fn: nn.Module
  task_encoder: nn.Module
  goal_from_timestep: Callable[[TimeStep], jax.Array]

  def setup(self):
    kernel_init = nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)
    self.task_fn = nn.Dense(128, kernel_init=kernel_init)

  def initialize(self, x: TimeStep):
    """Only used for initialization."""

    rng = jax.random.PRNGKey(0)
    batch_dims = (x.reward.shape[0],)
    rnn_state = self.initialize_carry(rng, batch_dims)

    return self(rnn_state, x, rng)

  def __call__(self, rnn_state, x: TimeStep, rng: jax.random.PRNGKey):
    embedding = self.observation_encoder(x.observation)

    rnn_in = RNNInput(obs=embedding, reset=x.first())
    rng, _rng = jax.random.split(rng)

    new_rnn_state, rnn_out = self.rnn(rnn_state, rnn_in, _rng)
    goal_embedding = jax.vmap(self.task_encoder)(self.goal_from_timestep(x))
    q_in = jnp.concatenate((rnn_out, goal_embedding), axis=-1)

    q_vals = self.q_fn(q_in)

    return Predictions(
      q_vals=q_vals,
      rnn_states=rnn_out,
    ), new_rnn_state

  def unroll(
    self, rnn_state, xs: TimeStep, rng: jax.random.PRNGKey, goal: Optional[Goal] = None
  ):
    # rnn_state: [B, D]
    # xs: [T, B] or [T]
    ref = rnn_state[0]
    has_batch = ref.ndim == 2
    assert ref.ndim in (1, 2), f"leading dims: {ref.shape[:-1]}"

    observation_encoder = self.observation_encoder
    q_fn = self.q_fn
    task_encoder = jax.vmap(self.task_encoder)

    if has_batch:
      observation_encoder = nn.BatchApply(observation_encoder)
      q_fn = nn.BatchApply(q_fn)
      task_encoder = jax.vmap(task_encoder)

    if goal is None:
      goal = self.goal_from_timestep(xs)

    embedding = observation_encoder(xs.observation)

    rng, _rng = jax.random.split(rng)
    rnn_in = RNNInput(obs=embedding, reset=xs.first())
    new_rnn_state, rnn_out = self.rnn.unroll(rnn_state, rnn_in, _rng)

    goal_embedding = task_encoder(goal)

    q_in = jnp.concatenate((rnn_out, goal_embedding), axis=-1)
    q_vals = q_fn(q_in)

    return Predictions(
      q_vals=q_vals,
      rnn_states=rnn_out,
    ), new_rnn_state

  def initialize_carry(self, *args, **kwargs):
    """Initializes the RNN state."""
    return self.rnn.initialize_carry(*args, **kwargs)


def make_float(x):
  return x.astype(jnp.float32)


def is_truncated(timestep):
  non_terminal = timestep.discount
  # either termination or truncation
  is_last = make_float(timestep.last())

  # truncated is discount=1 on AND is last
  truncated = (non_terminal + is_last) > 1
  return make_float(1 - truncated)


def episode_finished_mask(timesteps):
  # either termination or truncation
  is_last_t = make_float(timesteps.last())

  # time-step of termination and everything afterwards is masked out
  term_cumsum_t = jnp.cumsum(is_last_t, 0)
  loss_mask_t = make_float((term_cumsum_t + timesteps.discount) < 2)
  return loss_mask_t


@struct.dataclass
class HerLossFn(base.RecurrentLossFn):
  """Loss function of R2D2.

  https://openreview.net/forum?id=r1lyTjAqYX
  """

  extract_q: Callable[[jax.Array], jax.Array] = lambda preds: preds.q_vals
  her_coeff: float = 1.0
  max_priority_weight: float = 0.9
  importance_sampling_exponent: float = 0.6
  ngoals: int = 10
  sample_goals: Callable[[base.TimeStep, jax.random.PRNGKey], jax.Array] = None
  online_reward_fn: Callable[[base.TimeStep], jax.Array] = None
  her_reward_fn: Callable[[base.TimeStep, Goal], jax.Array] = None

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

    # Get N-step transformed TD error and loss.
    # [T]
    selector_actions = jnp.argmax(online_preds.q_vals, axis=-1)  # [T+1]
    q_t, target_q_t = base.q_learning_lambda_td(
      online_preds.q_vals[:-1],  # [T+1] --> [T]
      actions[:-1],  # [T+1] --> [T]
      target_preds.q_vals[1:],  # [T+1] --> [T]
      selector_actions[1:],  # [T+1] --> [T]
      rewards[1:],  # [T+1] --> [T]
      discounts[1:],
      is_last[1:],
      lambda_[1:],
    )  # [T+1] --> [T]

    # ensure target = 0 when episode terminates
    target_q_t = target_q_t * non_terminal[:-1]
    batch_td_error = target_q_t - q_t
    batch_td_error = batch_td_error * loss_mask[:-1]

    # [T]
    batch_loss = 0.5 * jnp.square(batch_td_error)

    # []
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
      "td_errors": batch_td_error,  # [T]
      "loss_mask": loss_mask,  # [T]
      "q_values": online_preds.q_vals,  # [T]
      "q_loss": batch_loss,  # [T]
      "q_target": target_q_t,
    }
    return batch_td_error, batch_loss_mean, metrics, log_info

  def error(
    self,
    data,
    online_preds: Predictions,
    online_state: AgentState,
    target_preds: Predictions,
    target_state: AgentState,
    params: Params,
    target_params: Params,
    steps: int,
    key_grad: jax.random.PRNGKey,
    **kwargs,
  ):
    assert self.online_reward_fn is not None, f"Please provide `online_reward_fn`"
    assert self.her_reward_fn is not None, f"Please provide `her_reward_fn`"
    assert self.sample_goals is not None, f"Please provide `sample_goals`"

    ##################
    # Q-learning loss on batch of data
    ##################
    # truncated is discount on AND is last
    loss_mask = is_truncated(data.timestep)

    all_metrics = {}
    all_log_info = {
      "n_updates": steps,
    }
    T, B = loss_mask.shape[:2]

    # [T, B-1], [T]
    td_error, loss, metrics, log_info = jax.vmap(self.loss_fn, 1, 0)(
      data.timestep,
      online_preds,
      target_preds,
      data.action,
      self.online_reward_fn(data.timestep),
      make_float(data.timestep.last()),
      data.timestep.discount,
      loss_mask,
    )
    # [B, T] --> [T, B]
    td_error = jnp.swapaxes(td_error, 0, 1)
    # first label online loss with online
    all_metrics.update({f"{k}/online": v for k, v in metrics.items()})
    # all_log_info["online"] = log_info
    td_error = jnp.concatenate((td_error, jnp.zeros(B)[None]), 0)
    td_error = jnp.abs(td_error)

    ##################
    # Q-learning on fake goals
    ##################
    key_grad = jax.random.split(key_grad, B + 1)
    # 1. sample fake goals from trajectory using data.timestep
    # [N, B, D]
    new_goals = jax.vmap(self.sample_goals, (1, 0), 1)(data.timestep, key_grad[1:])

    # 2. Compute Q-values for new goals
    def her_loss(
      new_goal: Union[
        jnp.ndarray,
        struct.PyTreeNode,
      ],
      sub_key_grad,
      timestep,
      actions,
      loss_mask,
      online_state,
      target_state,
    ):
      """new_goal is [D], sub_key_grad is [2], loss_mask is [T]"""

      sub_key_grad, sub_key_grad_ = jax.random.split(sub_key_grad)
      length = len(data.action)

      # [D] --> [T, D]
      expand_over_time = lambda x: jnp.tile(x[None], [length, 1])
      expanded_new_goal = jax.tree.map(expand_over_time, new_goal)

      # [T, ...], [2], [T, D]
      inputs = (timestep, sub_key_grad_, expanded_new_goal)
      unroll = functools.partial(self.network.apply, method=self.network.unroll)
      online_preds, _ = unroll(params, online_state, *inputs)
      target_preds, _ = unroll(target_params, target_state, *inputs)

      reward = self.her_reward_fn(timestep, new_goal)
      return self.loss_fn(
        timestep=timestep,
        online_preds=online_preds,
        target_preds=target_preds,
        actions=actions,
        rewards=reward,
        is_last=make_float(timestep.last()),
        non_terminal=timestep.discount,
        loss_mask=loss_mask,
      )

    her_loss = jax.vmap(her_loss, (0, 0, None, None, None, None, None))
    her_loss = jax.vmap(her_loss, (1, 1, 1, 1, 1, 0, 0), 0)

    key_grad = jax.random.split(key_grad[0], self.ngoals * B).reshape(
      self.ngoals, B, -1
    )

    # [T, N]
    _, her_loss, her_metrics, her_log_info = her_loss(
      new_goals,  # [N, B, D]
      key_grad,  # [N, B, 2]
      data.timestep,  # [T, B, D]
      data.action,  # [T, B]
      episode_finished_mask(data.timestep),  # [T, B]
      online_state,  # [B, D]
      target_state,  # [B, D]
    )

    loss = loss + self.her_coeff * her_loss.mean(1)
    all_metrics.update({f"{k}/her": v for k, v in her_metrics.items()})
    all_log_info["her"] = her_log_info
    if self.logger.learner_log_extra is not None:
      self.logger.learner_log_extra(all_log_info)

    return td_error, loss, all_metrics


############################################
# Environment Specific
############################################


def make_loss_fn_class(config) -> base.RecurrentLossFn:
  task_vector_fn, achievement_fn, position_fn = ENVIRONMENT_TO_GOAL_FNS[config["ENV"]]

  def sample_goals(timesteps, rng):
    """
    There is data at T time-points. We want to"""
    # T, C
    achievements = achievement_fn(timesteps)
    # T
    goal_achieved = achievements.sum(-1)

    position_achieved = jnp.ones_like(goal_achieved)

    # T
    logits = config["GOAL_BETA"] * goal_achieved + position_achieved
    probabilities = logits / jnp.sum(logits, axis=-1, keepdims=True)

    # N
    indices = Categorical(probs=probabilities).sample(
      seed=rng, sample_shape=(config["NUM_HER_GOALS"])
    )

    index = lambda x, i: jax.lax.dynamic_index_in_dim(x, i, keepdims=False)
    index = jax.vmap(index, (None, 0), 0)

    # [N, C] <- [T, C], [N]
    features_at_indices = index(achievements, indices)

    # [N, 2] <- [T, 2], [N]
    positions_at_indices = index(position_fn(timesteps), indices)
    assert positions_at_indices.shape[0] == features_at_indices.shape[0]
    assert positions_at_indices.shape[1] == 2
    return GoalPosition(
      features_at_indices,
      positions_at_indices + 1,  # reserve 0 for empty position
    )

  def online_reward_fn(timesteps):
    task_vector = task_vector_fn(timesteps)
    achievements = achievement_fn(timesteps)
    goal_reward = (task_vector * achievements).sum(-1)

    return 0.5 * goal_reward

  def her_reward_fn(timesteps, new_goal):
    # position reward
    position = new_goal.position
    player_position = position_fn(timesteps)
    position_reward = (player_position == position).all(1).astype(jnp.float32)

    # achievement reward
    goal = new_goal.goal
    achievements = achievement_fn(timesteps)
    goal_reward = (goal * achievements).sum(-1)

    assert goal_reward.ndim == 1
    assert position_reward.ndim == 1
    return 0.5 * position_reward + 0.5 * goal_reward

  return functools.partial(
    HerLossFn,
    discount=config["GAMMA"],
    importance_sampling_exponent=config.get("IMPORTANCE_SAMPLING_EXPONENT", 0.6),
    max_priority_weight=config.get("MAX_PRIORITY_WEIGHT", 0.9),
    tx_pair=(
      rlax.SIGNED_HYPERBOLIC_PAIR
      if config.get("TX_PAIR", "none") == "hyperbolic"
      else rlax.IDENTITY_PAIR
    ),
    step_cost=config.get("STEP_COST", 0.0),
    her_coeff=config.get("her_COEFF", 0.001),
    ngoals=config.get("NUM_HER_GOALS", 10),
    sample_goals=sample_goals,
    online_reward_fn=online_reward_fn,
    her_reward_fn=her_reward_fn,
  )


@jax.jit
def craftax_render_fn(state):
  image = render_craftax_pixels(state, block_pixel_size=BLOCK_PIXEL_SIZE_IMG)
  return image / 255.0


def crafax_learner_log_fn(data: dict, config: dict):
  def callback(d):
    n_updates = d.pop("n_updates")

    # Extract the relevant data
    # only use data from batch dim = 0
    # [T, B, ...] --> # [T, ...]
    d_ = jax.tree_util.tree_map(lambda x: x[:, 0], d)

    mask = d_["mask"]
    discounts = d_["data"].timestep.discount
    rewards = d_["data"].timestep.reward
    actions = d_["data"].action
    q_values = d_["q_values"]
    q_target = d_["q_target"]
    q_values_taken = rlax.batched_index(q_values, actions)
    td_errors = d_["td_errors"]
    q_loss = d_["q_loss"]
    # Create a figure with three subplots
    width = 0.3
    nT = len(rewards)  # e.g. 20 --> 8
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(int(width * nT), 16))

    # Plot rewards and q-values in the top subplot
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

    # Plot TD errors in the middle subplot
    ax2.plot(td_errors)
    format(ax2)
    ax2.set_title("TD Errors")

    # Plot Q-loss in the bottom subplot
    ax3.plot(q_loss)
    format(ax3)
    ax3.set_title("Q-Loss")

    # Plot episode quantities
    is_last = d_["data"].timestep.last()
    ax4.plot(discounts, label="Discounts")
    ax4.plot(mask, label="mask")
    ax4.plot(is_last, label="is_last")
    format(ax4)
    ax4.set_title("Episode markers")
    ax4.legend()

    # Adjust the spacing between subplots
    # plt.tight_layout()
    # log
    if wandb.run is not wandb.sdk.lib.disabled.RunDisabled:
      wandb.log({f"learner_example/q-values": wandb.Image(fig)})
    plt.close(fig)

    ##############################
    # plot images of env
    ##############################
    # timestep = jax.tree_util.tree_map(lambda x: jnp.array(x), d_['data'].timestep)
    timesteps: TimeStep = d_["data"].timestep

    # ------------
    # get images
    # ------------

    obs_images = []
    max_len = min(config.get("MAX_EPISODE_LOG_LEN", 40), len(rewards))
    for idx in range(max_len):

      def index(y):
        return jax.tree_util.tree_map(lambda x: x[idx], y)

      obs_image = craftax_render_fn(index(d_["data"].timestep.state.env_state))
      obs_images.append(obs_image)
    # ------------
    # plot
    # ------------
    actions_taken = [Action(a).name for a in actions]

    def panel_title_fn(timesteps, i):
      title = f"t={i}"
      title += f"\n{actions_taken[i]}"
      if i >= len(timesteps.reward) - 1:
        return title
      title += (
        f"\nr={timesteps.reward[i + 1]:.2f}, $\\gamma={timesteps.discount[i + 1]}$"
      )

      if hasattr(timesteps.observation, "achievements"):
        achieved = timesteps.observation.achievements[i + 1]
        if achieved.sum() > 1e-5:
          achievement_idx = achieved.argmax()
          try:
            achievement = Achievement(achievement_idx).name
            title += f"\n{achievement}"
          except ValueError:
            title += f"\nHealth?"
      elif hasattr(timesteps.state, "current_goal"):
        start_location = timesteps.state.start_position
        goal = timesteps.state.current_goal
        goal_name = Achievement(goal).name
        title += f"\nstart={start_location}\ngoal={goal}\ngoal={goal_name}"
      return title

    fig = plot_frames(
      timesteps=timesteps,
      frames=obs_images,
      panel_title_fn=panel_title_fn,
      ncols=6,
    )
    if wandb.run is not wandb.sdk.lib.disabled.RunDisabled:
      wandb.log({"learner_example/trajectory": wandb.Image(fig)})
    plt.close(fig)

  # this will be the value after update is applied
  n_updates = data["n_updates"] + 1
  is_log_time = n_updates % config["LEARNER_EXTRA_LOG_PERIOD"] == 0

  jax.lax.cond(
    is_log_time, lambda d: jax.debug.callback(callback, d), lambda d: None, data
  )


class DuellingMLP(nn.Module):
  hidden_dim: int
  out_dim: int = 0
  num_layers: int = 1
  norm_type: str = "none"
  activation: str = "relu"
  activate_final: bool = True
  use_bias: bool = False

  @nn.compact
  def __call__(self, x, train: bool = False):
    value_mlp = base.MLP(
      hidden_dim=self.hidden_dim,
      num_layers=self.num_layers,
      use_bias=self.use_bias,
      out_dim=1,
    )
    advantage_mlp = base.MLP(
      hidden_dim=self.hidden_dim,
      num_layers=self.num_layers,
      use_bias=self.use_bias,
      out_dim=self.out_dim,
    )
    assert self.out_dim > 0, "must have at least one action"

    value = value_mlp(x)  # [B, 1]
    advantages = advantage_mlp(x)  # [B, A]

    # Advantages have zero mean.
    advantages -= jnp.mean(advantages, axis=-1, keepdims=True)  # [B, A]

    q_values = value + advantages  # [B, A]

    return q_values


# --------------------
# Craftax
# --------------------


class TaskEncoder(nn.Module):
  """Task encoder for regular craftax setting"""

  vocab_size: int = 128

  @nn.compact
  def __call__(self, goal: GoalPosition):
    # task is binary
    kernel_init = nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)
    # [D]
    task = nn.Dense(128, kernel_init=kernel_init, use_bias=False)(goal.goal)

    # position is categorical
    # [2, D]
    position = jax.vmap(flax.linen.Embed(100, self.vocab_size))(goal.position)
    return jnp.concatenate((task, position.reshape(-1)))


class CraftaxMulitgoalTaskEncoder(nn.Module):
  """Task encoder for regular craftax setting"""

  @nn.compact
  def __call__(self, obs: struct.PyTreeNode):
    achieved = obs.state_features


def goal_from_env_timestep(t: TimeStep, env: str):
  task_vector_fn, _, position_fn = ENVIRONMENT_TO_GOAL_FNS[env]
  return GoalPosition(
    task_vector_fn(t).astype(jnp.float32), jnp.zeros_like(position_fn(t))
  )


def make_craftax_agent(
  config: dict,
  env: environment.Environment,
  env_params: environment.EnvParams,
  example_timestep: TimeStep,
  rng: jax.random.PRNGKey,
):
  rnn = base.ScannedRNN(hidden_dim=config["AGENT_RNN_DIM"])

  agent = RnnAgent(
    observation_encoder=CraftaxObsEncoder(
      hidden_dim=config["MLP_HIDDEN_DIM"],
      num_layers=config["NUM_MLP_LAYERS"],
      activation=config["ACTIVATION"],
      norm_type=config.get("NORM_TYPE", "none"),
      structured_inputs=config.get("STRUCTURED_INPUTS", False),
      use_bias=config.get("USE_BIAS", True),
      action_dim=env.action_space(env_params).n,
    ),
    rnn=rnn,
    q_fn=DuellingMLP(
      hidden_dim=config.get("Q_HIDDEN_DIM", 512),
      num_layers=config.get("NUM_Q_LAYERS", 2),
      out_dim=env.action_space(env_params).n,
      use_bias=config.get("USE_BIAS", True),
    ),
    task_encoder=TaskEncoder(),
    goal_from_timestep=partial(goal_from_env_timestep, env=config["ENV"]),
  )

  rng, _rng = jax.random.split(rng)
  network_params = agent.init(_rng, example_timestep, method=agent.initialize)

  def reset_fn(params, example_timestep, reset_rng):
    batch_dims = (example_timestep.reward.shape[0],)
    return agent.apply(
      params, batch_dims=batch_dims, rng=reset_rng, method=agent.initialize_carry
    )

  return agent, network_params, reset_fn


def make_multigoal_craftax_agent(
  config: dict,
  env: environment.Environment,
  env_params: environment.EnvParams,
  example_timestep: TimeStep,
  rng: jax.random.PRNGKey,
):
  rnn = base.ScannedRNN(hidden_dim=config["AGENT_RNN_DIM"])

  agent = RnnAgent(
    observation_encoder=CraftaxMultiGoalObsEncoder(
      hidden_dim=config["MLP_HIDDEN_DIM"],
      num_layers=config["NUM_MLP_LAYERS"],
      activation=config["ACTIVATION"],
      norm_type=config.get("NORM_TYPE", "none"),
      use_bias=config.get("USE_BIAS", True),
      action_dim=env.action_space(env_params).n,
    ),
    task_encoder=TaskEncoder(),
    rnn=rnn,
    q_fn=DuellingMLP(
      hidden_dim=config.get("Q_HIDDEN_DIM", 512),
      num_layers=config.get("NUM_Q_LAYERS", 2),
      out_dim=env.action_space(env_params).n,
      use_bias=config.get("USE_BIAS", True),
    ),
    goal_from_timestep=partial(goal_from_env_timestep, env=config["ENV"]),
  )
  rng, _rng = jax.random.split(rng)
  network_params = agent.init(_rng, example_timestep, method=agent.initialize)

  def reset_fn(params, example_timestep, reset_rng):
    batch_dims = (example_timestep.reward.shape[0],)
    return agent.apply(
      params, batch_dims=batch_dims, rng=reset_rng, method=agent.initialize_carry
    )

  return agent, network_params, reset_fn


# --------------------
# JaxMaze
# --------------------


def make_jaxmaze_agent(
  config: dict,
  env: environment.Environment,
  env_params: environment.EnvParams,
  example_timestep: TimeStep,
  rng: jax.random.PRNGKey,
):
  rnn = base.ScannedRNN(hidden_dim=config["AGENT_RNN_DIM"])

  agent = RnnAgent(
    observation_encoder=CategoricalJaxmazeObsEncoder(
      num_categories=max(10_000, env.total_categories(env_params)),
      embed_hidden_dim=config["EMBED_HIDDEN_DIM"],
      mlp_hidden_dim=config["MLP_HIDDEN_DIM"],
      num_embed_layers=config["NUM_EMBED_LAYERS"],
      num_mlp_layers=config["NUM_MLP_LAYERS"],
      activation=config["ACTIVATION"],
      norm_type=config.get("NORM_TYPE", "none"),
    ),
    rnn=rnn,
    q_fn=DuellingMLP(
      hidden_dim=config.get("Q_HIDDEN_DIM", 512),
      num_layers=config.get("NUM_Q_LAYERS", 1),
      out_dim=env.num_actions(env_params),
      use_bias=config.get("USE_BIAS", True),
    ),
    task_encoder=TaskEncoder(500),
    goal_from_timestep=partial(goal_from_env_timestep, env="jaxmaze"),
  )

  rng, _rng = jax.random.split(rng)
  network_params = agent.init(_rng, example_timestep, method=agent.initialize)

  def reset_fn(params, example_timestep, reset_rng):
    batch_dims = (example_timestep.reward.shape[0],)
    return agent.apply(
      params, batch_dims=batch_dims, rng=reset_rng, method=agent.initialize_carry
    )

  return agent, network_params, reset_fn
