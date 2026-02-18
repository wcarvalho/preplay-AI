"""
Hindsight experience replay


Things to remove:
1. POSITION_GOALS <-- hard-code to always be False. remove GOAL_BETA
2. TERMINATE_ON_REWARD <-- hard-code to always be False.

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
from visualizer import plot_frames
from craftax.craftax.renderer import render_craftax_pixels
from craftax.craftax.constants import BLOCK_PIXEL_SIZE_IMG, Achievement, Action
from jaxmaze import renderer
import functools
from functools import partial
from typing import Callable, NamedTuple, Union, Optional

import matplotlib

matplotlib.use("Agg")

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import rlax
import wandb
from flax import struct
from gymnax.environments import environment
from base_algorithm2 import TimeStep
import base_algorithm2 as base
from jaxneurorl.losses import cql_loss as compute_cql_loss

make_optimizer = base.make_optimizer
make_actor = base.make_actor

Agent = nn.Module
Params = flax.core.FrozenDict

Goal = flax.struct.PyTreeNode
AgentState = flax.struct.PyTreeNode
RNNInput = base.RNNInput


ENVIRONMENT_TO_GOAL_FNS = {
  "jaxmaze": (
    lambda t: jax.lax.stop_gradient(t.observation.task_w),   # task vector
    lambda t: jax.lax.stop_gradient(t.observation.state_features),  # state features
    lambda t: jax.lax.stop_gradient(t.observation.player_position),  # player position
  ),
  "craftax-multigoal": (
    lambda t: jax.lax.stop_gradient(t.observation.task_w),
    lambda t: jax.lax.stop_gradient(t.observation.state_features),
    lambda t: jax.lax.stop_gradient(t.observation.player_position),
  ),
  "craftax-gen": (
    lambda t: jax.lax.stop_gradient(jnp.ones_like(t.observation.achievements)),
    lambda t: jax.lax.stop_gradient(t.observation.achievements),
    lambda t: jax.lax.stop_gradient(t.observation.player_position),
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

    q_vals = jax.vmap(self.q_fn)(rnn_out, goal_embedding)

    return Predictions(
      q_vals=q_vals,
      rnn_states=rnn_out,
    ), new_rnn_state

  def unroll(self, rnn_state, xs: TimeStep, rng: jax.random.PRNGKey):
    """
    Docstring for unroll

    # rnn_state: [B, D]
    # xs: [T, B, ...]
    """

    # observations
    embedding = nn.BatchApply(self.observation_encoder)(xs.observation)

    # RNN
    rng, _rng = jax.random.split(rng)
    rnn_in = RNNInput(obs=embedding, reset=xs.first())
    new_rnn_state, rnn_out = self.rnn.unroll(rnn_state, rnn_in, _rng)

    # Goal
    goal: Goal = self.goal_from_timestep(xs)
    goal_embedding = jax.vmap(jax.vmap(self.task_encoder))(goal)

    # Q values
    q_vals = jax.vmap(jax.vmap(self.q_fn))(rnn_out, goal_embedding)

    return Predictions(
      q_vals=q_vals,
      rnn_states=rnn_out,
    ), new_rnn_state

  def apply_q(self, rnn_out, goal: Goal):
    """
    rnn_out [T, D],
    goal [T, D],
    """
    goal_embedding = jax.vmap(self.task_encoder)(goal)
    q_vals = jax.vmap(self.q_fn)(rnn_out, goal_embedding)

    return Predictions(q_vals=q_vals, rnn_states=rnn_out)

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
  return jax.lax.stop_gradient(make_float(1 - truncated))


def episode_finished_mask(timesteps):
  # either termination or truncation
  is_last_t = make_float(timesteps.last())

  # time-step of termination and everything afterwards is masked out
  term_cumsum_t = jnp.cumsum(is_last_t, 0)
  loss_mask_t = make_float((term_cumsum_t + timesteps.discount) < 2)
  return jax.lax.stop_gradient(loss_mask_t)


@struct.dataclass
class HerLossFn(base.RecurrentLossFn):
  """Loss function of R2D2.

  https://openreview.net/forum?id=r1lyTjAqYX
  """

  extract_q: Callable[[jax.Array], jax.Array] = lambda preds: preds.q_vals
  her_coeff: float = 1.0
  all_goals_coeff: float = 1.0
  max_priority_weight: float = 0.9
  importance_sampling_exponent: float = 0.6
  ngoals: int = 10
  sample_achieved_goals: Callable[[base.TimeStep, jax.random.PRNGKey], jax.Array] = None
  sample_td_goals: Callable[[base.TimeStep, jax.random.PRNGKey], jax.Array] = None
  online_reward_fn: Callable[[base.TimeStep], jax.Array] = None
  her_reward_fn: Callable[[base.TimeStep, Goal], jax.Array] = None
  terminate_on_reward: bool = False
  cql_alpha: float = 0.0
  cql_temp: float = 1.0
  all_goals_lambda: float = 0.3

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
    apply_cql=False,
    lambda_override=None,
  ):
    rewards = make_float(rewards)
    rewards = rewards - self.step_cost
    is_last = make_float(is_last)
    discounts = make_float(non_terminal) * self.discount
    effective_lambda = self.lambda_ if lambda_override is None else lambda_override
    lambda_ = jnp.ones_like(non_terminal) * effective_lambda

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
    assert batch_loss.ndim == 1, "only coded for T dimension"
    batch_loss_mean = (batch_loss * loss_mask[:-1]).sum(0) / (
      loss_mask[:-1].sum(0) + 1e-5
    )

    # CQL regularizer
    cql_loss_val = jnp.zeros((), dtype=batch_loss_mean.dtype)
    if apply_cql and self.cql_alpha > 0:
      cql_per_t = compute_cql_loss(
        online_preds.q_vals[:-1], actions[:-1], self.cql_temp
      )  # [T]
      cql_per_t = cql_per_t * loss_mask[:-1]
      cql_loss_val = (cql_per_t * loss_mask[:-1]).sum(0) / (
        loss_mask[:-1].sum(0) + 1e-5
      )
      cql_loss_val = self.cql_alpha * cql_loss_val
      batch_loss_mean = batch_loss_mean + cql_loss_val

    sorted_q = jnp.sort(online_preds.q_vals, axis=-1)
    q_gap = sorted_q[..., -1] - sorted_q[..., -2]  # [T+1]

    q_softmax = jax.nn.softmax(online_preds.q_vals, axis=-1)
    max_entropy = jnp.log(online_preds.q_vals.shape[-1])
    q_entropy = -jnp.sum(q_softmax * jnp.log(q_softmax + 1e-8), axis=-1) / max_entropy  # [T+1]

    metrics = {
      "0.q_loss": batch_loss.mean(),
      "0.q_td": jnp.abs(batch_td_error).mean(),
      "1.reward": rewards[1:].mean(),
      "z.q_mean": online_preds.q_vals.mean(),
      "z.q_var": online_preds.q_vals.var(),
      "z.q_top2_gap": q_gap.mean(),
      "z.q_entropy": q_entropy.mean(),
      "0.cql_loss": cql_loss_val.mean(),
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
    assert self.sample_achieved_goals is not None, f"Please provide `sample_goals`"

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

    # [T, B, ...] reward info
    online_reward_info = self.online_reward_fn(data.timestep)

    # [B, T-1], [T], [B, T], [B, T, D]
    td_error, loss, metrics, log_info = jax.vmap(self.loss_fn, 1, 0)(
      data.timestep,
      online_preds,
      target_preds,
      data.action,
      online_reward_info["reward"],
      make_float(data.timestep.last()),
      data.timestep.discount,
      loss_mask,
    )
    # [B, T] --> [T, B]
    td_error = jnp.swapaxes(td_error, 0, 1)

    swap = lambda x: jnp.swapaxes(x, 0, 1)
    log_info["reward_info"] = jax.tree_util.tree_map(swap, online_reward_info)

    # first label online loss with online
    all_metrics.update({f"0.online/{k}": v for k, v in metrics.items()})
    all_log_info["online"] = log_info

    if self.her_coeff > 0:
      ##################
      # Q-learning on fake goals
      ##################
      key_grad = jax.random.split(key_grad, B + 1)
      # 1. sample fake goals from trajectory using data.timestep

      # [N, B, D], [T, B], [N, B]
      new_goals, logits, goal_indices = jax.vmap(self.sample_achieved_goals, (1, 0), 1)(
        data.timestep,  # [T, B]
        key_grad[1:],  # [B, 2],
      )

      # 2. Compute Q-values for new goals
      def her_loss_fn(
        new_goal: Union[
          jnp.ndarray,
          struct.PyTreeNode,
        ],
        goal_index,
        sub_key_grad,
        timestep,
        actions,
        goal_logits,
        online_preds,
        target_preds,
      ):
        """new_goal is [D], goal_index is scalar, sub_key_grad is [2], ..."""

        sub_key_grad, sub_key_grad_ = jax.random.split(sub_key_grad)
        length = len(data.action)

        # [D] --> [T, D]
        expand_over_time = lambda x: jnp.tile(x[None], [length, 1])
        expanded_new_goal = jax.tree_util.tree_map(expand_over_time, new_goal)

        apply_q = functools.partial(self.network.apply, method=self.network.apply_q)
        online_preds = apply_q(params, online_preds.rnn_states, expanded_new_goal)
        target_preds = apply_q(
          target_params, target_preds.rnn_states, expanded_new_goal
        )

        her_reward_info = self.her_reward_fn(timestep, new_goal)

        # Compute per-goal episode mask
        episode_ids = jnp.cumsum(make_float(timestep.first()), axis=0) - 1  # [T]
        goal_episode_id = jax.lax.dynamic_index_in_dim(
          episode_ids, goal_index, keepdims=False
        )  # scalar
        episode_mask = (episode_ids == goal_episode_id).astype(jnp.float32)  # [T]

        # If terminate_on_reward: mask out timesteps AFTER goal_index
        if self.terminate_on_reward:
          time_indices = jnp.arange(len(timestep.reward))
          terminate_mask = (time_indices <= goal_index).astype(jnp.float32)
          episode_mask = episode_mask * terminate_mask

        loss_mask = episode_mask * is_truncated(timestep)  # [T]

        # Create modified discount that's 0 at goal_index (episode terminates there)
        if self.terminate_on_reward:
          goal_mask = jax.nn.one_hot(goal_index, len(timestep.discount))
          modified_discount = timestep.discount * (1.0 - goal_mask)
        else:
          modified_discount = timestep.discount

        td_error, batch_loss, metrics, log_info = self.loss_fn(
          timestep=timestep,
          online_preds=online_preds,
          target_preds=target_preds,
          actions=actions,
          rewards=her_reward_info["reward"],
          is_last=make_float(timestep.last()),
          non_terminal=modified_discount,
          loss_mask=loss_mask,
        )
        achieved_something = (goal_logits.sum() > 1e-5).astype(batch_loss.dtype)
        td_error = achieved_something * td_error
        batch_loss = achieved_something * batch_loss

        metrics = jax.tree_util.tree_map(lambda x: x * achieved_something, metrics)
        metrics["achieved_something"] = achieved_something

        log_info["reward_info"] = her_reward_info  # [T, ...]
        log_info["goal_index"] = goal_index
        log_info["goal_episode_id"] = goal_episode_id
        return td_error, batch_loss, metrics, log_info

      her_loss_fn = jax.vmap(her_loss_fn, (0, 0, 0, None, None, None, None, None))  # N
      her_loss_fn = jax.vmap(her_loss_fn, 1, 0)  # B

      key_grad, key_grad_ = jax.random.split(key_grad[0])
      key_grad_ = jax.random.split(key_grad_, self.ngoals * B).reshape(
        self.ngoals, B, -1
      )

      # [B, N, T], [B, N], [B, N, T], [B, N, T, D]
      _, her_loss, her_metrics, her_log_info = her_loss_fn(
        new_goals,  # [N, B, D]
        goal_indices,  # [N, B]
        key_grad_,  # [N, B, 2]
        data.timestep,  # [T, B, D]
        data.action,  # [T, B]
        logits,  # [T, B]
        online_preds,  # [T, B, D]
        target_preds,  # [T, B, D]
      )

      loss = loss + self.her_coeff * her_loss.mean(1)
      all_metrics.update({f"1.her/{k}": v for k, v in her_metrics.items()})
      all_log_info["her"] = jax.tree_util.tree_map(
        # only first goal
        lambda x: x[:, 0],
        her_log_info,
      )

    if self.all_goals_coeff > 0.0:
      # Get all possible goals (one-hot for each achievement type)
      # Call once with first batch element since goals are batch-independent
      key_grad, key_grad_ = jax.random.split(key_grad)

      key_grad_ = jax.random.split(key_grad_, B + 1)

      # GoalPosition with [N, B, D] and [N, B, 2]
      all_goals = jax.vmap(self.sample_td_goals, (1, 0), 1)(
        data.timestep, key_grad_[1:]
      )

      def all_goals_loss_fn(
        new_goal,
        timestep,
        actions,
        online_preds,
        target_preds,
      ):
        """new_goal: GoalPosition with scalar fields [D], timestep: [T, ...]"""
        length = len(data.action)
        expand = lambda x: jnp.tile(x[None], [length, 1])
        expanded_goal = jax.tree_util.tree_map(expand, new_goal)

        apply_q = functools.partial(self.network.apply, method=self.network.apply_q)
        on_preds = apply_q(params, online_preds.rnn_states, expanded_goal)
        tar_preds = apply_q(target_params, target_preds.rnn_states, expanded_goal)

        reward_info = self.her_reward_fn(timestep, new_goal)

        # Apply loss blindly — no achieved_something masking,
        # no terminate_on_reward, no episode-based goal masking
        ag_td_error, ag_batch_loss, ag_metrics, ag_log_info = self.loss_fn(
          timestep=timestep,
          online_preds=on_preds,
          target_preds=tar_preds,
          actions=actions,
          rewards=reward_info["reward"],
          is_last=make_float(timestep.last()),
          non_terminal=timestep.discount,
          loss_mask=is_truncated(timestep),
          apply_cql=True,
          lambda_override=self.all_goals_lambda,
        )
        return ag_td_error, ag_batch_loss, ag_metrics, ag_log_info

      # vmap over N goals, then over B batch
      ag_fn = jax.vmap(all_goals_loss_fn, (0, None, None, None, None))  # N goals
      ag_fn = jax.vmap(ag_fn, 1, 0)  # B batch (goals shared across batch)

      # , [B, N]
      _, ag_loss, ag_metrics, _ = ag_fn(
        all_goals,  # GoalPosition [N, D]
        data.timestep,  # [T, B, ...]
        data.action,  # [T, B]
        online_preds,  # tree with [T, B, ...]
        target_preds,  # tree with [T, B, ...]
      )

      # ag_loss: [B, N] -> mean over N goals
      loss = loss + self.all_goals_coeff * ag_loss.mean(1)
      all_metrics.update({f"2.all_goals/{k}": v for k, v in ag_metrics.items()})

    if self.logger.learner_log_extra is not None:
      self.logger.learner_log_extra(all_log_info)

    return td_error, loss, all_metrics


############################################
# Environment Specific
############################################


def make_loss_fn_class(config) -> base.RecurrentLossFn:
  def online_reward_fn(timesteps):
    task_vector_fn, achievement_fn, position_fn = ENVIRONMENT_TO_GOAL_FNS[config["ENV"]]
    task_vector = task_vector_fn(timesteps).astype(jnp.float32)
    achievements = achievement_fn(timesteps).astype(jnp.float32)
    goal_reward = (task_vector * achievements).sum(-1)

    reward = goal_reward
    reward = jax.tree_util.tree_map(jax.lax.stop_gradient, reward)
    return {
      "reward": reward,
      "goal_reward": goal_reward,
      "goal_task_vector": task_vector,
      "goal_achievements": achievements,
    }

  def her_reward_fn(timesteps, new_goal):
    task_vector_fn, achievement_fn, position_fn = ENVIRONMENT_TO_GOAL_FNS[config["ENV"]]

    # achievement reward
    goal_task_vector = new_goal.goal  # [D]
    goal_achievements = achievement_fn(timesteps)  # [T, D]
    goal_reward = (goal_task_vector[None] * goal_achievements).sum(-1)  # [T]

    # position reward
    position = new_goal.position - 1  # [2]
    position_achievement = position_fn(timesteps)  # [T, 2]
    position_reward = jax.vmap(jnp.equal, (None, 0))(position, position_achievement)
    position_reward = position_reward.all(-1).astype(jnp.float32)

    assert goal_reward.ndim == 1
    assert position_reward.ndim == 1
    if config["POSITION_GOALS"]:
      reward = 0.5 * position_reward + 0.5 * goal_reward
    else:
      position_reward = position_reward * 0.0
      reward = goal_reward

    reward = jax.tree_util.tree_map(jax.lax.stop_gradient, reward)

    T = goal_achievements.shape[0]
    goal_task_vector = jnp.tile(goal_task_vector[None], [T, 1])  # [T, D]
    position = jnp.tile(position[None], [T, 1])  # [T, 2]

    return {
      "reward": reward,
      "goal_reward": goal_reward,
      "goal_task_vector": goal_task_vector,
      "goal_achievements": goal_achievements,
      "position_reward": position_reward,
      "position_task_vector": position,
      "position_achievement": position_achievement,
    }

  def sample_achieved_goals(timesteps, rng):
    """
    There is data at T time-points. We want to"""
    # T, C
    task_vector_fn, achievement_fn, position_fn = ENVIRONMENT_TO_GOAL_FNS[config["ENV"]]

    achievements = achievement_fn(timesteps)
    # T, if vector non-zero, achieved
    goal_achieved = achievements.sum(-1)

    # all positions were achieved
    position_achieved = jnp.ones_like(goal_achieved)

    # T
    if config["POSITION_GOALS"]:
      N = config["NUM_HER_GOALS"]
      n_pos = N // 2
      n_ach = N - n_pos

      rng, rng1, rng2 = jax.random.split(rng, 3)

      # position goals: uniform sampling
      uniform_probs = jnp.ones(goal_achieved.shape[0]) / goal_achieved.shape[0]
      pos_indices = Categorical(probs=uniform_probs).sample(
        seed=rng1, sample_shape=(n_pos,)
      )

      # achievement goals: weighted by goal_achieved
      ach_logits = goal_achieved + 1e-5
      ach_probs = ach_logits / ach_logits.sum()
      ach_indices = Categorical(probs=ach_probs).sample(
        seed=rng2, sample_shape=(n_ach,)
      )

      indices = jnp.concatenate([pos_indices, ach_indices])
      logits = ach_logits + uniform_probs
      logits = logits / logits.sum(-1)
    else:
      logits = goal_achieved
      logits_ = logits + 1e-5
      probabilities = logits_ / logits_.sum(-1)

      # N
      rng, rng_ = jax.random.split(rng)
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
    if config["POSITION_GOALS"]:
      # reserve 0 for empty position
      position_goal = jax.lax.stop_gradient(positions_at_indices) + 1
    else:
      position_goal = jnp.zeros_like(positions_at_indices)
    goal = GoalPosition(
      jax.lax.stop_gradient(features_at_indices),
      position_goal,
    )

    # [N, D], [T], [N]
    return goal, logits, indices

  def get_all_goals(timesteps, rng):
    _, achievement_fn, position_fn = ENVIRONMENT_TO_GOAL_FNS[config["ENV"]]
    achievements = achievement_fn(timesteps)  # [T, D]
    D = achievements.shape[-1]
    # Each goal is a one-hot: pursue one achievement at a time
    goal_features = jnp.eye(D)  # [N, D] where N=D
    if config.get("POSITION_GOALS", False):
      obj_pos = timesteps.observation.object_positions[0]  # [D, 2]
      position_goal = obj_pos + 1  # reserve 0 for empty
    else:
      position_goal = jnp.zeros((D, 2))
    return GoalPosition(
      goal=goal_features, position=position_goal.astype(position_fn(timesteps).dtype)
    )

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
    her_coeff=config.get("HER_COEFF", 0.001),
    ngoals=config.get("NUM_HER_GOALS", 10),
    sample_achieved_goals=sample_achieved_goals,
    sample_td_goals=get_all_goals,
    online_reward_fn=online_reward_fn,
    her_reward_fn=her_reward_fn,
    terminate_on_reward=config.get("TERMINATE_ON_REWARD", True),
    all_goals_coeff=config.get("ALL_GOALS_COEFF", 1.0),
    all_goals_lambda=config.get("ALL_GOALS_LAMBDA", 0.3),
    cql_alpha=config.get("CQL_ALPHA", 0.0),
    cql_temp=config.get("CQL_TEMP", 1.0),
  )


@jax.jit
def craftax_render_fn(state):
  image = render_craftax_pixels(state, block_pixel_size=BLOCK_PIXEL_SIZE_IMG)
  return image / 255.0


def create_data_plots(d_, reward_info, is_her=False, subfig=None, goal_index=-1):
  """Create reward computation and training metrics plots.

  Args:
    d_: Dictionary with processed data (batch dimension already removed)
    reward_info: Dictionary with reward computation info (all values [T, ...])
    is_her: If True, plot both goal and position columns (2-col layout)
    subfig: Optional matplotlib SubFigure to plot into. If None, creates a new figure.

  Returns:
    (fig1, axes1, fig2, axes2): The two figures and their axes arrays.
  """

  data_rewards = d_["timesteps"].reward
  discounts = d_["timesteps"].discount
  actions = d_["actions"]
  q_values = d_["q_values"]
  q_target = d_["q_target"]
  q_values_taken = rlax.batched_index(q_values, actions)
  td_errors = d_["td_errors"]

  # Figure dimensions
  width = 0.3
  nT = len(data_rewards)
  height = 3
  fig_width = max(2, int(width * nT))
  ##############################
  # Plot 1: Reward Computation
  ##############################
  n_rows = 3
  n_cols = 2 if is_her else 1
  if subfig is not None:
    fig1 = subfig
    axes1 = subfig.subplots(n_rows, n_cols, sharex="col", gridspec_kw={"wspace": 0.3})
  else:
    fig1, axes1 = plt.subplots(
      n_rows,
      n_cols,
      figsize=(fig_width * n_cols, height * n_rows),
      sharex="col",
      gridspec_kw={"wspace": 0.3},
    )
  if n_cols == 1:
    axes1 = axes1[:, None]  # make 2D for uniform indexing

  loss_mask = d_.get("loss_mask")

  # Column 0: goal reward info
  ax = axes1[0, 0]
  ax.plot(reward_info["goal_reward"], label="Goal Reward")
  if loss_mask is not None:
    ax.plot(loss_mask, label="Loss Mask", alpha=0.5, linestyle="--")
  ax.set_title("Goal Reward")
  ax.legend()
  ax.grid(True)
  ax.set_xticks(range(nT))

  ax = axes1[1, 0]
  goal_tv = reward_info["goal_task_vector"].T  # [D, T]
  ax.imshow(goal_tv, aspect="auto", cmap="viridis", interpolation="nearest")
  ax.set_title("Goal Task Vector")
  ax.set_ylabel("Dims")
  ax.set_yticks(range(goal_tv.shape[0]))
  ax.set_xticks(range(nT))
  ax.grid(True, axis="x", alpha=0.3, color="white")

  ax = axes1[2, 0]
  goal_ach = reward_info["goal_achievements"].T  # [D, T]
  ax.imshow(goal_ach, aspect="auto", cmap="viridis", interpolation="nearest")
  ax.set_title("Goal Achievements")
  ax.set_ylabel("Dims")
  ax.set_yticks(range(goal_ach.shape[0]))
  ax.set_xticks(range(nT))
  ax.grid(True, axis="x", alpha=0.3, color="white")
  ax.set_xlabel("Time")

  if is_her:
    # Column 1: position reward info
    ax = axes1[0, 1]
    ax.plot(reward_info["position_reward"], label="Position Reward")
    if loss_mask is not None:
      ax.plot(loss_mask, label="Loss Mask", alpha=0.5, linestyle="--")
    ax.set_title("Position Reward")
    ax.legend()
    ax.grid(True)
    ax.set_xticks(range(nT))

    # Position Task Vector - show integer values as text
    ax = axes1[1, 1]
    pos_tv = reward_info["position_task_vector"]  # [T, 2]
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 1])
    ax.set_xlim(-0.5, nT - 0.5)
    for t in range(len(pos_tv)):
      for d in range(pos_tv.shape[1]):
        ax.text(t, d, str(int(pos_tv[t, d])), ha="center", va="center", fontsize=14)
    ax.set_title("Position Task Vector")
    ax.set_ylabel("Dims")
    ax.set_xticks(range(nT))
    ax.grid(True, axis="x", alpha=0.3)

    # Position Achievement - show integer values as text
    ax = axes1[2, 1]
    pos_ach = reward_info["position_achievement"]  # [T, 2]
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 1])
    ax.set_xlim(-0.5, nT - 0.5)
    for t in range(len(pos_ach)):
      for d in range(pos_ach.shape[1]):
        ax.text(t, d, str(int(pos_ach[t, d])), ha="center", va="center", fontsize=14)
    ax.set_title("Position Achievement")
    ax.set_ylabel("Dims")
    ax.set_xticks(range(nT))
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlabel("Time")

  # Add vertical line at goal index (top row only)
  if goal_index >= 0:
    for col in range(n_cols):
      axes1[0, col].axvline(
        goal_index, color="red", linestyle="--", alpha=0.7, label="Goal idx"
      )

  if subfig is None:
    fig1.tight_layout()

  ##############################
  # Plot 2: Training Metrics
  ##############################
  fig2, axes2 = plt.subplots(3, 1, figsize=(fig_width, height * 3), sharex=True)
  ax2a, ax2b, ax2c = axes2[0], axes2[1], axes2[2]

  # Rewards, Q-values, Q-targets
  ax2a.plot(reward_info["goal_reward"], label="Goal Reward")
  ax2a.plot(q_values_taken, label="Q-Values")
  ax2a.plot(q_target, label="Q-Targets")
  ax2a.set_title("Rewards and Q-Values")
  ax2a.legend()
  ax2a.grid(True)
  ax2a.set_xticks(range(nT))

  # TD errors
  ax2b.plot(td_errors)
  ax2b.set_title("TD Errors")
  ax2b.grid(True)
  ax2b.set_xticks(range(nT))

  # Add red vertical lines at loss_mask boundaries
  if loss_mask is not None:
    mask_indices = jnp.where(loss_mask == 1)[0]
    if len(mask_indices) > 0:
      left_boundary = int(mask_indices[0])
      right_boundary = int(mask_indices[-1])
      for ax in [ax2a, ax2b]:
        ax.axvline(left_boundary, color="red", linestyle="--", alpha=0.7, linewidth=2)
        ax.axvline(right_boundary, color="red", linestyle="--", alpha=0.7, linewidth=2)

  # Episode markers
  is_last = d_["timesteps"].last()
  ax2c.plot(discounts, label="Discounts")
  ax2c.plot(is_last, label="is_last")
  if loss_mask is not None:
    ax2c.plot(loss_mask, label="Loss Mask", alpha=0.5, linestyle="--")
  ax2c.set_title("Episode Markers")
  ax2c.legend()
  ax2c.grid(True)
  ax2c.set_xticks(range(nT))
  ax2c.set_xlabel("Time")

  fig2.tight_layout()

  return fig1, axes1, fig2, axes2


def jaxmaze_learner_log_fn(
  data: dict,
  config: dict,
  action_names: dict,
  render_fn: Callable,
  extract_task_info: Callable[[TimeStep], flax.struct.PyTreeNode] = lambda t: t,
  get_task_name: Callable = lambda t: "Task",
):
  def plot_individual(d, setting: str):
    from math import ceil

    is_her = setting == "her"
    # [B, T, ...] --> [T, ...]
    d_ = jax.tree_util.tree_map(lambda x: x[0], d)
    reward_info = d_["reward_info"]

    # Compute image sequence info first for layout
    timesteps: TimeStep = d_["timesteps"]
    nT = len(timesteps.reward)
    n_episode_steps = min(nT, config.get("MAX_EPISODE_LOG_LEN", 40))

    ncols_img = 5
    n_image_rows = ceil(n_episode_steps / ncols_img)

    # Create combined figure with subfigures
    n_cols = 2 if is_her else 1
    width = 0.3
    height = 2
    fig_width = max(2, int(width * nT))

    # Use maze dimensions to compute tight image cell sizes
    maze_height, maze_width, _ = timesteps.state.grid[0].shape
    img_cell_height = maze_height / 10.0  # scale down for figure inches
    img_cell_width = maze_width / 10.0
    row_height_img = img_cell_height * 2
    top_height = height * 3
    bottom_height = row_height_img * n_image_rows

    fig_combined = plt.figure(
      figsize=(int(fig_width * n_cols * 3 / 4), top_height + bottom_height)
    )
    subfigs = fig_combined.subfigures(
      2, 1, height_ratios=[top_height, bottom_height], hspace=0.02
    )

    # Top: reward computation plots
    goal_index = int(d_.get("goal_index", -1))
    _, axes1, fig2, axes2 = create_data_plots(
      d_, reward_info, is_her=is_her, subfig=subfigs[0], goal_index=goal_index
    )

    # Bottom: image sequence (tight spacing based on maze dimensions)
    axes_img = subfigs[1].subplots(
      n_image_rows, ncols_img, gridspec_kw={"hspace": 0.3, "wspace": 0.05}
    )
    if n_image_rows == 1:
      axes_img = axes_img[None, :]  # ensure 2D

    for idx in range(n_episode_steps):
      row, col = divmod(idx, ncols_img)
      ax = axes_img[row, col]
      state_at_t = jax.tree_util.tree_map(lambda x: x[idx], timesteps.state)
      img = render_fn(state_at_t)
      ax.imshow(img)
      # Add black border around first timestep
      if timesteps.first()[idx]:
        rect = Rectangle(
          (0, 0),
          img.shape[1],
          img.shape[0],
          linewidth=10,
          edgecolor="black",
          facecolor="none",
        )
        ax.add_patch(rect)

      # Add colored borders based on reward and state_features
      reward = float(timesteps.reward[idx])
      her_reward = float(reward_info["reward"][idx])
      features = timesteps.observation.state_features[idx]

      # Determine border color (YELLOW for mismatch is highest priority)
      rewards_mismatch = abs(reward - her_reward) > 1e-5
      has_reward = reward > 0
      has_features = features.sum() > 0

      if has_reward:
        color = "red"
        add_border = True
      elif has_features:
        color = "blue"
        add_border = True
      else:
        add_border = False

      if add_border:
        rect = Rectangle(
          (0, 0),
          img.shape[1],
          img.shape[0],
          linewidth=6,
          edgecolor=color,
          facecolor="none",
        )
        ax.add_patch(rect)

      ax.set_title(f"t={idx}, r_t={reward:.1f}, r_h={her_reward:.1f}", fontsize=7)
      ax.axis("off")
    # Hide unused subplots
    for idx in range(n_episode_steps, n_image_rows * ncols_img):
      row, col = divmod(idx, ncols_img)
      axes_img[row, col].axis("off")

    subfigs[1].subplots_adjust(hspace=0.05, wspace=0.05)
    fig_combined.tight_layout(pad=0.5)

    if wandb.run is not wandb.sdk.lib.disabled.RunDisabled:
      wandb.log(
        {
          f"learner_example/{setting}/reward_computation": wandb.Image(fig_combined),
          f"learner_example/{setting}/training_metrics": wandb.Image(fig2),
        }
      )
    plt.close(fig_combined)
    plt.close(fig2)

  # this will be the value after update is applied
  n_updates = data["n_updates"] + 1
  is_log_time = n_updates % config["LEARNER_EXTRA_LOG_PERIOD"] == 0

  def plot_both(d):
    # plot_individual(d["online"], "online")
    plot_individual(d["her"], "her")

  jax.lax.cond(
    is_log_time,
    lambda d: jax.debug.callback(plot_both, d),
    lambda d: None,
    data,
  )


def crafax_learner_log_fn(data: dict, config: dict):
  def plot_individual(d, setting: str):
    from math import ceil

    is_her = setting == "her"
    # [B, T, ...] --> [T, ...]
    d_ = jax.tree_util.tree_map(lambda x: x[0], d)
    reward_info = d_["reward_info"]

    # Compute image sequence info first for layout
    timesteps: TimeStep = d_["timesteps"]
    nT = len(timesteps.reward)
    n_episode_steps = min(nT, config.get("MAX_EPISODE_LOG_LEN", 40))
    actions = d_["actions"]
    actions_taken = [Action(a).name for a in actions]

    ncols_img = 5
    n_image_rows = ceil(n_episode_steps / ncols_img)

    # Create combined figure with subfigures
    n_cols = 2 if is_her else 1
    width = 0.3
    height = 2
    fig_width = max(2, int(width * nT))

    # Use rendered image dimensions to compute tight cell sizes
    img_cell_height = BLOCK_PIXEL_SIZE_IMG * 9 / 80.0  # approximate craftax grid height
    row_height_img = img_cell_height * 2
    top_height = height * 3
    bottom_height = row_height_img * n_image_rows

    fig_combined = plt.figure(
      figsize=(int(fig_width * n_cols * 3 / 4), top_height + bottom_height)
    )
    subfigs = fig_combined.subfigures(
      2, 1, height_ratios=[top_height, bottom_height], hspace=0.02
    )

    # Top: reward computation plots
    goal_index = int(d_.get("goal_index", -1))
    _, axes1, fig2, axes2 = create_data_plots(
      d_, reward_info, is_her=is_her, subfig=subfigs[0], goal_index=goal_index
    )

    # Bottom: image sequence (tight spacing)
    axes_img = subfigs[1].subplots(
      n_image_rows, ncols_img, gridspec_kw={"hspace": 0.3, "wspace": 0.05}
    )
    if n_image_rows == 1:
      axes_img = axes_img[None, :]  # ensure 2D
    for idx in range(n_episode_steps):
      row, col = divmod(idx, ncols_img)
      ax = axes_img[row, col]
      state_at_t = jax.tree_util.tree_map(lambda x: x[idx], timesteps.state.env_state)
      img = craftax_render_fn(state_at_t)
      ax.imshow(img)

      # Add black border around first timestep
      if timesteps.first()[idx]:
        rect = Rectangle(
          (0, 0),
          img.shape[1],
          img.shape[0],
          linewidth=10,
          edgecolor="black",
          facecolor="none",
        )
        ax.add_patch(rect)

      # Add colored borders based on reward and state_features
      reward = float(timesteps.reward[idx])
      her_reward = float(reward_info["reward"][idx])
      features = timesteps.observation.state_features[idx]

      # Determine border color (YELLOW for mismatch is highest priority)
      rewards_mismatch = abs(reward - her_reward) > 1e-5
      has_reward = reward > 0
      has_features = features.sum() > 0

      if rewards_mismatch:
        color = "yellow"
        add_border = True
      elif has_reward:
        color = "red"
        add_border = True
      elif has_features:
        color = "blue"
        add_border = True
      else:
        add_border = False

      if add_border:
        rect = Rectangle(
          (0, 0),
          img.shape[1],
          img.shape[0],
          linewidth=6,
          edgecolor=color,
          facecolor="none",
        )
        ax.add_patch(rect)

      ax.set_title(
        f"t={idx}, r_t={reward:.1f}, r_h={her_reward:.1f}\n{actions_taken[idx]}",
        fontsize=7,
      )
      ax.axis("off")
    # Hide unused subplots
    for idx in range(n_episode_steps, n_image_rows * ncols_img):
      row, col = divmod(idx, ncols_img)
      axes_img[row, col].axis("off")

    subfigs[1].subplots_adjust(hspace=0.05, wspace=0.05)
    fig_combined.tight_layout(pad=0.5)

    if wandb.run is not wandb.sdk.lib.disabled.RunDisabled:
      wandb.log(
        {
          f"learner_example/{setting}/reward_computation": wandb.Image(fig_combined),
          f"learner_example/{setting}/training_metrics": wandb.Image(fig2),
        }
      )
    plt.close(fig_combined)
    plt.close(fig2)

  def plot_both(d):
    # plot_individual(d["online"], "online")
    plot_individual(d["her"], "her")

  # this will be the value after update is applied
  n_updates = data["n_updates"] + 1
  is_log_time = n_updates % config["LEARNER_EXTRA_LOG_PERIOD"] == 0

  jax.lax.cond(
    is_log_time,
    lambda d: jax.debug.callback(plot_both, d),
    lambda d: None,
    data,
  )


class DuellingDotMLP(nn.Module):
  hidden_dim: int
  num_actions: int = 0
  num_layers: int = 1
  norm_type: str = "none"
  activation: str = "relu"
  activate_final: bool = True
  use_bias: bool = False

  @nn.compact
  def __call__(self, x, task, train: bool = False):
    task_dim = task.shape[-1]
    x = jnp.concatenate((x, task), axis=-1)
    value_mlp = base.MLP(
      hidden_dim=self.hidden_dim,
      num_layers=self.num_layers,
      use_bias=self.use_bias,
      out_dim=task_dim,
    )
    advantage_mlp = base.MLP(
      hidden_dim=self.hidden_dim,
      num_layers=self.num_layers,
      use_bias=self.use_bias,
      out_dim=self.num_actions * task_dim,
    )
    assert self.num_actions > 0, "must have at least one action"

    value = value_mlp(x)  # [C]
    advantages = advantage_mlp(x)  # [A*C]
    advantages = advantages.reshape(self.num_actions, task_dim)  # [A, C]

    # Advantages have zero mean across actions.
    advantages -= jnp.mean(advantages, axis=0, keepdims=True)  # [A, C]

    sf = value[None, :] + advantages  # [A, C]
    q_values = (sf * task[None, :]).sum(-1)  # [A]

    return q_values


class DotMLP(nn.Module):
  hidden_dim: int
  num_actions: int = 0
  num_layers: int = 1
  norm_type: str = "none"
  activation: str = "relu"
  activate_final: bool = True
  use_bias: bool = False

  @nn.compact
  def __call__(self, x, task, train: bool = False):
    task_dim = task.shape[-1]
    mlp = base.MLP(
      hidden_dim=self.hidden_dim,
      num_layers=self.num_layers,
      use_bias=self.use_bias,
      out_dim=self.num_actions * task_dim,
    )
    assert self.num_actions > 0, "must have at least one action"

    sf = sf.reshape(self.num_actions, task_dim)  # [A, C]

    q_values = (sf * task[None, :]).sum(-1)  # [A]

    return q_values


# --------------------
# Craftax
# --------------------


class TaskEncoder(nn.Module):
  """Task encoder for regular craftax setting"""

  vocab_size: int = 512

  @nn.compact
  def __call__(self, goal: GoalPosition):
    # task is binary
    kernel_init = nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)
    # [D]
    task = nn.Dense(128, kernel_init=kernel_init, use_bias=False)(goal.goal)

    # position is categorical
    # [2, D]
    position = jax.vmap(flax.linen.Embed(self.vocab_size, 128))(goal.position)
    return jnp.concatenate((task, position.reshape(-1)))


def goal_from_env_timestep(t: TimeStep, env: str, position_goals: bool = False):
  task_vector_fn, _, position_fn = ENVIRONMENT_TO_GOAL_FNS[env]
  task_w = task_vector_fn(t).astype(jnp.float32)
  if position_goals:
    position = (task_w[..., None] * t.observation.object_positions).sum(-2) + 1
  else:
    position = jnp.zeros_like(position_fn(t))
  return GoalPosition(task_w, position)


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
    q_fn=DuellingDotMLP(
      hidden_dim=config.get("Q_HIDDEN_DIM", 512),
      num_layers=config.get("NUM_Q_LAYERS", 2),
      num_actions=env.action_space(env_params).n,
      use_bias=config.get("USE_BIAS", True),
    ),
    task_encoder=TaskEncoder(100),
    goal_from_timestep=partial(
      goal_from_env_timestep,
      env=config["ENV"],
      position_goals=config.get("POSITION_GOALS", False),
    ),
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
    task_encoder=TaskEncoder(100),
    rnn=rnn,
    q_fn=DuellingDotMLP(
      hidden_dim=config.get("Q_HIDDEN_DIM", 512),
      num_layers=config.get("NUM_Q_LAYERS", 2),
      num_actions=env.action_space(env_params).n,
      use_bias=config.get("USE_BIAS", True),
    ),
    goal_from_timestep=partial(
      goal_from_env_timestep,
      env=config["ENV"],
      position_goals=config.get("POSITION_GOALS", False),
    ),
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
    q_fn=DuellingDotMLP(
      hidden_dim=config.get("Q_HIDDEN_DIM", 512),
      num_layers=config.get("NUM_Q_LAYERS", 1),
      num_actions=env.num_actions(env_params),
      use_bias=config.get("USE_BIAS", False),
    ),
    task_encoder=TaskEncoder(100),
    goal_from_timestep=partial(
      goal_from_env_timestep,
      env="jaxmaze",
      position_goals=config.get("POSITION_GOALS", False),
    ),
  )

  rng, _rng = jax.random.split(rng)
  network_params = agent.init(_rng, example_timestep, method=agent.initialize)

  def reset_fn(params, example_timestep, reset_rng):
    batch_dims = (example_timestep.reward.shape[0],)
    return agent.apply(
      params, batch_dims=batch_dims, rng=reset_rng, method=agent.initialize_carry
    )

  return agent, network_params, reset_fn
