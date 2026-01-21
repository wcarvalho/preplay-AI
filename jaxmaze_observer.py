import abc
import collections
from dataclasses import field
from typing import Dict, Union, Optional, Callable, Optional

import jax
import jax.numpy as jnp
import flashbax as fbx
from flax import struct
from flax.training.train_state import TrainState
import numpy as np
import wandb
import matplotlib.pyplot as plt

from jaxmaze import renderer
from visualizer import plot_frames
from jaxneurorl.agents.basics import TimeStep

Number = Union[int, float, np.float32, jnp.float32]


class Observer(abc.ABC):
  """An interface for collecting metrics/counters from actor and env."""

  @abc.abstractmethod
  def observe_first(self, first_timestep: TimeStep, agent_state: jax.Array) -> None:
    """Observes the initial state and initial time-step.

    Usually state will be all zeros and time-step will be output of reset."""

  @abc.abstractmethod
  def observe(
    self,
    predictions: struct.PyTreeNode,
    action: jax.Array,
    next_timestep: TimeStep,
  ) -> None:
    """Observe state and action that are due to observation of time-step.

    Should be state after previous time-step along"""


@struct.dataclass
class BasicObserverState:
  episode_returns: jax.Array
  episode_lengths: jax.Array
  finished: jax.Array
  action_buffer: fbx.trajectory_buffer.TrajectoryBufferState
  timestep_buffer: fbx.trajectory_buffer.TrajectoryBufferState
  prediction_buffer: fbx.trajectory_buffer.TrajectoryBufferState
  task_info_buffer: fbx.trajectory_buffer.TrajectoryBufferState
  idx: jax.Array = field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))
  episodes: jax.Array = field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))
  env_steps: jax.Array = field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))


def get_first(b):
  return jax.tree_util.tree_map(lambda x: x[0], b)


def add_first_to_buffer(
  buffer: fbx.trajectory_buffer.TrajectoryBufferState,
  buffer_state: struct.PyTreeNode,
  x: struct.PyTreeNode,
):
  """
  x: [num_envs, ...]
  get first env data and dummy dummy time dim.
  """
  x = jax.tree_util.tree_map(lambda y: y[:1, np.newaxis], x)
  return buffer.add(buffer_state, x)


class TaskObserver(Observer):
  """This is an observer that keeps track of timesteps, actions, and predictions.

  It only uses information from the first envionment. Annoying to track each env.

  """

  def __init__(
    self,
    log_period: int = 50_000,
    max_episode_length: int = 200,
    max_num_episodes: int = 200,
    extract_task_info: Callable[[TimeStep], struct.PyTreeNode] = None,
    **kwargs,
  ):
    assert extract_task_info is not None

    self.extract_task_info = extract_task_info

    self.log_period = log_period
    self.max_episode_length = max_episode_length
    self.buffer = fbx.make_trajectory_buffer(
      max_length_time_axis=max_episode_length * max_num_episodes,
      min_length_time_axis=1,
      sample_batch_size=1,  # unused
      add_batch_size=1,
      sample_sequence_length=1,  # unused
      period=1,
    )
    self.task_info_buffer = fbx.make_trajectory_buffer(
      max_length_time_axis=max_num_episodes * 10,
      min_length_time_axis=1,
      sample_batch_size=1,  # unused
      add_batch_size=1,
      sample_sequence_length=1,  # unused
      period=1,
    )

  def init(self, example_timestep, example_action, example_predictions):
    observer_state = BasicObserverState(
      episode_returns=jnp.zeros((self.log_period), dtype=jnp.float32),
      episode_lengths=jnp.zeros((self.log_period), dtype=jnp.int32),
      finished=jnp.zeros((self.log_period), dtype=jnp.int32),
      task_info_buffer=self.task_info_buffer.init(
        get_first(self.extract_task_info(example_timestep))
      ),
      timestep_buffer=self.buffer.init(get_first(example_timestep)),
      action_buffer=self.buffer.init(get_first(example_action)),
      prediction_buffer=self.buffer.init(get_first(example_predictions)),
    )
    return observer_state

  def observe_first(
    self,
    observer_state: BasicObserverState,
    first_timestep: TimeStep,
    agent_state: Optional[struct.PyTreeNode] = None,
  ) -> BasicObserverState:
    del agent_state

    task_info_buffer = add_first_to_buffer(
      buffer=self.buffer,
      buffer_state=observer_state.task_info_buffer,
      x=self.extract_task_info(first_timestep),
    )

    observer_state = observer_state.replace(
      task_info_buffer=task_info_buffer,
      timestep_buffer=add_first_to_buffer(
        self.buffer, observer_state.timestep_buffer, first_timestep
      ),
    )

    return observer_state

  def observe(
    self,
    observer_state: BasicObserverState,
    predictions: struct.PyTreeNode,
    action: jax.Array,
    next_timestep: TimeStep,
    agent_state: Optional[struct.PyTreeNode] = None,
    **kwargs,
  ) -> None:
    """Update log and flush if terminal + log period hit.


    Args:
        agent_state (struct.PyTreeNode): _description_
        predictions (struct.PyTreeNode): _description_
        action (jax.Array): _description_
        next_timestep (TimeStep): _description_
    """
    del agent_state

    # only use first time-step
    first_next_timestep = get_first(next_timestep)

    def advance_episode(os):
      # beginning new episode
      next_idx = os.idx + 1
      task_info_buffer = add_first_to_buffer(
        buffer=self.buffer,
        buffer_state=os.task_info_buffer,
        x=self.extract_task_info(next_timestep),
      )

      return os.replace(
        idx=next_idx,
        task_info_buffer=task_info_buffer,
        finished=os.finished.at[os.idx].add(1),
        episode_lengths=os.episode_lengths.at[next_idx].add(1),
        episode_returns=os.episode_returns.at[next_idx].add(first_next_timestep.reward),
        timestep_buffer=add_first_to_buffer(
          self.buffer, os.timestep_buffer, next_timestep
        ),
        action_buffer=add_first_to_buffer(self.buffer, os.action_buffer, action),
        prediction_buffer=add_first_to_buffer(
          self.buffer, os.prediction_buffer, predictions
        ),
      )

    def update_episode(os):
      # within same episode
      # update return/length information
      idx = os.idx
      # update observer state
      return os.replace(
        episode_lengths=os.episode_lengths.at[idx].add(1),
        episode_returns=os.episode_returns.at[idx].add(first_next_timestep.reward),
        timestep_buffer=add_first_to_buffer(
          self.buffer, os.timestep_buffer, next_timestep
        ),
        action_buffer=add_first_to_buffer(self.buffer, os.action_buffer, action),
        prediction_buffer=add_first_to_buffer(
          self.buffer, os.prediction_buffer, predictions
        ),
      )

    observer_state = jax.lax.cond(
      first_next_timestep.first(), advance_episode, update_episode, observer_state
    )

    return observer_state


def experience_logger(
  train_state: TrainState,
  observer_state: BasicObserverState,
  key: str = "train",
  render_fn: Callable = None,
  log_details_period: int = 0,
  action_names: Optional[dict] = None,
  extract_task_info: Callable[[TimeStep], struct.PyTreeNode] = lambda t: t,
  get_task_name: Callable = lambda t: "Task",
  max_len: int = 40,
  trajectory: Optional[struct.PyTreeNode] = None,
):
  def callback(ts: TrainState, os: BasicObserverState):
    # main
    task_info_buffer = os.task_info_buffer.experience
    len_task_info = max(
      (jax.tree_util.tree_map(lambda x: x.shape[-1], task_info_buffer)).values()
    )
    end = min(os.idx + 1, len(os.episode_lengths), len_task_info)

    # --------------------
    # per-task logging
    # --------------------
    metrics = collections.defaultdict(list)

    return_key = lambda name: f"{key}/0.1 {name.capitalize()} - AvgReturn"
    length_key = lambda name: f"{key}/1.1 {name.capitalize()} - AvgLength"

    for idx in range(end):
      task_info = jax.tree_util.tree_map(
        lambda x: x[0, idx], os.task_info_buffer.experience
      )
      task_name = get_task_name(task_info)

      if os.finished[idx] > 0:
        metrics[return_key(task_name)].append(os.episode_returns[idx])
        metrics[length_key(task_name)].append(os.episode_lengths[idx])

    metrics = {k: np.array(v).mean() for k, v in metrics.items()}
    metrics.update(
      {
        f"{key}/z. avg_episode_length": os.episode_lengths[:end].mean(),
        f"{key}/0.0 avg_episode_return": os.episode_returns[:end].mean(),
        f"{key}/num_actor_steps": ts.timesteps,
        f"{key}/num_learner_updates": ts.n_updates,
      }
    )
    if wandb.run is not None:
      wandb.log(metrics)

    if log_details_period and (int(ts.n_logs) % int(log_details_period) == 0):
      timesteps = jax.tree_util.tree_map(lambda x: x[0], os.timestep_buffer.experience)
      actions = jax.tree_util.tree_map(lambda x: x[0], os.action_buffer.experience)
      # predictions = jax.tree_util.tree_map(lambda x: x[0], os.prediction_buffer.experience)

      # Get maze dimensions and create figure
      maze_height, maze_width, _ = timesteps.state.grid[0].shape
      fig, ax = plt.subplots(1, figsize=(8, 8))

      # Get mask for states within episode (non-terminal)
      non_terminal = timesteps.discount
      is_last = timesteps.last()
      term_cumsum = jnp.cumsum(is_last, -1)
      in_episode = (term_cumsum + non_terminal) < 2

      # Get actions and positions for trajectory
      episode_actions = actions[in_episode][:-1]  # Actions that led to each state
      episode_positions = jax.tree_util.tree_map(
        lambda x: x[in_episode][:-1], timesteps.state.agent_pos
      )

      # Render initial state as background
      initial_state = jax.tree_util.tree_map(lambda x: x[0], timesteps.state)
      img = render_fn(initial_state)

      # Place arrows showing trajectory
      renderer.place_arrows_on_image(
        img,
        episode_positions,
        episode_actions,
        maze_height,
        maze_width,
        arrow_scale=5,
        ax=ax,
      )

      # Add title with task and reward information
      index = lambda t, idx: jax.tree_util.tree_map(lambda x: x[idx], t)
      first_timestep = index(timesteps, 0)
      task_name = get_task_name(extract_task_info(first_timestep))

      total_reward = timesteps.reward[in_episode].sum()

      title = f"{task_name}\nReward: {total_reward:.2f}"
      ax.set_title(title, fontsize=10)
      ax.axis("off")  # Remove axes for cleaner look

      if wandb.run is not None:
        wandb.log({f"{key}_example/trajectory": wandb.Image(fig)})
      plt.close(fig)

  jax.debug.callback(callback, train_state, observer_state)
