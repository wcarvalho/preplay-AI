
import abc
import collections
from typing import Dict, Union, Optional, Callable, Optional

import jax
import jax.numpy as jnp
import flashbax as fbx
from flax import struct
from flax.training.train_state import TrainState
import numpy as np
import wandb
import matplotlib.pyplot as plt


from visualizer import plot_frames
from jaxneurorl.agents.basics import TimeStep

Number = Union[int, float, np.float32, jnp.float32]


@struct.dataclass
class BasicObserverState:
  episode_returns: jax.Array
  episode_lengths: jax.Array
  finished: jax.Array
  idx: jax.Array = jnp.array(0, dtype=jnp.int32)


def get_first(b): return jax.tree_map(lambda x:x[0], b)

def add_first_to_buffer(
    buffer: fbx.trajectory_buffer.TrajectoryBufferState,
    buffer_state: struct.PyTreeNode,
    x: struct.PyTreeNode):
  """
  x: [num_envs, ...]
  get first env data and dummy dummy time dim.
  """
  x = jax.tree_map(lambda y: y[:1, np.newaxis], x)
  return buffer.add(buffer_state, x)



class Observer:
  """This is an observer that keeps track of timesteps, actions, and predictions.

  It only uses information from the first envionment. Annoying to track each env.

  """
  def __init__(
      self,
      log_period: int = 50_000,
      max_episode_length: int = 200,
      max_num_episodes: int = 200,
      **kwargs,
      ):


    self.log_period = log_period
    self.max_episode_length = max_episode_length
    self.buffer = fbx.make_trajectory_buffer(
        max_length_time_axis=max_episode_length*max_num_episodes,
        min_length_time_axis=1,
        sample_batch_size=1,  # unused
        add_batch_size=1,
        sample_sequence_length=1,  # unused
        period=1,
    )

  def init(self, *args, **kwargs):

    observer_state = BasicObserverState(
      episode_returns=jnp.zeros((self.log_period), dtype=jnp.float32),
      episode_lengths=jnp.zeros((self.log_period), dtype=jnp.int32),
      finished=jnp.zeros((self.log_period), dtype=jnp.int32),
    )
    return observer_state

  def observe_first(
    self,
    observer_state: BasicObserverState,
    first_timestep: TimeStep,
    agent_state: Optional[struct.PyTreeNode] = None,
    ) -> BasicObserverState:
    del first_timestep, agent_state
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

    # only use first env
    first_next_timestep = get_first(next_timestep)

    def advance_episode(os):
      # beginning new episode
      next_idx = os.idx + 1
      return os.replace(
        idx=next_idx,
        finished=os.finished.at[os.idx].add(1),
        episode_lengths=os.episode_lengths.at[next_idx].add(1),
        episode_returns=os.episode_returns.at[next_idx].add(first_next_timestep.reward),
        )

    def update_episode(os):
      # within same episode
      return os.replace(
        episode_lengths=os.episode_lengths.at[os.idx].add(1),
        episode_returns=os.episode_returns.at[os.idx].add(first_next_timestep.reward),

      )

    observer_state = jax.lax.cond(
      first_next_timestep.first(),
      advance_episode,
      update_episode,
      observer_state
    )

    return observer_state