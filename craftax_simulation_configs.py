from typing import Tuple, List
from craftax_web_env import CraftaxSymbolicWebEnvNoAutoReset, EnvParams
from craftax_experiment_configs import PATHS_CONFIGS, make_block_env_params
import jax.random
import jax.numpy as jnp
import jax.tree_util as jtu
from flax import struct
import numpy as np
import os
import jax
import craftax_utils

MAX_START_POSITIONS = 5


class TaskConfig(struct.PyTreeNode):
  """Configuration for a single experimental block"""

  world_seed: int
  start_positions: Tuple[int, int]
  goal_object: int = None
  placed_goals: List[Tuple[int, int]] = None
  goal_locations: List[Tuple[int, int]] = None


def make_start_position(start_positions):
  start_position = jnp.zeros((MAX_START_POSITIONS, 2), dtype=jnp.int32)
  return start_position.at[: len(start_positions)].set(jnp.asarray(start_positions))


OPTIMAL_PATHS = {}
env = CraftaxSymbolicWebEnvNoAutoReset()
env_params = EnvParams()

TRAIN_CONFIGS = []
TEST_CONFIGS = []
for i in range(len(PATHS_CONFIGS)):
  config = PATHS_CONFIGS[i]
  # Create cache path
  cache_dir = "craftax_cache/optimal_paths"
  os.makedirs(cache_dir, exist_ok=True)

  block_env_params = make_block_env_params(config, env_params)

  def get_params_and_path(
    world_seed,
    start_position,
    goal_object,
    recompute=True,
  ):
    cache_file = f"path_{world_seed}_{start_position}_{goal_object}.npy"
    cache_file = os.path.join(cache_dir, cache_file)
    params = block_env_params.replace(
      start_positions=make_start_position(start_position),
    )
    if os.path.exists(cache_file) and not recompute:
      path = np.load(cache_file)
    else:
      # Calculate path for each start position + test object location
      obs, state = env.reset(jax.random.PRNGKey(0), params)

      path, _ = craftax_utils.astar(state, goal_object)
      path = np.array(path)
      np.save(cache_file, path)
    return params, path

  def get_path_waypoints(path, num_segments=10):
    indices = np.linspace(0, len(path) - 1, num_segments + 1, dtype=int)
    # Get waypoints at those indices
    return path[indices]

  def add_to_configs(world_seed, start_position, goal_object, configs):
    params, path = get_params_and_path(
      world_seed=world_seed,
      start_position=start_position,
      goal_object=goal_object,
    )
    waypoints = get_path_waypoints(path)
    for waypoint in waypoints:
      configs.append(
        TaskConfig(
          world_seed=world_seed,
          start_positions=waypoint,
          goal_object=goal_object,
          placed_goals=params.placed_goals,
          goal_locations=params.goal_locations,
        )
      )

  for start_position in config.start_eval_positions + config.start_train_positions:
    # first test
    add_to_configs(
      world_seed=config.world_seed,
      start_position=start_position,
      goal_object=config.test_objects[0],
      configs=TEST_CONFIGS,
    )

    # then train main
    add_to_configs(
      world_seed=config.world_seed,
      start_position=start_position,
      goal_object=config.train_objects[0],
      configs=TRAIN_CONFIGS,
    )

    # then train distractor
    add_to_configs(
      world_seed=config.world_seed,
      start_position=start_position,
      goal_object=config.train_objects[1],
      configs=TRAIN_CONFIGS,
    )

TRAIN_CONFIGS = jtu.tree_map(lambda *x: jnp.array(x), *TRAIN_CONFIGS)
TEST_CONFIGS = jtu.tree_map(lambda *x: jnp.array(x), *TEST_CONFIGS)
from pprint import pprint

pprint(TRAIN_CONFIGS)
pprint(TEST_CONFIGS)
