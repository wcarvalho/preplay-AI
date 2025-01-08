"""
Based off of: craftax.craftax.envs.craftax_symbolic_env.CraftaxSymbolicEnvNoAutoReset

Changes:
1. Added a goal-conditioned reward
2. Environment starts with 2 crafting tables (to make it easier for people to have multiple tasks)
3. Humans start with 20 strength to more easily kill enemies
4. structured obs with {image, task-vector}
5. sample world set from pre-defined set of seeds
6. goal-conditioned reward
"""
import jax
from jax.tree_util import tree_map
from jax import lax
from gymnax.environments import spaces, environment
from typing import Tuple, Optional
import chex
from flax import struct

from craftax.craftax.envs.common import log_achievements_to_info
from craftax.environment_base.environment_bases import EnvironmentNoAutoReset
from craftax.craftax.constants import *
from craftax.craftax.game_logic import craftax_step, is_game_over
from craftax.craftax.craftax_state import EnvState, EnvParams, StaticEnvParams
from craftax.craftax.renderer import render_craftax_symbolic
from craftax.craftax.world_gen.world_gen import generate_world


# NOTE: slight hack to import all game logic functions
from craftax.craftax.world_gen.world_gen import generate_world, generate_smoothworld, generate_dungeon, ALL_SMOOTHGEN_CONFIGS, ALL_DUNGEON_CONFIGS, Mobs, get_new_full_inventory, get_new_empty_inventory
from craftax.craftax.game_logic import *
from craftax.craftax.util.game_logic_utils import *


@struct.dataclass
class EnvParams:
    max_timesteps: int = 100000
    day_length: int = 300

    always_diamond: bool = False

    mob_despawn_distance: int = 14
    max_attribute: int = 5

    god_mode: bool = False
    world_seeds: Tuple[int, ...] = tuple()
    goals: Tuple[int, ...] = tuple()
    goal_vector: jax.Array = jnp.zeros(1)
    goal: int = 0

    fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)


@struct.dataclass
class StaticEnvParams:
    map_size: Tuple[int, int] = (48, 48)
    num_levels: int = 9

    # Mobs
    max_melee_mobs: int = 3
    max_passive_mobs: int = 3
    max_growing_plants: int = 10
    max_ranged_mobs: int = 2
    max_mob_projectiles: int = 3
    max_player_projectiles: int = 3
    use_precondition: bool = False

    # Custom addiitons for human experiments
    initial_crafting_tables: bool = True
    initial_strength: int = 20


class Observation(struct.PyTreeNode):
    image: chex.Array

def get_map_obs_shape():
    num_mob_classes = 5
    num_mob_types = 8
    num_blocks = len(BlockType)
    num_items = len(ItemType)

    return (
        OBS_DIM[0],
        OBS_DIM[1],
        num_blocks + num_items + num_mob_classes * num_mob_types + 1,
    )


def get_flat_map_obs_shape():
    map_obs_shape = get_map_obs_shape()
    return map_obs_shape[0] * map_obs_shape[1] * map_obs_shape[2]


def get_inventory_obs_shape():
    return 51


def generate_world(rng, params, static_params):
    player_position = jnp.array(
        [static_params.map_size[0] // 2, static_params.map_size[1] // 2]
    )

    # Generate smoothgens (overworld, caves, elemental levels, boss level)
    rngs = jax.random.split(rng, 7)
    rng, _rng = rngs[0], rngs[1:]

    smoothgens = jax.vmap(generate_smoothworld, in_axes=(0, None, None, 0, None))(
        _rng, static_params, player_position, ALL_SMOOTHGEN_CONFIGS, params
    )

    # Generate dungeons
    rngs = jax.random.split(rng, 4)
    rng, _rng = rngs[0], rngs[1:]
    dungeons = jax.vmap(generate_dungeon, in_axes=(0, None, 0))(
        _rng, static_params, ALL_DUNGEON_CONFIGS
    )

    # Splice smoothgens and dungeons in order of levels
    map, item_map, light_map, ladders_down, ladders_up = tree_map(
        lambda x, y: jnp.stack(
            (x[0], y[0], x[1], y[1], y[2], x[2], x[3], x[4], x[5]), axis=0
        ),
        smoothgens,
        dungeons,
    )

    # Mobs
    def generate_empty_mobs(max_mobs):
        return Mobs(
            position=jnp.zeros(
                (static_params.num_levels, max_mobs, 2), dtype=jnp.int32
            ),
            health=jnp.ones((static_params.num_levels,
                            max_mobs), dtype=jnp.float32),
            mask=jnp.zeros((static_params.num_levels, max_mobs), dtype=bool),
            attack_cooldown=jnp.zeros(
                (static_params.num_levels, max_mobs), dtype=jnp.int32
            ),
            type_id=jnp.zeros(
                (static_params.num_levels, max_mobs), dtype=jnp.int32),
        )

    melee_mobs = generate_empty_mobs(static_params.max_melee_mobs)
    ranged_mobs = generate_empty_mobs(static_params.max_ranged_mobs)
    passive_mobs = generate_empty_mobs(static_params.max_passive_mobs)

    # Projectiles
    def _create_projectiles(max_num):
        projectiles = generate_empty_mobs(max_num)

        projectile_directions = jnp.ones(
            (static_params.num_levels, max_num, 2), dtype=jnp.int32
        )

        return projectiles, projectile_directions

    mob_projectiles, mob_projectile_directions = _create_projectiles(
        static_params.max_mob_projectiles
    )
    player_projectiles, player_projectile_directions = _create_projectiles(
        static_params.max_player_projectiles
    )

    # Plants
    growing_plants_positions = jnp.zeros(
        (static_params.max_growing_plants, 2), dtype=jnp.int32
    )
    growing_plants_age = jnp.zeros(
        static_params.max_growing_plants, dtype=jnp.int32)
    growing_plants_mask = jnp.zeros(
        static_params.max_growing_plants, dtype=bool)

    # Potion mapping for episode
    rng, _rng = jax.random.split(rng)
    potion_mapping = jax.random.permutation(_rng, jnp.arange(6))

    # Inventory
    inventory = tree_map(
        lambda x, y: jax.lax.select(params.god_mode, x, y),
        get_new_full_inventory(),
        get_new_empty_inventory(),
    )

    rng, _rng = jax.random.split(rng)

    # ========================================
    # Add 2 random crafting tables
    # ========================================
    def find_empty_position(rng, current_map):
        # Generate multiple candidate positions
        rng, _rng = jax.random.split(rng)
        num_candidates = 10  # Try multiple positions to increase success chance
        positions = jax.random.randint(
            _rng,
            shape=(num_candidates, 2),
            minval=2,
            maxval=jnp.array([static_params.map_size[0]-2,
                             static_params.map_size[1]-2])
        )

        # Check which positions are empty (0 typically represents empty space)
        is_empty = current_map[positions[:, 0], positions[:, 1]] == 0

        # Take the first empty position found
        valid_pos = positions[jnp.argmax(is_empty)]
        return valid_pos, rng

    # ========================================
    # NOTE: MAIN DIFFERENCE: 
    # Add 2 random crafting tables
    # ========================================
    if static_params.initial_crafting_tables:
      # Place first table
      rng, _rng = jax.random.split(rng)
      pos, _rng = find_empty_position(_rng, map[0])
      map = map.at[0, pos[0], pos[1]].set(BlockType.CRAFTING_TABLE.value)

      # Place second table
      rng, _rng = jax.random.split(_rng)
      pos, _rng = find_empty_position(_rng, map[0])
      map = map.at[0, pos[0], pos[1]].set(BlockType.CRAFTING_TABLE.value)

    state = EnvState(
        map=map,
        item_map=item_map,
        mob_map=jnp.zeros(
            (static_params.num_levels, *static_params.map_size), dtype=bool
        ),
        light_map=light_map,
        down_ladders=ladders_down,
        up_ladders=ladders_up,
        chests_opened=jnp.zeros(static_params.num_levels, dtype=bool),
        monsters_killed=jnp.zeros(static_params.num_levels, dtype=jnp.int32)
        .at[0]
        .set(10),  # First ladder starts open
        player_position=player_position,
        player_direction=jnp.asarray(Action.UP.value, dtype=jnp.int32),
        player_level=jnp.asarray(0, dtype=jnp.int32),
        player_health=jnp.asarray(9.0, dtype=jnp.float32),
        player_food=jnp.asarray(9, dtype=jnp.int32),
        player_drink=jnp.asarray(9, dtype=jnp.int32),
        player_energy=jnp.asarray(9, dtype=jnp.int32),
        player_mana=jnp.asarray(9, dtype=jnp.int32),
        player_recover=jnp.asarray(0.0, dtype=jnp.float32),
        player_hunger=jnp.asarray(0.0, dtype=jnp.float32),
        player_thirst=jnp.asarray(0.0, dtype=jnp.float32),
        player_fatigue=jnp.asarray(0.0, dtype=jnp.float32),
        player_recover_mana=jnp.asarray(0.0, dtype=jnp.float32),
        is_sleeping=False,
        is_resting=False,
        player_xp=jnp.asarray(0, dtype=jnp.int32),
        player_dexterity=jnp.asarray(1, dtype=jnp.int32),
        # NOTE: MAIN DIFFERENCE: Humans start with 20 strength
        player_strength=jnp.asarray(static_params.initial_strength, dtype=jnp.int32),  
        player_intelligence=jnp.asarray(1, dtype=jnp.int32),
        inventory=inventory,
        sword_enchantment=jnp.asarray(0, dtype=jnp.int32),
        bow_enchantment=jnp.asarray(0, dtype=jnp.int32),
        armour_enchantments=jnp.array([0, 0, 0, 0], dtype=jnp.int32),
        melee_mobs=melee_mobs,
        ranged_mobs=ranged_mobs,
        passive_mobs=passive_mobs,
        mob_projectiles=mob_projectiles,
        mob_projectile_directions=mob_projectile_directions,
        player_projectiles=player_projectiles,
        player_projectile_directions=player_projectile_directions,
        growing_plants_positions=growing_plants_positions,
        growing_plants_age=growing_plants_age,
        growing_plants_mask=growing_plants_mask,
        potion_mapping=potion_mapping,
        learned_spells=jnp.array([False, False], dtype=bool),
        boss_progress=jnp.asarray(0, dtype=jnp.int32),
        boss_timesteps_to_spawn_this_round=jnp.asarray(
            BOSS_FIGHT_SPAWN_TURNS, dtype=jnp.int32
        ),
        achievements=jnp.zeros((len(Achievement),), dtype=bool),
        light_level=jnp.asarray(calculate_light_level(
            0, params), dtype=jnp.float32),
        state_rng=_rng,
        timestep=jnp.asarray(0, dtype=jnp.int32),
    )

    return state

class CraftaxSymbolicWebEnvNoAutoReset(EnvironmentNoAutoReset):
    def __init__(self, static_env_params: Optional[StaticEnvParams] = None):
        super().__init__()

        if static_env_params is None:
            static_env_params = self.default_static_params()
        self.static_env_params = static_env_params

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()

    def step_env(
        self, rng: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """
        
        Changes:
         1. have single achievement complete task.
         2. collapse multiple actions to "do" action. this makes it easier for people.

        """

        def step(state, action, params):
          state, _ = craftax_step(
              rng, state, action, params, self.static_env_params)
          goal_achieved = jax.lax.dynamic_index_in_dim(
              state.achievements, params.goal, keepdims=False)
          goal_achieved = goal_achieved.astype(jnp.float32)
          return state, goal_achieved

        def mapped_do_step(state, params):
          do_mapped_actions = jnp.asarray((
              Action.DO.value,
              Action.MAKE_WOOD_SWORD.value),
          )
          states, rewards = jax.vmap(
              step, in_axes=(None, 0, None))(
                  state, do_mapped_actions, params)
          import pdb; pdb.set_trace()
          best_idx = jnp.argmax(rewards)
          best_state = jax.lax.dynamic_index_in_dim(states, best_idx, keepdims=False)
          return best_state

        state, reward = jax.lax.cond(
            action == Action.DO.value,
            lambda s, p: mapped_do_step(s, p),
            lambda s, p: step(s, action, p),
            state, params
        )

        done = reward > 0
        info = log_achievements_to_info(state, done)
        info["discount"] = self.discount(state, params)

        return (
            lax.stop_gradient(self.get_obs(
                state=state,
                params=params)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, rng: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """NOTE: main change is to select world seed from a set of seeds"""
        if params.world_seeds is not None and len(params.world_seeds) > 0:
          reset_seeds = jnp.asarray(params.world_seeds)
          rng, _rng = jax.random.split(rng)
          selected_seed = jax.random.choice(_rng, reset_seeds)
          world_rng = jax.random.PRNGKey(selected_seed)
        else:
          rng, world_rng = jax.random.split(rng)
        state = generate_world(world_rng, params, self.static_env_params)

        obs = self.get_obs(
            state=state,
            params=params,
        )
        return obs, state

    def get_obs(self,
                state: EnvState,
                params: EnvParams):
        del params
        return render_craftax_symbolic(state)

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return is_game_over(state, params, self.static_env_params)

    @property
    def name(self) -> str:
        return "Craftax-Symbolic-NoAutoReset-v1"

    @property
    def num_actions(self) -> int:
        return len(Action)

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(len(Action))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        flat_map_obs_shape = get_flat_map_obs_shape()
        inventory_obs_shape = get_inventory_obs_shape()

        obs_shape = flat_map_obs_shape + inventory_obs_shape

        return spaces.Box(
            0.0,
            1.0,
            (obs_shape,),
            dtype=jnp.float32,
        )
