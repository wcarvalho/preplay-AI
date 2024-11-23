"""
Changes:
1. Added a set of seeds to EnvParams so that train + env can have different seeds
"""
import jax
from jax import lax
from gymnax.environments import spaces, environment
from typing import Tuple, Optional
import chex
from flax import struct

from craftax.craftax.envs.common import log_achievements_to_info
from craftax.environment_base.environment_bases import EnvironmentNoAutoReset
from craftax.craftax.constants import *
from craftax.craftax.game_logic import craftax_step, is_game_over
from craftax.craftax.craftax_state import EnvState, StaticEnvParams
from craftax.craftax.renderer import render_craftax_symbolic
from craftax.craftax.world_gen.world_gen import generate_world, generate_smoothworld, generate_dungeon, ALL_SMOOTHGEN_CONFIGS, ALL_DUNGEON_CONFIGS, Mobs, get_new_full_inventory, get_new_empty_inventory
# NOTE: slight hack to import all game logic functions
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
    reset_seeds: chex.Array = jnp.arange(10_000)

    fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)

dummy_achievements = jnp.zeros(
    len(ACHIEVEMENT_REWARD_MAP)+1,
    dtype=jnp.float32)

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


class Observation(struct.PyTreeNode):
    image: chex.Array
    achievable: chex.Array
    achievements: chex.Array
    task_w: chex.Array


def get_possible_achievements(state, static_params):
    """Returns a boolean array indicating which achievements are currently possible based only on visible state."""
    
    possible = jnp.zeros_like(state.achievements)
    
    # Get visible map view (same logic as render_craftax_symbolic)
    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)
    padded_grid = jnp.pad(
        state.map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )
    tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2
    visible_map = jax.lax.dynamic_slice(padded_grid, tl_corner, OBS_DIM)
    
    # Get visible mobs (similar to render_craftax_symbolic mob logic)
    def is_mob_visible(mob_pos, mob_mask):
        local_position = (
            mob_pos
            - state.player_position
            + jnp.array([OBS_DIM[0], OBS_DIM[1]]) // 2
        )
        on_screen = jnp.logical_and(
            local_position >= 0,
            local_position < jnp.array([OBS_DIM[0], OBS_DIM[1]])
        ).all()
        return jnp.logical_and(on_screen, mob_mask)
    
    # Resource collection possibilities (only visible blocks)
    possible = possible.at[Achievement.COLLECT_WOOD.value].set(
        jnp.any(visible_map == BlockType.TREE.value)
    )
    
    possible = possible.at[Achievement.COLLECT_STONE.value].set(
        jnp.logical_and(
            state.inventory.wood_pickaxe,
            jnp.any(visible_map == BlockType.STONE.value)
        )
    )
    
    possible = possible.at[Achievement.COLLECT_COAL.value].set(
        jnp.logical_and(
            state.inventory.wood_pickaxe,
            jnp.any(visible_map == BlockType.COAL.value)
        )
    )
    
    possible = possible.at[Achievement.COLLECT_IRON.value].set(
        jnp.logical_and(
            state.inventory.stone_pickaxe,
            jnp.any(visible_map == BlockType.IRON.value)
        )
    )
    
    possible = possible.at[Achievement.COLLECT_DIAMOND.value].set(
        jnp.logical_and(
            state.inventory.iron_pickaxe,
            jnp.any(visible_map == BlockType.DIAMOND.value)
        )
    )
    
    # Crafting possibilities (only if crafting table is visible)
    crafting_table_visible = jnp.any(visible_map == BlockType.CRAFTING_TABLE.value)
    furnace_visible = jnp.any(visible_map == BlockType.FURNACE.value)
    
    possible = possible.at[Achievement.MAKE_WOOD_PICKAXE.value].set(
        jnp.logical_and(
            state.inventory.wood >= 1,
            crafting_table_visible
        )
    )
    
    possible = possible.at[Achievement.MAKE_STONE_PICKAXE.value].set(
        jnp.logical_and(
            jnp.logical_and(state.inventory.wood >= 1, state.inventory.stone >= 1),
            crafting_table_visible
        )
    )
    
    possible = possible.at[Achievement.MAKE_IRON_PICKAXE.value].set(
        jnp.logical_and(
            jnp.logical_and(
                state.inventory.wood >= 1,
                jnp.logical_and(
                    state.inventory.stone >= 1,
                    jnp.logical_and(
                        state.inventory.iron >= 1,
                        state.inventory.coal >= 1
                    )
                )
            ),
            jnp.logical_and(crafting_table_visible, furnace_visible)
        )
    )
    
    # Combat/Mob possibilities (only visible mobs)
    visible_zombies = jax.vmap(
        lambda idx: is_mob_visible(state.zombies.position[idx], state.zombies.mask[idx])
    )(jnp.arange(state.zombies.mask.shape[0]))
    
    visible_skeletons = jax.vmap(
        lambda idx: is_mob_visible(state.skeletons.position[idx], state.skeletons.mask[idx])
    )(jnp.arange(state.skeletons.mask.shape[0]))
    
    visible_cows = jax.vmap(
        lambda idx: is_mob_visible(state.cows.position[idx], state.cows.mask[idx])
    )(jnp.arange(state.cows.mask.shape[0]))
    
    possible = possible.at[Achievement.DEFEAT_ZOMBIE.value].set(
        jnp.any(visible_zombies)
    )
    
    possible = possible.at[Achievement.DEFEAT_SKELETON.value].set(
        jnp.any(visible_skeletons)
    )
    
    # Food/Water possibilities (only visible resources)
    possible = possible.at[Achievement.EAT_COW.value].set(
        jnp.any(visible_cows)
    )
    
    possible = possible.at[Achievement.EAT_PLANT.value].set(
        jnp.any(visible_map == BlockType.RIPE_PLANT.value)
    )
    
    possible = possible.at[Achievement.COLLECT_DRINK.value].set(
        jnp.any(visible_map == BlockType.WATER.value)
    )
    
    # Other possibilities (based on player state)
    possible = possible.at[Achievement.WAKE_UP.value].set(
        jnp.logical_and(state.is_sleeping, state.player_energy >= 9)
    )
    
    # Filter out already achieved
    possible = jnp.logical_and(possible, jnp.logical_not(state.achievements))
    
    return possible

def describe_possible_achievements(possible_achievements):
    """
    Converts a binary achievement vector into human-readable text.
    
    Args:
        possible_achievements: Binary array indicating which achievements are possible
        
    Returns:
        List of strings describing currently possible achievements
    """
    possible_tasks = []
    
    # Map achievement indices to descriptions
    achievement_descriptions = {
        Achievement.COLLECT_WOOD.value: "Collect wood from a tree",
        Achievement.COLLECT_STONE.value: "Mine stone with a wooden pickaxe",
        Achievement.COLLECT_COAL.value: "Mine coal with a wooden pickaxe",
        Achievement.COLLECT_IRON.value: "Mine iron with a stone pickaxe",
        Achievement.COLLECT_DIAMOND.value: "Mine diamond with an iron pickaxe",
        Achievement.COLLECT_SAPLING.value: "Collect a sapling",
        Achievement.PLACE_PLANT.value: "Plant a sapling",
        Achievement.EAT_PLANT.value: "Eat from a ripe plant",
        Achievement.COLLECT_DRINK.value: "Drink water",
        Achievement.PLACE_TABLE.value: "Place a crafting table",
        Achievement.PLACE_FURNACE.value: "Place a furnace",
        Achievement.PLACE_STONE.value: "Place stone",
        Achievement.MAKE_WOOD_PICKAXE.value: "Craft a wooden pickaxe",
        Achievement.MAKE_STONE_PICKAXE.value: "Craft a stone pickaxe",
        Achievement.MAKE_IRON_PICKAXE.value: "Craft an iron pickaxe",
        Achievement.MAKE_WOOD_SWORD.value: "Craft a wooden sword",
        Achievement.MAKE_STONE_SWORD.value: "Craft a stone sword",
        Achievement.MAKE_IRON_SWORD.value: "Craft an iron sword",
        Achievement.DEFEAT_ZOMBIE.value: "Defeat a zombie",
        Achievement.DEFEAT_SKELETON.value: "Defeat a skeleton",
        Achievement.EAT_COW.value: "Kill and eat a cow",
        Achievement.WAKE_UP.value: "Wake up from sleep"
    }
    
    for idx, is_possible in enumerate(possible_achievements):
        if is_possible:
            description = achievement_descriptions.get(idx, f"Unknown achievement {idx}")
            possible_tasks.append(description)
            
    return possible_tasks

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
    map, item_map, light_map, ladders_down, ladders_up = jax.tree.map(
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
    inventory = jax.tree.map(
        lambda x, y: jax.lax.select(params.god_mode, x, y),
        get_new_full_inventory(),
        get_new_empty_inventory(),
    )

    rng, _rng = jax.random.split(rng)

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
        player_strength=jnp.asarray(1, dtype=jnp.int32),
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

def craftax_step(rng, state, action, params, static_params):
    init_achievements = state.achievements
    init_health = state.player_health

    # Interrupt action if sleeping or resting
    action = jax.lax.select(state.is_sleeping, Action.NOOP.value, action)
    action = jax.lax.select(state.is_resting, Action.NOOP.value, action)

    # Change floor
    state = change_floor(state, action, params, static_params)

    # Crafting
    state = do_crafting(state, action)

    # Interact (mining, melee attacking, eating plants, drinking water)
    rng, _rng = jax.random.split(rng)
    state = do_action(_rng, state, action, static_params)

    # Placing
    state = place_block(state, action, static_params)

    # Shooting
    state = shoot_projectile(state, action, static_params)

    # Casting
    state = cast_spell(state, action, static_params)

    # Potions
    state = drink_potion(state, action)

    # Read
    rng, _rng = jax.random.split(rng)
    state = read_book(_rng, state, action)

    # Enchant
    rng, _rng = jax.random.split(rng)
    state = enchant(_rng, state, action)

    # Boss
    state = boss_logic(state, static_params)

    # Attributes
    state = level_up_attributes(state, action, params)

    # Movement
    state = move_player(state, action, params)

    # Mobs
    rng, _rng = jax.random.split(rng)
    state = update_mobs(_rng, state, params, static_params)

    rng, _rng = jax.random.split(rng)
    state = spawn_mobs(state, _rng, params, static_params)

    # Plants
    state = update_plants(state, static_params)

    # Intrinsics
    state = update_player_intrinsics(state, action, static_params)

    # Cap inv
    state = clip_inventory_and_intrinsics(state, params)

    # Inventory achievements
    state = calculate_inventory_achievements(state)

    # NOTE: MAIN CHANGE is just exposing achievements
    # normally, just summed and provided to agent
    achievements = jnp.concatenate((
          (state.achievements.astype(int) - init_achievements.astype(int)).astype(jnp.float32),
          jnp.expand_dims(state.player_health - init_health, 0).astype(jnp.float32)
        ))
    achievement_coefficients = jnp.concatenate(
        (ACHIEVEMENT_REWARD_MAP, jnp.array([0.1]))
    )
    reward = (achievements * achievement_coefficients).sum()

    rng, _rng = jax.random.split(rng)

    state = state.replace(
        timestep=state.timestep + 1,
        light_level=calculate_light_level(state.timestep + 1, params),
        state_rng=_rng,
    )

    return state, reward, achievements, achievement_coefficients

class CraftaxSymbolicEnvNoAutoReset(EnvironmentNoAutoReset):
    """
    Made following changes:
    1. sample world set from pre-defined set of seeds
    2. structured obs with {image, achievable, achieved, task-vector}
    3. option to have task-vector double with zeros (to accomodate successor features)
    """
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

        state, reward, achievements, achievement_coefficients = craftax_step(rng, state, action, params, self.static_env_params)

        done = self.is_terminal(state, params)
        info = log_achievements_to_info(state, done)
        info["discount"] = self.discount(state, params)

        return (
            lax.stop_gradient(self.get_obs(state, achievements, achievement_coefficients, params)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, rng: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """NOTE: main change is to select world seed from a set of seeds"""
        reset_seeds = params.reset_seeds
        rng, _rng = jax.random.split(rng)
        selected_seed = jax.random.choice(_rng, reset_seeds)
        world_rng = jax.random.PRNGKey(selected_seed)
        state = generate_world(world_rng, params, self.static_env_params)

        return self.get_obs(state, dummy_achievements, dummy_achievements, params), state

    def get_obs(self, state: EnvState, achievements: chex.Array, achievement_coefficients: chex.Array, params: EnvParams):
        task_w = jnp.concatenate(
            (achievement_coefficients,
             jnp.zeros_like(achievement_coefficients)))
        #achievable=get_possible_achievements(
        #        state, self.static_env_params)
        return Observation(
            image=render_craftax_symbolic(state),
            achievable=achievements,
            achievements=achievements,
            task_w=task_w)

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

