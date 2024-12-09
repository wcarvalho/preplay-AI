"""
MAIN CHANGE WAS TO ADD EVALUATION with test_env_params

TESTING:
JAX_DEBUG_NANS=True \
JAX_DISABLE_JIT=1 \
HYDRA_FULL_ERROR=1 JAX_TRACEBACK_FILTERING=off python -m ipdb -c continue craftax_ppo_trainer.py \
  app.parallel=none \
  app.debug=True \
  app.wandb=False

RUNNING ON SLURM:
python craftax_ppo_trainer.py \
  app.parallel=slurm
"""

from typing import Any, Callable, Dict, Union, Optional, Sequence, NamedTuple
from functools import partial  # for @partial decorator
import os
import sys
import time

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import chex  # for type checking
import hydra  # for hydra configuration
from omegaconf import DictConfig  # for hydra type hints

from safetensors.flax import save_file
from flax.traverse_util import flatten_dict

from flax.training import orbax_utils
from flax import struct
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

import wandb
from flax.linen.initializers import constant, orthogonal
import distrax

# Assuming these are local imports from your project
from craftax.craftax_env import make_craftax_env_from_name
from jaxneurorl import launcher

from craftax.craftax.constants import Achievement


import time

import jax.numpy as jnp
import numpy as np
import wandb

MAX_SCORE = 226.0
batch_logs = {}
log_times = []


def create_log_dict(info, config):
    to_log = {
        "episode_return": info["returned_episode_returns"],
        "episode_length": info["returned_episode_lengths"],
    }

    sum_achievements = 0
    for k, v in info.items():
        if "achievements" in k.lower():
            to_log[k] = v
            sum_achievements += v / 100.0

    to_log["achievements"] = sum_achievements

    if config.get("TRAIN_ICM") or config.get("USE_RND"):
        to_log["intrinsic_reward"] = info["reward_i"]
        to_log["extrinsic_reward"] = info["reward_e"]

        if config.get("TRAIN_ICM"):
            to_log["icm_inverse_loss"] = info["icm_inverse_loss"]
            to_log["icm_forward_loss"] = info["icm_forward_loss"]
        elif config.get("USE_RND"):
            to_log["rnd_loss"] = info["rnd_loss"]

    return to_log


def batch_log(update_step, log, config):
    update_step = int(update_step)
    if update_step not in batch_logs:
        batch_logs[update_step] = []

    batch_logs[update_step].append(log)

    if len(batch_logs[update_step]) == config["NUM_REPEATS"]:
        agg_logs = {}
        for key in batch_logs[update_step][0]:
            agg = []
            if key in ["goal_heatmap"]:
                agg = [batch_logs[update_step][0][key]]
            else:
                for i in range(config["NUM_REPEATS"]):
                    val = batch_logs[update_step][i][key]
                    if not jnp.isnan(val):
                        agg.append(val)

            if len(agg) > 0:
                if key in [
                    "episode_length",
                    "episode_return",
                    "exploration_bonus",
                    "e_mean",
                    "e_std",
                    "rnd_loss",
                ]:
                    agg_logs[key] = np.mean(agg)
                else:
                    agg_logs[key] = np.array(agg)

        log_times.append(time.time())

        if config["DEBUG"]:
            if len(log_times) == 1:
                print("Started logging")
            elif len(log_times) > 1:
                dt = log_times[-1] - log_times[-2]
                steps_between_updates = (
                    config["NUM_STEPS"] * config["NUM_ENVS"] * config["NUM_REPEATS"]
                )
                sps = steps_between_updates / dt
                agg_logs["sps"] = sps

        wandb.log(agg_logs)

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

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)

class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        obs, env_state = self._env.reset(key, params)
        # NOTE: change, technically episode length is 1 here, not 0
        state = LogEnvState(env_state, 0.0, 1, 0.0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float],
        params=None,
    ):
        """
        # NOTE: MAJOR change from original code is that they assume that finishing an episode leads to the first state of the next episode in the same index.
        In this codebase, when you finish an episode, you see the first state of the next episode in the NEXT index.
        """
        
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
            returned_episode_returns=(state.returned_episode_returns * (1 - done_float)
            + new_episode_return * done_float),
            returned_episode_lengths=state.returned_episode_lengths * (1 - done_int)
            + new_episode_length * done_int,
            timestep=state.timestep + 1,
        )
        return obs, state, reward, done, info


class OptimisticResetVecEnvWrapper(GymnaxWrapper):
    """
    Provides efficient 'optimistic' resets.
    The wrapper also necessarily handles the batching of environment steps and resetting.
    reset_ratio: the number of environment workers per environment reset.  Higher means more efficient but a higher
    chance of duplicate resets.
    """

    def __init__(self, env, num_envs: int, reset_ratio: int):
        super().__init__(env)

        self.num_envs = num_envs
        self.reset_ratio = reset_ratio
        assert (
            num_envs % reset_ratio == 0
        ), "Reset ratio must perfectly divide num envs."
        self.num_resets = self.num_envs // reset_ratio

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs_st, state_st, reward, done, info = self.step_fn(
            rngs, state, action, params)

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_resets)
        obs_re, state_re = self.reset_fn(rngs, params)

        rng, _rng = jax.random.split(rng)
        reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)

        being_reset = jax.random.choice(
            _rng,
            jnp.arange(self.num_envs),
            shape=(self.num_resets,),
            p=done,
            replace=False,
        )
        reset_indexes = reset_indexes.at[being_reset].set(
            jnp.arange(self.num_resets))

        obs_re = jax.tree.map(lambda x: x[reset_indexes], obs_re)
        state_re = jax.tree.map(lambda x: x[reset_indexes], state_re)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree.map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.tree.map(
                lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st
            )

            return state, obs

        state, obs = jax.vmap(auto_reset)(
            done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info

class ScannedRNN(nn.Module):
    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        def embed(x): 
            return nn.Dense(
              self.config["LAYER_SIZE"],
              kernel_init=orthogonal(np.sqrt(2)),
              bias_init=constant(0.0),
          )(x)
        #####
        # image
        #####
        image = obs.image
        embedding = embed(image)
        embedding = nn.relu(embedding)
        #####
        # action + achievable
        #####
        def embed_binary(x): 
            return nn.Dense(
              128,
              kernel_init=nn.initializers.variance_scaling(
                  1.0, 'fan_in', 'normal', out_axis=0),
              bias_init=constant(0.0),
          )(x)
        previous_action = jax.nn.one_hot(obs.previous_action, self.action_dim)
        to_concat = (
            embedding,
            embed_binary(previous_action),
            embed_binary(obs.achievable.astype(jnp.float32)))
        embedding = jnp.concatenate(to_concat, axis=-1)
        embedding = embed(image)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = embed(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = embed(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = embed(embedding)
        critic = nn.relu(critic)
        critic = embed(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def create_metrics(log_states, dones, prefix="eval"):
    """Create evaluation metrics dictionary and compute episode averages.
    
    Args:
        log_states: LogEnvState containing episode returns and lengths
        achievements: Array of achievement values [T, B, A]
        dones: Boolean array indicating episode completion
        prefix: String prefix for metric keys (default: "eval")
        
    Returns:
        Dictionary of averaged evaluation metrics
    """
    main_key = prefix
    ach_key = f"{prefix}-achievements"

    metrics = {}
    metrics[f"{main_key}/0.episode_return"] = log_states.returned_episode_returns
    metrics[f"{main_key}/0.score"] = (
        log_states.returned_episode_returns/MAX_SCORE)*100.0
    metrics[f"{main_key}/0.episode_length"] = log_states.returned_episode_lengths

    achievements = log_states.env_state.achievements  # [T, B, A]
    achievements = achievements * dones[:, :, None] * 100.0  # Scale
    for achievement in Achievement:
        name = f"Achievements/{achievement.name.lower()}"
        metrics[f"{ach_key}/{name}"] = achievements[:, :, achievement.value]

    # Compute final metrics (average over episodes)
    metrics = jax.tree.map(
        lambda x: (x * dones).sum() / (1e-5 + dones.sum()),
        metrics
    )

    return metrics

def evaluate_model(network, params, rng, env, test_env_params, config):
    """Evaluate the model on test environments."""
    
    def _eval_step(carry, unused):
        hstate, obs, done, rng, log_state = carry
        
        # Select action
        ac_in = jax.tree.map(lambda x: x[jnp.newaxis, :], (obs, done))
        hstate, pi, value = network.apply(params, hstate, ac_in)
        rng, _rng = jax.random.split(rng)
        action = pi.sample(seed=_rng)
        action = action.squeeze(0)
        
        # Step environment
        rng, _rng = jax.random.split(rng)
        next_obs, log_state, reward, done, info = env.step(_rng, log_state, action, test_env_params)
        
        return (hstate, next_obs, done, rng, log_state), (reward, done, log_state)
    
    # Initialize evaluation
    rng, _rng = jax.random.split(rng)
    obs, log_state = env.reset(_rng, test_env_params)
    init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["LAYER_SIZE"])
    done = jnp.zeros((config["NUM_ENVS"]), dtype=bool)
    
    # Run evaluation for fixed number of steps
    carry = (init_hstate, obs, done, rng, log_state)
    _, (rewards, dones, log_states) = jax.lax.scan(
        _eval_step, carry, None, config["NUM_STEPS"] * 100
    )
    metrics = create_metrics(log_states, dones)

    return metrics

def make_train(config, env, env_params, test_env_params):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env_params).n, config=config)
        rng, _rng = jax.random.split(rng)
        init_obs, init_env_state = env.reset(rng, env_params)
        init_x = (
            init_obs,
            jnp.zeros((config["NUM_ENVS"])),
        )
        init_x = jax.tree.map(lambda x: x[None], init_x)
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["LAYER_SIZE"]
        )
        network_params = network.init(_rng, init_hstate, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["LAYER_SIZE"]
        )

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            ############################################################
            # COLLECT TRAIN TRAJECTORIES
            ############################################################
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    hstate,
                    rng,
                    update_step,
                ) = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = jax.tree.map(lambda x: x[np.newaxis, :], (last_obs, last_done))
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(
                    _rng, env_state, action, env_params
                )
                transition = Transition(
                    last_done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done,
                    hstate,
                    rng,
                    update_step,
                )
                return runner_state, (transition, env_state)

            initial_hstate = runner_state[-3]
            runner_state, (traj_batch, traj_states) = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step,
            ) = runner_state
            ac_in = jax.tree.map(lambda x: x[np.newaxis, :], (last_obs, last_done))
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = (
                        reward + config["GAMMA"] * next_value * (1 - next_done) - value
                    )
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                    )
                    return (gae, value, done), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, last_done),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params, init_hstate[0], (traj_batch.obs, traj_batch.done)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            init_hstate = initial_hstate[None, :]  # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]

            metric = create_metrics(traj_states, traj_batch.done)
            rng = update_state[-1]
            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, update_step):
                    to_log = create_log_dict(metric, config)
                    batch_log(update_step, to_log, config)

                jax.debug.callback(callback, metric, update_step)

            ############################################################
            # EVALUATION
            ############################################################
            def do_eval(args):
                train_state, eval_rng, update_step = args
                eval_metrics = evaluate_model(
                    network=network, 
                    params=train_state.params, 
                    rng=eval_rng, 
                    env=env, 
                    test_env_params=test_env_params, 
                    config=config,

                )
                
                if config["DEBUG"] and config["USE_WANDB"]:
                    def eval_callback(eval_metrics, update_step):
                        to_log = create_log_dict(eval_metrics, config)
                        # Add 'eval_' prefix to all metrics
                        to_log = {f"eval_{k}": v for k, v in to_log.items()}
                        batch_log(update_step, to_log, config)
                    
                    jax.debug.callback(eval_callback, eval_metrics, update_step)
                
                return rng

            # Condition for evaluation (every 10% of training)
            should_eval = ((update_step + 1) % (config["NUM_UPDATES"] // 10)) == 0
            rng, eval_rng = jax.random.split(rng)
            jax.lax.cond(
                should_eval,
                do_eval,
                lambda x: None,
                (train_state, eval_rng, update_step)
            )

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step + 1,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            _rng,
            0,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train

def run_single(
    config: dict,
    save_path: str = None):

    from craftax_env import CraftaxSymbolicEnvNoAutoReset
    config['STRUCTURED_INPUTS'] = True
    config['NUM_ENV_SEEDS'] = config.get('NUM_ENV_SEEDS', 10_000)

    static_env_params = CraftaxSymbolicEnvNoAutoReset.default_static_params().replace(
      use_precondition=config.get('USE_PRECONDITION', False))

    env = CraftaxSymbolicEnvNoAutoReset(static_env_params=static_env_params)
    
    if config['NUM_ENV_SEEDS']:
      env_params = env.default_params.replace(
        world_seeds=tuple(np.arange(config['NUM_ENV_SEEDS'])))
      test_env_params = env.default_params.replace(
        world_seeds=tuple(np.arange(config['NUM_ENV_SEEDS'], config['NUM_ENV_SEEDS'] + config['TEST_NUM_ENVS'])))
    else:
      env_params = env.default_params
      test_env_params = env.default_params

    # Wrap with some extra logging
    env = LogWrapper(env)

    # Wrap with a batcher, maybe using optimistic resets
    env = OptimisticResetVecEnvWrapper(
        env,
        num_envs=config["NUM_ENVS"],
        reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
    )

    train_fn = make_train(
        config=config,
        env=env,
        env_params=env_params,
        test_env_params=test_env_params,
        )

    start_time = time.time()
    train_vjit = jax.jit(jax.vmap(train_fn))

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    outs = jax.block_until_ready(train_vjit(rngs))
    elapsed_time = time.time() - start_time
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))

    #---------------
    # save model weights
    #---------------
    alg_name = config['ALG']
    if save_path is not None:
        def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
            flattened_dict = flatten_dict(params, sep=',')
            save_file(flattened_dict, filename)

        model_state = outs['runner_state'][0]
        # save only params of the firt run
        params = jax.tree.map(lambda x: x[0], model_state.params)
        os.makedirs(save_path, exist_ok=True)

        save_params(params, f'{save_path}/{alg_name}.safetensors')
        print(f'Parameters of first batch saved in {save_path}/{alg_name}.safetensors')

        config_filename = f'{save_path}/{alg_name}.config'
        import pickle
        # Save the dictionary as a pickle file
        with open(config_filename, 'wb') as f:
          pickle.dump(config, f)
        print(f'Config saved in {config_filename}')

def sweep(search: str = ''):
  search = search or 'ppo'
  ############################################################
  # Testing
  ############################################################
  metric = {
    'name': 'evaluator_performance-0/0.score',
    'goal': 'maximize',
  }
  if search == 'ppo':
    sweep_config = {
        'metric': metric,
        'parameters': {
            #"ENV": {'values': ['craftax']},
            "SEED": {'values': list(range(1,2))},
        },
        'overrides': ['alg=ppo', 'rlenv=craftax-10m', 'user=wilka'],
        'group': 'ppo-1',
    }

  else:
    raise NotImplementedError(search)


  return sweep_config

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'configs'))
@hydra.main(
    version_base=None,
    config_path=CONFIG_PATH,
    config_name="config_craftax")
def main(config: DictConfig):
  launcher.run(
      config,
      trainer_filename=__file__,
      absolute_config_path=CONFIG_PATH,
      run_fn=run_single,
      sweep_fn=sweep,
      folder=os.environ.get(
          'RL_RESULTS_DIR', '/tmp/rl_results_dir')
  )


if __name__ == '__main__':
  main()
