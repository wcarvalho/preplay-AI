"""

TESTING:
JAX_DEBUG_NANS=True \
JAX_DISABLE_JIT=1 \
HYDRA_FULL_ERROR=1 JAX_TRACEBACK_FILTERING=off python -m ipdb -c continue craftax_trainer.py \
  app.debug=True \
  app.wandb=False \
  app.search=ql

RUNNING ON SLURM:
python craftax_trainer.py \
  app.parallel=slurm \
  app.search=usfa
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

MAX_SCORE = 226


import wandb
from functools import partial
from jaxneurorl import loggers
from jaxneurorl import utils
from jaxneurorl import launcher
from jaxneurorl.agents import value_based_pqn as vpq
from jaxneurorl.agents import value_based_basics as vbb
from typing import Any, Callable, Dict, Union, Optional
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.constants import Achievement

import jax
import chex
from flax import struct
import functools
import jax.numpy as jnp
import time


import hydra
from omegaconf import DictConfig


from safetensors.flax import save_file
from flax.traverse_util import flatten_dict

import numpy as np


from jaxneurorl.agents import value_based_pqn as pqn
from jaxneurorl.agents import value_based_basics as vbb
from jaxneurorl.wrappers import TimestepWrapper

import housemaze_experiments
import craftax_observer
import networks
import offtask_dyna
import qlearning_craftax
import qlearning_sf_aux_craftax
import usfa_craftax as usfa

@struct.dataclass
class AlgorithmConstructor:
  make_agent: Callable
  make_optimizer: Callable
  make_loss_fn_class: Callable
  make_actor: Callable


def get_pqn_fns(config):
  encoder = functools.partial(
    networks.CraftaxObsEncoder,
    #pqn.MLP,
    hidden_dim=config["MLP_HIDDEN_DIM"],
    num_layers=config['NUM_MLP_LAYERS'],
    activation=config['ACTIVATION'],
    norm_type=config.get('NORM_TYPE', 'batch_norm'),
  )

  return AlgorithmConstructor(
      make_agent=functools.partial(
         pqn.make_agent, ObsEncoderCls=encoder),
      make_optimizer=pqn.make_optimizer,
      make_loss_fn_class=pqn.make_loss_fn_class,
      make_actor=pqn.make_actor,
  )

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
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float],
        params=None,
    ):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )

        return obs, state, reward, done, info

class OptimisticResetVecEnvWrapper(object):
    """
    Provides efficient 'optimistic' resets.
    The wrapper also necessarily handles the batching of environment steps and resetting.
    reset_ratio: the number of environment workers per environment reset.  Higher means more efficient but a higher
    chance of duplicate resets.
    """

    def __getattr__(self, name):
      # provide proxy access to regular attributes of wrapped object
        return getattr(self._env, name)

    def __init__(self, env, num_envs: int, reset_ratio: int):
        self._env = env

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
        timestep = self.reset_fn(rngs, params)
        return timestep

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, prior_timestep, action, params=None):

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        timestep_st = self.step_fn(
            rngs, prior_timestep, action, params)

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_resets)
        timestep_re = self.reset_fn(rngs, params)

        rng, _rng = jax.random.split(rng)
        reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)

        # NOTE: CHANGE: if PRIOR timestep is last, then we are resetting
        # before, if CURRENT timestep is last, then we are resetting
        done = prior_timestep.last()
        being_reset = jax.random.choice(
            _rng,
            jnp.arange(self.num_envs),
            shape=(self.num_resets,),
            p=done,
            replace=False,
        )
        reset_indexes = reset_indexes.at[being_reset].set(
            jnp.arange(self.num_resets))

        timestep_re = jax.tree.map(lambda x: x[reset_indexes], timestep_re)

        # Auto-reset environment based on termination
        def auto_reset(done, timestep_re, timestep_st):
            return jax.tree.map(
                lambda x, y: jax.lax.select(done, x, y), timestep_re, timestep_st
            )

        timestep = jax.vmap(auto_reset)(
            done, timestep_re, timestep_st)

        return timestep


def craftax_experience_logger(
      train_state,
      observer_state,
      key: str = 'train',
      num_seeds: int = None,
      trajectory: Optional[struct.PyTreeNode] = None,
      **kwargs):

    key = f'{key}-{num_seeds}' if num_seeds is not None else key
    def callback(ts, os, traj):
        # [Time, Num Envs, Achievements]
        log_state = traj.timestep.state
        achievements = traj.timestep.state.env_state.achievements
        done = traj.timestep.last()

        achievements = achievements * done[:,:,None] * 100.0

        infos = {}
        infos["0.avg_episode_return"] = log_state.returned_episode_returns*100.0
        infos["0.avg_episode_length"] = log_state.returned_episode_lengths
        for achievement in Achievement:
          name = f"Achievements/{achievement.name.lower()}"
          infos[f"1.{name}"] = achievements[achievement.value]

        #infos["timestep"] = log_state.timestep
        infos["returned_episode"] = done
        metrics = jax.tree_map(
                lambda x: (x * infos["returned_episode"]).sum()
                / (1e-5+infos["returned_episode"].sum()),
                infos,
            )
        metrics = {f'{key}/{k}': v for k, v in metrics.items()}

        if wandb.run is not None:
          wandb.log(metrics)

    jax.debug.callback(callback, train_state, observer_state, trajectory)

def make_logger(
        config: dict,
        env,
        env_params,
        learner_log_extra: Optional[Callable[[Any], Any]] = None
):

  return loggers.Logger(
      gradient_logger=loggers.default_gradient_logger,
      learner_logger=loggers.default_learner_logger,
      experience_logger=partial(craftax_experience_logger,
                                num_seeds=config.get('NUM_ENV_SEEDS', None)),
      learner_log_extra=partial(learner_log_extra, config=config) if learner_log_extra is not None else None,
  )

def run_single(
        config: dict,
        save_path: str = None):

    rng = jax.random.PRNGKey(config["SEED"])
    config["TEST_NUM_ENVS"] = config.get("TEST_NUM_ENVS", None) or config["NUM_ENVS"]

    if config['ENV'] == 'classic':
      vec_env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
      env_params = vec_env.default_params
      test_env_params = vec_env.default_params

    elif config['ENV'] == 'craftax':
      vec_env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
      env_params = vec_env.default_params
      test_env_params = vec_env.default_params

    elif config['ENV'] == 'craftax-gen':
      from craftax_env import CraftaxSymbolicEnvNoAutoReset
      config['STRUCTURED_INPUTS'] = True
      config['NUM_ENV_SEEDS'] = config.get('NUM_ENV_SEEDS', 10_000)

      env = CraftaxSymbolicEnvNoAutoReset()

      env_params = env.default_params.replace(
         reset_seeds=tuple(np.arange(config['NUM_ENV_SEEDS'])))
      test_env_params = env.default_params.replace(
         reset_seeds=tuple(np.arange(config['NUM_ENV_SEEDS'], config['NUM_ENV_SEEDS'] + config['TEST_NUM_ENVS'])))

    else:
      raise NotImplementedError(config["ENV"])

    vec_env = OptimisticResetVecEnvWrapper(
        TimestepWrapper(LogWrapper(env), autoreset=False),
        num_envs=config["NUM_ENVS"],
        reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
    )


    if config['ALG'] == 'qlearning':
        train_fn = vbb.make_train(
          config=config,
          env=vec_env,
          make_agent=qlearning_craftax.make_craftax_agent,
          make_optimizer=qlearning_craftax.make_optimizer,
          make_loss_fn_class=qlearning_craftax.make_loss_fn_class,
          make_actor=qlearning_craftax.make_actor,
          make_logger=make_logger,
          train_env_params=env_params,
          test_env_params=test_env_params,
          ObserverCls=craftax_observer.Observer,
          vmap_env=False,
        )
    elif config['ALG'] == 'qlearning_sf_aux':
        train_fn = vbb.make_train(
          config=config,
          env=vec_env,
          make_agent=qlearning_sf_aux_craftax.make_craftax_agent,
          make_optimizer=qlearning_sf_aux_craftax.make_optimizer,
          make_loss_fn_class=qlearning_sf_aux_craftax.make_loss_fn_class,
          make_actor=qlearning_sf_aux_craftax.make_actor,
          make_logger=make_logger,
          train_env_params=env_params,
          test_env_params=test_env_params,
          ObserverCls=craftax_observer.Observer,
          vmap_env=False,
        )

    elif config['ALG'] == 'pqn':
        constructor = get_pqn_fns(config)
        train_fn = vpq.make_train(
          config=config,
          env=vec_env,
          make_agent=constructor.make_agent,
          make_optimizer=constructor.make_optimizer,
          make_loss_fn_class=constructor.make_loss_fn_class,
          make_actor=constructor.make_actor,
          make_logger=make_logger,
          train_env_params=env_params,
          test_env_params=test_env_params,
          ObserverCls=craftax_observer.Observer,
          vmap_env=False,
        )
    elif config['ALG'] == 'usfa':
        train_fn = vbb.make_train(
          config=config,
          env=vec_env,
          make_agent=usfa.make_craftax_agent,
          make_optimizer=usfa.make_optimizer,
          make_loss_fn_class=usfa.make_loss_fn_class,
          make_actor=usfa.make_actor,
          make_logger=partial(make_logger,
                              learner_log_extra=usfa.learner_log_extra),
          train_env_params=env_params,
          test_env_params=test_env_params,
          ObserverCls=craftax_observer.Observer,
          vmap_env=False,
        )
    elif config['ALG'] == 'alphazero':
      import mctx
      from jaxneurorl.agents import alphazero
      import alphazero_craftax
      max_value = config.get('MAX_VALUE', 10)
      num_bins = config['NUM_BINS']

      discretizer = utils.Discretizer(
          max_value=max_value,
          num_bins=num_bins,
          min_value=-max_value)

      num_train_simulations = config.get('NUM_SIMULATIONS', 4)

      mcts_policy = functools.partial(
          mctx.gumbel_muzero_policy,
          max_depth=config.get('MAX_SIM_DEPTH', None),
          num_simulations=num_train_simulations,
          gumbel_scale=config.get('GUMBEL_SCALE', 1.0))
      eval_mcts_policy = functools.partial(
          mctx.gumbel_muzero_policy,
          max_depth=config.get('MAX_SIM_DEPTH', None),
          num_simulations=config.get(
            'NUM_EVAL_SIMULATIONS', num_train_simulations),
          gumbel_scale=config.get('GUMBEL_SCALE', 1.0))

      train_fn = vbb.make_train(
          config=config,
          env=vec_env,
          make_agent=functools.partial(alphazero_craftax.make_agent,
            model_env=TimestepWrapper(env, autoreset=False)),
          make_optimizer=alphazero.make_optimizer,
          make_loss_fn_class=functools.partial(
              alphazero.make_loss_fn_class,
              discretizer=discretizer),
          make_actor=functools.partial(
              alphazero.make_actor,
              discretizer=discretizer,
              mcts_policy=mcts_policy,
              eval_mcts_policy=eval_mcts_policy),
          make_logger=make_logger,
          train_env_params=env_params,
          test_env_params=test_env_params,
          ObserverCls=craftax_observer.Observer,
          vmap_env=False,
      )

    else:
      raise NotImplementedError(config['ALG'])
    
    start_time = time.time()
    train_vjit = jax.jit(jax.vmap(train_fn))

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
  search = search or 'ql'
  if search == 'ql':
    sweep_config = {
        'metric': {
            'name': 'evaluator_performance/0.0 avg_episode_return',
            'goal': 'maximize',
        },
        'parameters': {
            "NUM_ENV_SEEDS": {'values': [100, 10_000]},
            #"AUX_COEFF": {'values': [1.0]},
            #"NUM_ENVS": {'values': [32, 64]},
        },
        'overrides': ['alg=ql', 'rlenv=craftax-10m', 'user=wilka'],
        'group': 'ql-6',
    }
  elif search == 'pqn':
    sweep_config = {
        'metric': {
            'name': 'evaluator_performance/0.0 avg_episode_return',
            'goal': 'maximize',
        },
        'parameters': {
            "FIXED_EPSILON": {'values': [0, 2]},
            "LAMBDA": {'values': [.95, .5]},
            "MAX_GRAD_NORM": {'values': [.5, 10]},
            "LR_LINEAR_DECAY": {'values': [True, False]},
            "LR": {'values': [.001, .0003]},
            "NUM_ENVS": {'values': [256]},
        },
        'overrides': ['alg=pqn-craftax', 'rlenv=craftax-10m', 'user=wilka'],
        'group': 'pqn-7',
    }
  elif search == 'usfa':
    sweep_config = {
        'metric': {
            'name': 'evaluator_performance/0.0 avg_episode_return',
            'goal': 'maximize',
        },
        'parameters': {
            "NUM_ENV_SEEDS": {'values': [100, 10_000]},
            "TX_PAIR": {'values': ['hyperbolic']},
            #"NSAMPLES": {'values': [1, 5]},
            "AUX_COEFF": {'values': [1.0, .1, .01]},
        },
        'overrides': ['alg=usfa', 'rlenv=craftax-10m', 'user=wilka'],
        'group': 'usfa-6',
    }
  elif search == 'alphazero':
    sweep_config = {
        'metric': {
            'name': 'evaluator_performance/0.0 avg_episode_return',
            'goal': 'maximize',
        },
        'parameters': {
            "NUM_ENV_SEEDS": {'values': [100, 10_000]},
            "NUM_SIMULATIONS": {'values': [2, 4]},
        },
        'overrides': ['alg=alphazero', 'rlenv=craftax-10m', 'user=wilka'],
        'group': 'alphazero-1',
    }
  elif search == 'ql_sf':
    sweep_config = {
        'metric': {
            'name': 'evaluator_performance/0.0 avg_episode_return',
            'goal': 'maximize',
        },
        'parameters': {
            "NUM_ENV_SEEDS": {'values': [100, 10_000]},
            "ALG": {'values': ['qlearning_sf_aux']},
            #"NUM_ENVS": {'values': [32, 64]},
        },
        'overrides': ['alg=ql', 'rlenv=craftax-10m', 'user=wilka'],
        'group': 'ql-sf-2',
    }
  else:
    raise NotImplementedError(search)

  return sweep_config

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'configs'))
@hydra.main(
    version_base=None,
    config_path=CONFIG_PATH,
    config_name="config")
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
