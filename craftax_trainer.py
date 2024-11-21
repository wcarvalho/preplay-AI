"""

TESTING:
JAX_DEBUG_NANS=True \
JAX_DISABLE_JIT=1 \
HYDRA_FULL_ERROR=1 JAX_TRACEBACK_FILTERING=off python -m ipdb -c continue craftax_trainer.py \
  app.debug=True \
  app.wandb=False \
  app.search=pqn

RUNNING ON SLURM:
python housemaze_trainer.py \
  app.parallel=slurm_wandb \
  app.search=dynaq_shared
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from functools import partial
from housemaze.human_dyna import multitask_env
from housemaze import utils as housemaze_utils
from housemaze import renderer
from jaxneurorl import loggers
from jaxneurorl import utils
from jaxneurorl import launcher
from jaxneurorl.agents import value_based_pqn as vpq
from jaxneurorl.agents import value_based_basics as vbb
from typing import Any, Callable, Dict, Union, Optional
from craftax.craftax_env import make_craftax_env_from_name


import jax
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
import observers as humansf_observers
import networks
import offtask_dyna
import usfa
import qlearning
import alphazero

@struct.dataclass
class AlgorithmConstructor:
  make_agent: Callable
  make_optimizer: Callable
  make_loss_fn_class: Callable
  make_actor: Callable


def get_qlearning_fns(config, num_categories=10_000,):
  HouzemazeObsEncoder = functools.partial(
      networks.CraftaxObsEncoder,
      num_categories=num_categories,
      embed_hidden_dim=config["EMBED_HIDDEN_DIM"],
      mlp_hidden_dim=config["MLP_HIDDEN_DIM"],
      num_embed_layers=config["NUM_EMBED_LAYERS"],
      num_mlp_layers=config['NUM_MLP_LAYERS'],
      activation=config['ACTIVATION'],
      norm_type=config.get('NORM_TYPE', 'none'),
  )

  return AlgorithmConstructor(
      make_agent=functools.partial(
          qlearning.make_agent,
          ObsEncoderCls=HouzemazeObsEncoder,
      ),
      make_optimizer=qlearning.make_optimizer,
      make_loss_fn_class=qlearning.make_loss_fn_class,
      make_actor=qlearning.make_actor,
  )

def get_pqn_fns(config, num_categories=10_000,):
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

        obs_re = obs_re[reset_indexes]
        state_re = jax.tree.map(lambda x: x[reset_indexes], state_re)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree.map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return state, obs

        state, obs = jax.vmap(auto_reset)(
            done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


def run_single(
        config: dict,
        save_path: str = None):

    rng = jax.random.PRNGKey(config["SEED"])

    if config["rlenv"]['ENV_NAME'] == 'classic':
       env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
    elif config["rlenv"]['ENV_NAME'] == 'craftax':
      env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    else:
      raise NotImplementedError(config["ENV"])
    
    env = OptimisticResetVecEnvWrapper(
        env,
        num_envs=config["NUM_ENVS"],
        reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
    )
    test_env = OptimisticResetVecEnvWrapper(
        env,
        num_envs=config["TEST_NUM_ENVS"],
        reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["TEST_NUM_ENVS"]),
    )
    env = TimestepWrapper(env, autoreset=False)
    test_env = TimestepWrapper(test_env, autoreset=False)

    env_params = env.default_params
    test_env_params = test_env.default_params

    if config['ALG'] == 'qlearning':
        constructor = get_qlearning_fns(config)
        train_fn = vbb.make_train(
          config=config,
          env=env,
          make_agent=constructor.make_agent,
          make_optimizer=constructor.make_optimizer,
          make_loss_fn_class=constructor.make_loss_fn_class,
          make_actor=constructor.make_actor,
          train_env_params=env_params,
          test_env_params=test_env_params,
          vmap_env=False,
        )
    elif config['ALG'] == 'pqn':
        constructor = get_pqn_fns(config)
        train_fn = vpq.make_train(
          config=config,
          env=env,
          make_agent=constructor.make_agent,
          make_optimizer=constructor.make_optimizer,
          make_loss_fn_class=constructor.make_loss_fn_class,
          make_actor=constructor.make_actor,
          train_env_params=env_params,
          test_env_params=test_env_params,
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
            "env.exp": {'values': ['exp2']},
            "TOTAL_TIMESTEPS": {'values': [int(1e9), int(1e6)]},
        },
        'overrides': ['alg=ql', 'rlenv=craftax', 'user=wilka'],
        'group': 'ql-1',
    }
  elif search == 'pqn':
    sweep_config = {
        'metric': {
            'name': 'evaluator_performance/0.0 avg_episode_return',
            'goal': 'maximize',
        },
        'parameters': {
            "env.exp": {'values': ['exp2']},
            "TOTAL_TIMESTEPS": {'values': [int(1e9), int(1e6)]},
        },
        'overrides': ['alg=pqn-craftax', 'rlenv=craftax', 'user=wilka'],
        'group': 'pqn-1',
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
