"""

TESTING:
JAX_DEBUG_NANS=True \
JAX_DISABLE_JIT=1 \
HYDRA_FULL_ERROR=1 JAX_TRACEBACK_FILTERING=off python -m ipdb -c continue projects/humansf/housemaze_trainer.py \
  app.debug=True \
  app.wandb=False \
  app.search=usfa

RUNNING ON SLURM:
python projects/humansf/housemaze_trainer.py \
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


from projects.humansf import housemaze_experiments
from projects.humansf import observers as humansf_observers
from projects.humansf import networks
from projects.humansf import offtask_dyna
from projects.humansf import usfa
from projects.humansf import qlearning
from projects.humansf import alphazero

from projects.humansf.housemaze_trainer import AlgorithmConstructor


def get_qlearning_fns(config, num_categories=10_000,):
  HouzemazeObsEncoder = functools.partial(
      networks.CategoricalHouzemazeObsEncoder,
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
        state_re = jax.tree_map(lambda x: x[reset_indexes], state_re)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree_map(
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

    #env = CraftaxSymbolicEnvNoAutoReset()

    #from craftax.craftax.renderer import render_craftax_pixels
    #render = jax.jit(partial(render_craftax_pixels,
    #                 block_pixel_size=constants.BLOCK_PIXEL_SIZE_IMG))
    ## def render_craftax(state):


    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env_params = env.default_params
    env = OptimisticResetVecEnvWrapper(
        env,
        num_envs=config["NUM_ENVS"],
        reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
    )
    test_env = OptimisticResetVecEnvWrapper(
        env,
        num_envs=config["TEST_NUM_ENVS"],

    #seed = 7
    #rng = jax.random.PRNGKey(seed)

    #obs, env_state = env.reset_env(rng, env_params)
    #image = render(env_state)
    
    # plt.imshow(image)




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
            #"setting": {'values': [40_000_000]},
        },
        'overrides': ['alg=ql', 'rlenv=craftax', 'user=wilka'],
        'group': 'ql-1',
    }
  elif search == 'dynaq_shared':
    sweep_config = {
        'metric': {
            'name': 'evaluator_performance/0.0 avg_episode_return',
            'goal': 'maximize',
        },
        'parameters': {
            'ALG': {'values': ['dynaq_shared']},
        },
        'overrides': ['alg=dyna', 'rlenv=craftax', 'user=wilka'],
        'group': 'dynaq-1',
    }
  elif search == 'pqn':
    sweep_config = {
        'metric': {
            'name': 'evaluator_performance/0.0 avg_episode_return',
            'goal': 'maximize',
        },
        'parameters': {
            "env.exp": {'values': [
                'maze3_open',
            ]},
            "BATCH_SIZE": {'values': [512*128, 256*128, 128*128]},
            "NORM_TYPE": {'values': ['layer_norm', 'none']},
            "NORM_QFN": {'values': ['layer_norm', 'none']},
            "TOTAL_TIMESTEPS": {'values': [100_000_000]},
        },
        'overrides': ['alg=pqn', 'rlenv=craftax', 'user=wilka'],
        'group': 'pqn-1',
    }
  elif search == 'alpha':
    sweep_config = {
        'metric': {
            'name': 'evaluator_performance/0.0 avg_episode_return',
            'goal': 'maximize',
        },
        'parameters': {
            "config_name": {'values': ['alpha_housemaze']},
            'TOTAL_TIMESTEPS': {'values': [5e6]},
        }
    }

  else:
    raise NotImplementedError(search)

  return sweep_config


@hydra.main(
    version_base=None,
    config_path='configs',
    config_name="config")
def main(config: DictConfig):
  launcher.run(
      config,
      trainer_filename=__file__,
      config_path='projects/humansf/configs',
      run_fn=run_single,
      sweep_fn=sweep,
      folder=os.environ.get(
          'RL_RESULTS_DIR', '/tmp/rl_results_dir')
  )


if __name__ == '__main__':
  main()
