"""

TESTING:
JAX_DEBUG_NANS=True \
JAX_DISABLE_JIT=1 \
HYDRA_FULL_ERROR=1 JAX_TRACEBACK_FILTERING=off python -m ipdb -c continue jaxmaze_trainer.py \
  app.debug=True \
  app.wandb=False \
  app.search=preplay

RUNNING ON SLURM:
python jaxmaze_trainer.py \
  app.parallel=slurm_wandb \
  app.search=dynaq_shared
"""

from typing import Any, Callable, Dict, Union, Optional, Tuple


import os
import jax
from flax import struct
import functools
import jax.numpy as jnp
import time
import pickle


import hydra
from omegaconf import DictConfig


from safetensors.flax import save_file, load_file
from flax.traverse_util import flatten_dict, unflatten_dict

import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


from jaxneurorl.agents import value_based_basics as vbb
from jaxneurorl import launcher
from jaxneurorl import loggers

import qlearning_housemaze
import usfa_housemaze
import multitask_preplay_housemaze
import dyna_craftax
import multitask_preplay_craftax_v2
import housemaze_observer as humansf_observers
from housemaze.human_dyna import experiments as housemaze_experiments

from housemaze import renderer
from housemaze import utils as housemaze_utils
from housemaze.human_dyna import multitask_env


def make_logger(
  config: dict,
  env: multitask_env.HouseMaze,
  env_params: multitask_env.EnvParams,
  action_names: dict,
  render_fn: Callable = None,
  extract_task_info: Callable = None,
  get_task_name: Callable = None,
  learner_log_extra: Optional[Callable[[Any], Any]] = None,
):
  return loggers.Logger(
    gradient_logger=loggers.default_gradient_logger,
    learner_logger=loggers.default_learner_logger,
    experience_logger=functools.partial(
      humansf_observers.experience_logger,
      action_names=action_names,
      extract_task_info=extract_task_info,
      get_task_name=get_task_name,
      render_fn=render_fn,
      max_len=config["MAX_EPISODE_LOG_LEN"],
    ),
    learner_log_extra=learner_log_extra,
  )


@struct.dataclass
class AlgorithmConstructor:
  make_agent: Callable
  make_optimizer: Callable
  make_loss_fn_class: Callable
  make_actor: Callable


def extract_task_info(timestep: multitask_env.TimeStep):
  state = timestep.state
  return {
    "map_idx": state.map_idx,
    "current_label": state.current_label,
    "is_train_task": state.is_train_task,
    "category": state.task_object,
  }


def task_from_variables(variables, keys, label2name):
  current_label = variables["current_label"]
  category = keys[variables["category"]]
  is_train_task = variables["is_train_task"]
  label = "1.train" if is_train_task else "0.TEST"
  setting = label2name.get(int(current_label))
  exp_name = {
    "big_m3_maze1": "two_paths",
    "big_m1_maze3": "shortcut",
    "big_m1_maze3_shortcut": "shortcut",
  }.get(setting, setting)
  # return f"{label} \n {exp_name} - {category}"
  return f"{label} \n {exp_name}"


def save_training_state(
  params: Dict, config: Dict, save_path: str, alg_name: str
) -> None:
  """Save model parameters and config to disk.

  Args:
      params: Model parameters to save
      config: Configuration dictionary to save
      save_path: Directory to save files in
      alg_name: Name of algorithm for file naming
  """
  os.makedirs(save_path, exist_ok=True)

  # Save parameters
  param_path = os.path.join(save_path, f"{alg_name}.safetensors")
  flattened_dict = flatten_dict(params, sep=",")
  save_file(flattened_dict, param_path)
  print(f"Parameters saved in {param_path}")

  # Save config
  config_path = os.path.join(save_path, f"{alg_name}.config")
  import pickle

  with open(config_path, "wb") as f:
    pickle.dump(config, f)
  print(f"Config saved in {config_path}")


def load_training_state(
  save_path: str, alg_name: str
) -> Tuple[Optional[Dict], Optional[Dict]]:
  """Load model parameters and config from disk if they exist.

  Args:
      save_path: Directory containing saved files
      alg_name: Name of algorithm for file naming

  Returns:
      Tuple of (parameters dict, config dict) if files exist, else (None, None)
  """
  if save_path is None:
    return None, None

  param_path = os.path.join(save_path, f"{alg_name}.safetensors")
  config_path = os.path.join(save_path, f"{alg_name}.config")

  if not (os.path.exists(param_path) and os.path.exists(config_path)):
    return None, None

  print(f"Loading parameters from {param_path}")
  # Load and unflatten parameters
  flattened_params = load_file(param_path)
  params = unflatten_dict(flattened_params, sep=",")

  # Load config
  with open(config_path, "rb") as f:
    saved_config = pickle.load(f)
  print("Loaded previous training configuration")

  return params, saved_config


def run_single(config: dict, save_path: str = None):
  # Try to load previous training state
  initial_params = None
  if save_path is not None:
    initial_params, _ = load_training_state(save_path, config["ALG"])

  rng = jax.random.PRNGKey(config["SEED"])
  ###################
  # load data
  ###################
  exp = config["rlenv"]["ENV_KWARGS"].pop("exp")
  try:
    exp_fn = getattr(housemaze_experiments, exp, None)
  except Exception as e:
    raise RuntimeError(e)
  env_params, test_env_params, task_objects, idx2maze = exp_fn(config)

  image_dict = housemaze_utils.load_image_dict()
  # Reshape the images to separate the blocks
  images = image_dict["images"]
  reshaped = images.reshape(len(images), 8, 4, 8, 4, 3)

  # Take the mean across the block dimensions
  image_dict["images"] = reshaped.mean(axis=(2, 4)).astype(np.uint8)

  ###################
  # load env
  ###################
  float_obs = config.get("FLOAT_OBS", False)
  env_params = env_params.replace(float_obs=float_obs)
  test_env_params = test_env_params.replace(float_obs=float_obs)

  if config["ALG"] == "usfa":
    from housemaze.human_dyna import sf_task_runner

    task_runner = sf_task_runner.TaskRunner(
      task_objects=task_objects,
      radius=config.get("VIS_RADIUS", 5),
      vis_coeff=config.get("VIS_COEFF", 0.0),
    )
    # def success_fn(timestep):
    #  features = timestep.observation.state_features
    #  task_w = timestep.observation.task_w
    #  # only first half count. 2nd half are about visibility.
    #  half = len(features)//2
    #  import ipdb; ipdb.set_trace()
    #  return (features[:half]*task_w[:half]).sum(-1)
  else:
    task_runner = multitask_env.TaskRunner(task_objects=task_objects)
    # success_fn = lambda timestep: timestep.rewards > .5
  keys = image_dict["keys"]
  env = multitask_env.HouseMaze(
    task_runner=task_runner,
    num_categories=200,
  )

  env = housemaze_utils.AutoResetWrapper(env)

  ###################
  ## custom observer
  ###################
  action_names = {action.value: action.name for action in env.action_enum()}

  def housemaze_render_fn(state: multitask_env.EnvState):
    return renderer.create_image_from_grid(
      state.grid, state.agent_pos, state.agent_dir, image_dict
    )

  observer_class = functools.partial(
    humansf_observers.TaskObserver,
    extract_task_info=extract_task_info,
    action_names=action_names,
    # success_fn=success_fn,
  )

  get_task_name = functools.partial(task_from_variables, keys=keys, label2name=idx2maze)
  ##################
  # algorithms
  ##################
  alg_name = config["ALG"]

  train_objects = env_params.reset_params.train_objects[0]
  test_objects = env_params.reset_params.test_objects[0]
  train_tasks = jnp.array([env.task_runner.task_vector(o) for o in train_objects])
  test_tasks = jnp.array([env.task_runner.task_vector(o) for o in test_objects])
  all_tasks = jnp.concatenate((train_tasks, test_tasks), axis=0)

  if alg_name == "qlearning":
    train_fn = vbb.make_train(
      config=config,
      env=env,
      save_path=save_path,
      train_env_params=env_params,
      test_env_params=test_env_params,
      ObserverCls=observer_class,
      initial_params=initial_params,
      make_agent=qlearning_housemaze.make_housemaze_agent,
      make_optimizer=qlearning_housemaze.make_optimizer,
      make_loss_fn_class=qlearning_housemaze.make_loss_fn_class,
      make_actor=qlearning_housemaze.make_actor,
      make_logger=functools.partial(
        make_logger,
        render_fn=housemaze_render_fn,
        extract_task_info=extract_task_info,
        get_task_name=get_task_name,
        action_names=action_names,
        learner_log_extra=functools.partial(
          qlearning_housemaze.learner_log_extra,
          config=config,
          action_names=action_names,
          extract_task_info=extract_task_info,
          get_task_name=get_task_name,
          render_fn=housemaze_render_fn,
        ),
      ),
    )
  elif alg_name == "usfa":
    train_fn = vbb.make_train(
      config=config,
      env=env,
      save_path=save_path,
      train_env_params=env_params,
      test_env_params=test_env_params,
      ObserverCls=observer_class,
      initial_params=initial_params,
      make_agent=functools.partial(
        usfa_housemaze.make_agent,
        train_tasks=train_tasks,
        all_tasks=all_tasks,
      ),
      make_logger=functools.partial(
        make_logger,
        render_fn=housemaze_render_fn,
        extract_task_info=extract_task_info,
        get_task_name=get_task_name,
        action_names=action_names,
        learner_log_extra=functools.partial(
          usfa_housemaze.learner_log_extra,
          config=config,
          action_names=action_names,
          extract_task_info=extract_task_info,
          get_task_name=get_task_name,
          render_fn=housemaze_render_fn,
        ),
      ),
    )
  elif alg_name in ("dyna"):
    train_fn = dyna_craftax.make_train(
      config=config,
      env=env,
      save_path=save_path,
      train_env_params=env_params,
      test_env_params=test_env_params,
      ObserverCls=observer_class,
      initial_params=initial_params,
      make_agent=dyna_craftax.make_jaxmaze_agent,
      make_logger=functools.partial(
        make_logger,
        render_fn=housemaze_render_fn,
        extract_task_info=extract_task_info,
        get_task_name=get_task_name,
        action_names=action_names,
      ),
    )
  elif alg_name in ("preplay"):
    train_fn = multitask_preplay_craftax_v2.make_train_jaxmaze_multigoal(
      config=config,
      env=env,
      save_path=save_path,
      train_env_params=env_params,
      test_env_params=test_env_params,
      ObserverCls=observer_class,
      initial_params=initial_params,
      model_env=env,
      make_logger=functools.partial(
        make_logger,
        render_fn=housemaze_render_fn,
        extract_task_info=extract_task_info,
        get_task_name=get_task_name,
        action_names=action_names,
        learner_log_extra=functools.partial(
          multitask_preplay_housemaze.learner_log_extra,
          config=config,
          action_names=action_names,
          extract_task_info=extract_task_info,
          get_task_name=get_task_name,
          render_fn=housemaze_render_fn,
          sim_idx=0,
        ),
      ),
      task_objects=task_objects,
      all_tasks=all_tasks,
    )

  else:
    raise NotImplementedError(alg_name)

  start_time = time.time()
  train_vjit = jax.jit(jax.vmap(train_fn))

  rngs = jax.random.split(rng, config["NUM_SEEDS"])
  outs = jax.block_until_ready(train_vjit(rngs))
  elapsed_time = time.time() - start_time
  print("Elapsed time: {:.2f} seconds".format(elapsed_time))

  # ---------------
  # save model weights
  # ---------------
  if save_path is not None:
    model_state = outs["runner_state"][0]
    params = jax.tree_map(lambda x: x[0], model_state.params)
    save_training_state(params, config, save_path, config["ALG"])


def sweep(search: str = ""):
  search = search or "ql"
  if search == "ql":
    sweep_config = {
      "metric": {
        "name": "evaluator_performance/0.0 avg_episode_return",
        "goal": "maximize",
      },
      "parameters": {
        "SEED": {"values": list(range(1))},
        "env.exp": {"values": ["exp4"]},
        "STEP_COST": {"values": [1e-4, 1e-3, 0]},
        # "FLOAT_OBS": {"values": [True]},
        "AGENT_RNN_DIM": {"values": [1024]},
      },
      "overrides": ["alg=ql", "rlenv=housemaze", "user=wilka"],
      "group": "ql-11-fixed-obs-large-rnn-float",
    }
  elif search == "ql2":
    sweep_config = {
      "metric": {
        "name": "evaluator_performance/0.0 avg_episode_return",
        "goal": "maximize",
      },
      "parameters": {
        "SEED": {"values": list(range(1))},
        "env.exp": {"values": ["exp4"]},
        "STEP_COST": {"values": [1e-4, 1e-3, 0]},
        "FLOAT_OBS": {"values": [False]},
        "NUM_EMBED_LAYERS": {"values": [1, 0]},
        # "AGENT_RNN_DIM": {"values": [1024]},
      },
      "overrides": ["alg=ql", "rlenv=housemaze", "user=wilka"],
      "group": "ql-11-fixed-obs-large-rnn-cat",
    }
  elif search == "usfa":
    sweep_config = {
      "metric": {
        "name": "evaluator_performance/0.0 avg_episode_return",
        "goal": "maximize",
      },
      "parameters": {
        "SEED": {"values": list(range(1, 2))},
        "env.exp": {"values": ["exp4"]},
        # "STEP_COST": {"values": [0.01, 0.005, 0.001]},
        "LEARN_VECTORS": {"values": ["TRAIN", "ALL_TASKS"]},
        # "VIS_COEFF": {"values": [0.0, 0.1]},
      },
      "overrides": ["alg=usfa", "rlenv=housemaze", "user=wilka"],
      "group": "usfa-landmark-3",
    }
  elif search == "dyna":
    sweep_config = {
      "metric": {
        "name": "evaluator_performance/0.0 avg_episode_return",
        "goal": "maximize",
      },
      "parameters": {
        "ALG": {"values": ["dyna"]},
        "SEED": {"values": list(range(1))},
        "env.exp": {"values": ["exp4"]},
        "COMBINE_REAL_SIM": {"values": [True]},
        # "SIM_EPSILON_SETTING": {"values": [3]},
        # "OFFTASK_SIMULATION": {"values": [False]},
        # "TOTAL_TIMESTEPS": {"values": [100_000_000]},
      },
      "overrides": ["alg=preplay_jaxmaze", "rlenv=housemaze", "user=wilka"],
      "group": "dyna-new-4",
    }
  elif search == "preplay":
    sweep_config = {
      "metric": {
        "name": "evaluator_performance/0.0 avg_episode_return",
        "goal": "maximize",
      },
      "parameters": {
        "ALG": {"values": ["preplay"]},
        "SEED": {"values": list(range(3))},
        "env.exp": {"values": ["exp4"]},
        "MAINQ_COEFF": {"values": [1e-1, 1e-2]},
        "OFFTASK_COEFF": {"values": [1e-1, 1e-2]},
        # "GAMMA": {"values": [0.99, 0.991]},
        "LR": {"values": [0.001, 0.0003]},
        # "KNOWN_OFFTASK_GOAL": {"values": [False, True]},
      },
      "overrides": ["alg=preplay_jaxmaze", "rlenv=housemaze", "user=wilka"],
      "group": "preplay-eps-1",
    }
  # elif search == "dynaq_shared":
  #  sweep_config = {
  #    "metric": {
  #      "name": "evaluator_performance/0.0 avg_episode_return",
  #      "goal": "maximize",
  #    },
  #    "parameters": {
  #      "ALG": {"values": ["dynaq_shared"]},
  #      "SEED": {"values": list(range(1, 2))},
  #      "env.exp": {"values": ["exp4"]},
  #      "GAMMA": {"values": [0.99, 0.992]},
  #      "STEP_COST": {"values": [0.0001]},
  #      "NUM_Q_LAYERS": {"values": [1, 2, 3]},
  #      "Q_HIDDEN_DIM": {"values": [512, 1024]},
  #    },
  #    "overrides": ["alg=preplay", "rlenv=housemaze", "user=wilka"],
  #    "group": "dynaq-big-6",
  #  }
  # =================================================================
  # final
  # =================================================================
  elif search == "ql-final":
    sweep_config = {
      "metric": {
        "name": "evaluator_performance/0.0 avg_episode_return",
        "goal": "maximize",
      },
      "parameters": {
        "SEED": {"values": list(range(1, 11))},
        "env.exp": {"values": ["exp4"]},
      },
      "overrides": ["alg=ql", "rlenv=housemaze", "user=wilka"],
      "group": "ql-final-rotations-2",
    }
  elif search == "usfa-final":
    sweep_config = {
      "metric": {
        "name": "evaluator_performance/0.0 avg_episode_return",
        "goal": "maximize",
      },
      "parameters": {
        "SEED": {"values": list(range(1, 11))},
        "env.exp": {"values": ["exp4"]},
      },
      "overrides": ["alg=usfa", "rlenv=housemaze", "user=wilka"],
      "group": "usfa-final-rotations-2",
    }
  elif search == "dyna-final":
    sweep_config = {
      "metric": {
        "name": "evaluator_performance/0.0 avg_episode_return",
        "goal": "maximize",
      },
      "parameters": {
        "ALG": {"values": ["dyna"]},
        "SEED": {"values": list(range(1, 11))},
        "env.exp": {"values": ["exp4"]},
      },
      "overrides": ["alg=dyna_jaxmaze", "rlenv=housemaze", "user=wilka"],
      "group": "dyna-final-rotations-4",
    }
  elif search == "preplay-old-final":
    sweep_config = {
      "metric": {
        "name": "evaluator_performance/0.0 avg_episode_return",
        "goal": "maximize",
      },
      "parameters": {
        "ALG": {"values": ["dynaq_shared"]},
        "SEED": {"values": list(range(6, 11))},
        "env.exp": {"values": ["exp4"]},
      },
      "overrides": ["alg=preplay", "rlenv=housemaze", "user=wilka"],
      "group": "preplay-old-final-rotations-4",
    }
  elif search == "preplay-final":
    sweep_config = {
      "metric": {
        "name": "evaluator_performance/0.0 avg_episode_return",
        "goal": "maximize",
      },
      "parameters": {
        "ALG": {"values": ["preplay"]},
        "SEED": {"values": list(range(1, 11))},
        "env.exp": {"values": ["exp4"]},
      },
      "overrides": ["alg=preplay_jaxmaze", "rlenv=housemaze", "user=wilka"],
      "group": "preplay-final-rotations-6",
    }

  elif search == "preplay-ablation":
    sweep_config = {
      "metric": {
        "name": "evaluator_performance/0.0 avg_episode_return",
        "goal": "maximize",
      },
      "parameters": {
        "ALG": {"values": ["preplay"]},
        "SEED": {"values": list(range(1, 5))},
        "MAINQ_COEFF": {"values": [0.0]},
        "env.exp": {"values": ["exp4"]},
      },
      "overrides": ["alg=preplay_jaxmaze", "rlenv=housemaze", "user=wilka"],
      "group": "preplay-ablation-rotations-5",
    }

  else:
    raise NotImplementedError(search)

  return sweep_config


CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "configs"))


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(config: DictConfig):
  launcher.run(
    config,
    trainer_filename=__file__,
    absolute_config_path=CONFIG_PATH,
    run_fn=run_single,
    sweep_fn=sweep,
    folder=os.environ.get("RL_RESULTS_DIR", "/tmp/rl_results_dir"),
  )


if __name__ == "__main__":
  main()
