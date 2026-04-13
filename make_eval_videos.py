"""Generate evaluation videos for trained preplay agents.

Usage:
  uv run python make_eval_videos.py \
    "alg=preplay,tota=2560,tota=50000000,exp=two_paths" \
    --out ~/Desktop/videos/preplay


The script:
1. Auto-discovers the checkpoint on the server via SSH
2. Downloads preplay.safetensors + preplay.config
3. Reconstructs env + agent locally
4. Runs greedy episodes for train (microwave) and test (stove) tasks
5. Creates videos: row 0 = env image, row 1 = growing Q-value heatmap
"""

import argparse
import hashlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax.traverse_util import unflatten_dict
from matplotlib.patches import Rectangle
from safetensors.flax import load_file

from jaxmaze import renderer
from jaxmaze import utils as jaxmaze_utils
from jaxmaze.human_dyna import experiments as jaxmaze_experiments
from jaxmaze.human_dyna import multitask_env_fast
from jaxmaze.utils import actions_from_path, find_optimal_path

from jaxneurorl.agents import value_based_basics as vbb
from networks import CategoricalJaxmazeObsEncoder
from multitask_preplay import DuellingMLP, PreplayAgent

# ─── Server config ───────────────────────────────────────────────────────────
SSH_HOST = "rcfas_login"
SAVE_DATA_BASE = (
  "/n/holylfs06/LABS/kempner_fellow_wcarvalho/"
  "jax_rl_results/jaxmaze_trainer/*/save_data"
)


def find_checkpoints_on_server(wandb_name_pattern: str, seed: int = None) -> list[str]:
  """SSH to server, find checkpoint directories matching a pattern (supports * globs).

  Returns list of (wandb_name, remote_dir, safetensors_fname, group_name) tuples.
  """
  # Use bash glob via ls instead of find for proper wildcard support
  seed_glob = f"seed={seed}" if seed is not None else "seed=*"
  search_pattern = (
    f"{SAVE_DATA_BASE}/*/{wandb_name_pattern}/{seed_glob}/preplay*.safetensors"
  )
  # Sort by modification time (newest first) so we pick the latest checkpoint
  cmd = [
    "ssh",
    SSH_HOST,
    f"ls -1t {search_pattern} 2>/dev/null",
  ]
  print(f"  SSH search: {search_pattern}")
  result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
  paths = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
  if not paths:
    print(f"No checkpoints found for pattern: {wandb_name_pattern}")
    if result.stderr.strip():
      print(f"  stderr: {result.stderr.strip()}")
    sys.exit(1)

  # Deduplicate by wandb_name: keep lowest seed, latest group.
  # Since paths are sorted newest-first by mtime, the first file we see
  # in a given (wandb_name, seed) directory is the most recently modified.
  # Path: .../save_data/{group}/{wandb_name}/seed={N}/preplay_1.safetensors
  by_name = {}
  for p in paths:
    parts = Path(p).parts
    wandb_name = parts[-3]
    seed_name = parts[-2]  # e.g. "seed=3"
    group_name = parts[-4]
    seed_dir = str(Path(p).parent)
    safetensors_fname = parts[-1]
    # Keep lowest seed; if same seed, keep latest group.
    # Within same directory, first occurrence wins (newest by mtime).
    if wandb_name not in by_name or seed_name < by_name[wandb_name][3]:
      by_name[wandb_name] = (seed_dir, safetensors_fname, group_name, seed_name)
    elif seed_name == by_name[wandb_name][3] and seed_dir > by_name[wandb_name][0]:
      by_name[wandb_name] = (seed_dir, safetensors_fname, group_name, seed_name)

  return [
    (name, remote_dir, fname, group, seed)
    for name, (remote_dir, fname, group, seed) in sorted(by_name.items())
  ]


def download_checkpoint(
  wandb_name: str,
  out_dir: Path,
  group_name: str = "",
  seed_name: str = "seed=1",
  remote_dir: str = None,
  safetensors_fname: str = None,
) -> Path:
  """Download checkpoint files, caching per wandb name."""
  ckpt_dir = out_dir / group_name / wandb_name / seed_name / "checkpoint"
  ckpt_dir.mkdir(parents=True, exist_ok=True)

  # Normalize local name to preplay.safetensors for consistency
  safetensor_path = ckpt_dir / "preplay.safetensors"
  config_path = ckpt_dir / "preplay.config"

  if safetensor_path.exists() and config_path.exists():
    print(f"  Using cached checkpoint")
    return ckpt_dir

  if remote_dir is None:
    matches = find_checkpoints_on_server(wandb_name)
    if not matches:
      print(f"No checkpoint found for: {wandb_name}")
      sys.exit(1)
    _, remote_dir, safetensors_fname, _, _ = matches[-1]

  print(f"  Downloading from {remote_dir}...")
  # Download safetensors (remote name may have step suffix, save as preplay.safetensors)
  remote_st = safetensors_fname or "preplay.safetensors"
  subprocess.run(
    ["scp", f"{SSH_HOST}:{remote_dir}/{remote_st}", str(safetensor_path)],
    check=True,
  )
  subprocess.run(
    ["scp", f"{SSH_HOST}:{remote_dir}/preplay.config", str(config_path)],
    check=True,
  )

  return ckpt_dir


def load_config(config_path: Path) -> dict:
  """Load config from a pickle file."""
  with open(config_path, "rb") as f:
    return pickle.load(f)


def load_params(param_path: Path) -> dict:
  """Load model parameters from a safetensors file."""
  flattened_params = load_file(str(param_path))
  return unflatten_dict(flattened_params, sep=",")


def load_checkpoint(ckpt_dir: Path):
  """Load params and config from checkpoint directory."""
  params = load_params(ckpt_dir / "preplay.safetensors")
  config = load_config(ckpt_dir / "preplay.config")
  return params, config


def discover_local_checkpoints(local_dir: Path):
  """Find all checkpoint files in a local directory, sorted by training progress.

  Returns list of (safetensor_path, label, progress_pct).
  E.g.: [(Path(".../preplay_1.safetensors"), "ckpt_1_20pct", 20), ...]
  """
  # Find the config file to determine algorithm name
  config_files = list(local_dir.glob("*.config"))
  if not config_files:
    print(f"No .config file found in {local_dir}")
    sys.exit(1)
  alg_name = config_files[0].stem  # e.g. "preplay"

  safetensor_files = list(local_dir.glob(f"{alg_name}*.safetensors"))
  if not safetensor_files:
    print(f"No {alg_name}*.safetensors files found in {local_dir}")
    sys.exit(1)

  checkpoints = []
  for f in safetensor_files:
    stem = f.stem  # e.g. "preplay_1" or "preplay"
    if "_" in stem and stem.rsplit("_", 1)[1].isdigit():
      idx = int(stem.rsplit("_", 1)[1])
      pct = idx * 20
      label = f"ckpt_{idx}_{pct}pct"
    else:
      # Final checkpoint (no index suffix)
      idx = 999  # sort last
      pct = 100
      label = "ckpt_final_100pct"
    checkpoints.append((f, label, pct, idx))

  checkpoints.sort(key=lambda x: x[3])
  return [(path, label, pct) for path, label, pct, _ in checkpoints]


def setup_env_and_agent(config, wandb_name: str = ""):
  """Reconstruct environment and agent from config."""
  # ─── Environment setup (mirrors jaxmaze_trainer.py:184-240) ──────────
  exp_name = config["rlenv"]["ENV_KWARGS"].get("exp")
  if exp_name is None:
    # exp was popped during training before config was saved; fetch from wandb
    import wandb

    api = wandb.Api()
    project = config.get("PROJECT", "housemaze")
    entity = config.get("entity", "wcarvalho92")
    print(f"  exp not in config, fetching from wandb ({entity}/{project})...")
    runs = api.runs(f"{entity}/{project}", filters={"display_name": wandb_name})
    if runs:
      exp_name = runs[0].config.get("rlenv", {}).get("ENV_KWARGS", {}).get("exp")
    if exp_name is None:
      raise ValueError(f"Cannot determine exp from config or wandb: {wandb_name}")
    print(f"  Found exp={exp_name} from wandb")
  # store it back so the config copy has it
  if "exp" not in config["rlenv"]["ENV_KWARGS"]:
    config["rlenv"]["ENV_KWARGS"]["exp"] = exp_name

  # Need to pop exp like run_single does
  config_copy = {k: v for k, v in config.items()}
  config_copy["rlenv"] = {k: v for k, v in config["rlenv"].items()}
  config_copy["rlenv"]["ENV_KWARGS"] = {
    k: v for k, v in config["rlenv"]["ENV_KWARGS"].items()
  }
  exp = config_copy["rlenv"]["ENV_KWARGS"].pop("exp")

  exp_fn = getattr(jaxmaze_experiments, exp)
  env_params, test_env_params, task_objects, idx2maze = exp_fn(config_copy)

  image_dict = jaxmaze_utils.load_image_dict()
  images = image_dict["images"]
  reshaped = images.reshape(len(images), 8, 4, 8, 4, 3)
  image_dict["images"] = reshaped.mean(axis=(2, 4)).astype(np.uint8)

  task_runner = multitask_env_fast.TaskRunner(task_objects=task_objects)
  env = multitask_env_fast.HouseMaze(
    task_runner=task_runner,
    num_categories=200,
  )

  keys = image_dict["keys"]
  action_names = {action.value: action.name for action in env.action_enum()}

  # ─── Agent setup (mirrors multitask_preplay.py:1845-1903) ────────────
  rnn = vbb.ScannedRNN(
    hidden_dim=config.get("AGENT_RNN_DIM", 256),
    cell_type=config.get("RNN_CELL_TYPE", "OptimizedLSTMCell"),
    unroll_output_state=True,
  )

  observation_encoder = CategoricalJaxmazeObsEncoder(
    num_categories=max(10_000, env.total_categories(env_params)),
    embed_hidden_dim=config["EMBED_HIDDEN_DIM"],
    mlp_hidden_dim=config["MLP_HIDDEN_DIM"],
    num_embed_layers=config["NUM_EMBED_LAYERS"],
    num_mlp_layers=config["NUM_MLP_LAYERS"],
    activation=config["ACTIVATION"],
    norm_type=config.get("NORM_TYPE", "none"),
    include_task=config.get("OBS_INCLUDE_GOAL", True),
  )

  agent = PreplayAgent(
    observation_encoder=observation_encoder,
    rnn=rnn,
    main_q_head=DuellingMLP(
      hidden_dim=config.get("Q_HIDDEN_DIM", 512),
      num_layers=config.get("NUM_PRED_LAYERS", 1),
      activation=config["ACTIVATION"],
      activate_final=False,
      use_bias=config.get("USE_BIAS", False),
      out_dim=env.num_actions(env_params),
    ),
    env=env,
    env_params=env_params,
    get_task=lambda x: x.observation.task_w,
  )

  # Build task vectors
  train_objects = env_params.reset_params.train_objects[0]
  test_objects = env_params.reset_params.test_objects[0]
  train_tasks = jnp.array([env.task_runner.task_vector(o) for o in train_objects])
  test_tasks = jnp.array([env.task_runner.task_vector(o) for o in test_objects])

  return (
    env,
    env_params,
    test_env_params,
    agent,
    image_dict,
    keys,
    action_names,
    task_objects,
    train_objects,
    test_objects,
    train_tasks,
    test_tasks,
  )


def run_episode(
  env, env_params, agent, params, rng, max_steps=100, epsilon=0.0, start_pos=None
):
  """Run an evaluation episode. Returns list of (state, q_vals, action).

  Args:
    epsilon: Probability of random action. 0.0 = fully greedy.
    start_pos: Optional custom start position (row, col). If provided, overrides
      the agent's starting position after env.reset().
  """
  rng, reset_rng = jax.random.split(rng)
  timestep = env.reset(reset_rng, env_params)
  n_actions = len(env.action_enum()) - 1  # exclude done

  if start_pos is not None:
    new_state = timestep.state.replace(agent_pos=jnp.array(start_pos))
    reset_action = jnp.array(env.num_actions(env_params))
    new_obs = env.make_observation(new_state, prev_action=reset_action)
    timestep = timestep.replace(state=new_state, observation=new_obs)

  # Initialize RNN state
  batch_dims = (1,)
  rng, init_rng = jax.random.split(rng)
  rnn_state = agent.apply(
    params, batch_dims=batch_dims, rng=init_rng, method=agent.initialize_carry
  )

  steps = []
  for t in range(max_steps):
    # Add batch dim for agent
    ts_batch = jax.tree_util.tree_map(lambda x: x[None], timestep)
    rng, step_rng = jax.random.split(rng)
    preds, rnn_state = agent.apply(params, rnn_state, ts_batch, step_rng)

    q_vals = preds.q_vals[0]  # remove batch dim  [A]

    # Epsilon-greedy action selection
    rng, eps_rng = jax.random.split(rng)
    if epsilon > 0 and float(jax.random.uniform(eps_rng)) < epsilon:
      rng, act_rng = jax.random.split(rng)
      action = jax.random.randint(act_rng, shape=(), minval=0, maxval=n_actions)
    else:
      action = jnp.argmax(q_vals)

    steps.append(
      {
        "state": timestep.state,
        "q_vals": np.array(q_vals),
        "action": int(action),
        "reward": float(timestep.reward),
      }
    )

    # Check if episode ended
    if timestep.last():
      break

    # Step environment
    rng, env_rng = jax.random.split(rng)
    timestep = env.step(env_rng, timestep, action, env_params)

  return steps


def run_episode_optimal(
  env, env_params, agent, params, rng, max_steps=100, start_pos=None
):
  """Run episode using BFS-optimal actions, recording model Q-values at each step."""
  rng, reset_rng = jax.random.split(rng)
  timestep = env.reset(reset_rng, env_params)

  if start_pos is not None:
    new_state = timestep.state.replace(agent_pos=jnp.array(start_pos))
    reset_action = jnp.array(env.num_actions(env_params))
    new_obs = env.make_observation(new_state, prev_action=reset_action)
    timestep = timestep.replace(state=new_state, observation=new_obs)

  # Compute BFS path and convert to actions
  state = timestep.state
  path = find_optimal_path(
    np.array(state.grid),
    np.array(state.agent_pos),
    int(state.task_object),
    pass_through_objects=True,
  )
  optimal_actions = actions_from_path(path)
  # Remove trailing "done" action — it's out of bounds for Q-values
  n_valid = len(env.action_enum()) - 1
  optimal_actions = [a for a in optimal_actions if int(a) < n_valid]

  # Initialize RNN state
  batch_dims = (1,)
  rng, init_rng = jax.random.split(rng)
  rnn_state = agent.apply(
    params, batch_dims=batch_dims, rng=init_rng, method=agent.initialize_carry
  )

  steps = []
  for t in range(min(max_steps, len(optimal_actions))):
    ts_batch = jax.tree_util.tree_map(lambda x: x[None], timestep)
    rng, step_rng = jax.random.split(rng)
    preds, rnn_state = agent.apply(params, rnn_state, ts_batch, step_rng)
    q_vals = preds.q_vals[0]

    action = int(optimal_actions[t])

    steps.append(
      {
        "state": timestep.state,
        "q_vals": np.array(q_vals),
        "action": action,
        "reward": float(timestep.reward),
      }
    )

    if timestep.last():
      break

    rng, env_rng = jax.random.split(rng)
    timestep = env.step(env_rng, timestep, jnp.array(action), env_params)

  # Record terminal timestep (carries the goal reward from the last action)
  if timestep.last():
    ts_batch = jax.tree_util.tree_map(lambda x: x[None], timestep)
    rng, step_rng = jax.random.split(rng)
    preds, rnn_state = agent.apply(params, rnn_state, ts_batch, step_rng)
    q_vals = preds.q_vals[0]
    steps.append(
      {
        "state": timestep.state,
        "q_vals": np.array(q_vals),
        "action": 0,  # dummy — episode is over
        "reward": float(timestep.reward),
      }
    )

  return steps


def fig_to_array(fig):
  """Convert matplotlib figure to numpy RGB array."""
  import io

  buf = io.BytesIO()
  fig.savefig(buf, format="png", bbox_inches="tight", dpi=fig.dpi)
  buf.seek(0)
  frame = imageio.v2.imread(buf)[:, :, :3]
  plt.close(fig)
  return frame


def render_env_image(steps, current_step, image_dict, title=""):
  """Render the environment image with trajectory arrows. Fixed 8x6 inches."""
  n_steps_so_far = current_step + 1
  state = steps[current_step]["state"]

  env_img = renderer.create_image_from_grid(
    state.grid, state.agent_pos, state.agent_dir, image_dict
  )
  maze_height, maze_width, _ = state.grid.shape

  fig, ax = plt.subplots(1, 1, figsize=(8, 6))
  fig.set_dpi(120)

  positions = [steps[i]["state"].agent_pos for i in range(n_steps_so_far)]
  actions = [steps[i]["action"] for i in range(n_steps_so_far - 1)]

  if len(actions) > 0:
    renderer.place_arrows_on_image(
      env_img,
      positions[:-1],
      actions,
      maze_height,
      maze_width,
      arrow_scale=5,
      ax=ax,
    )
  else:
    ax.imshow(env_img)
  ax.set_title(title, fontsize=18)
  ax.axis("off")
  return fig_to_array(fig)


def render_heatmap(steps, current_step, action_names):
  """Render the Q-value heatmap. Width scales with timesteps, height is fixed."""
  n_steps_so_far = current_step + 1

  q_values = np.array([steps[i]["q_vals"] for i in range(n_steps_so_far)])  # [T, A]
  taken_actions = [steps[i]["action"] for i in range(n_steps_so_far)]

  q_T = q_values.T  # [A, T]
  n_actions = q_T.shape[0]

  # Fixed width per column so numbers are always readable
  col_width = 0.7
  fig_width = max(5, n_steps_so_far * col_width + 2.5)
  fig_height = 4

  fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
  fig.set_dpi(120)

  im = ax.imshow(
    q_T,
    aspect="equal",
    cmap="coolwarm",
    interpolation="nearest",
  )

  for t in range(n_steps_so_far):
    for a in range(n_actions):
      ax.text(
        t,
        a,
        f"{q_T[a, t]:.2f}",
        ha="center",
        va="center",
        fontsize=9,
        color="black",
      )
    # Black rectangle: argmax action
    best_a = int(np.argmax(q_values[t]))
    ax.add_patch(
      Rectangle(
        (t - 0.45, best_a - 0.45),
        0.9,
        0.9,
        linewidth=2,
        edgecolor="black",
        facecolor="none",
      )
    )
    # Cyan rectangle: taken action
    taken_a = taken_actions[t]
    ax.add_patch(
      Rectangle(
        (t - 0.35, taken_a - 0.35),
        0.7,
        0.7,
        linewidth=3,
        edgecolor="cyan",
        facecolor="none",
      )
    )

  # Highlight current step column
  ax.add_patch(
    Rectangle(
      (current_step - 0.5, -0.5),
      1.0,
      n_actions,
      linewidth=3,
      edgecolor="yellow",
      facecolor="none",
      linestyle="--",
    )
  )

  ax.set_title("Q-values (black=argmax, cyan=taken, yellow=current)", fontsize=14)
  ax.set_yticks(range(n_actions))
  ax.set_yticklabels(
    [action_names.get(i, str(i)) for i in range(n_actions)], fontsize=12
  )
  ax.set_xticks(range(n_steps_so_far))
  ax.tick_params(labelsize=10)
  fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
  return fig_to_array(fig)


def render_qvalue_lineplot(steps, current_step, action_names, gamma=0.99):
  """Render a line plot of Q-values over time. Shows Q-taken, Q-max, and Return G."""
  n_steps_so_far = current_step + 1
  total = len(steps)

  q_values = np.array([steps[i]["q_vals"] for i in range(n_steps_so_far)])  # [T, A]
  taken_actions = [steps[i]["action"] for i in range(n_steps_so_far)]
  rewards = [steps[i]["reward"] for i in range(n_steps_so_far)]

  # Compute returns backwards over ALL steps
  all_rewards = np.array([steps[i]["reward"] for i in range(total)])
  returns = np.zeros(total)
  for t in reversed(range(total)):
    r_next = all_rewards[t + 1] if t + 1 < total else 0
    returns[t] = r_next + gamma * (returns[t + 1] if t + 1 < total else 0)

  q_taken = np.array([q_values[t, taken_actions[t]] for t in range(n_steps_so_far)])
  q_max = q_values.max(axis=-1)

  fig, ax = plt.subplots(1, 1, figsize=(8, 4))
  fig.set_dpi(120)

  ax.plot(q_taken, label="Q-taken", color="tab:orange", linewidth=2)
  ax.plot(q_max, label="Q-max", color="tab:green", linewidth=2)
  ax.plot(rewards, label="Reward", color="tab:blue", linewidth=2)
  ax.plot(
    returns[:n_steps_so_far],
    label="Return G",
    color="tab:cyan",
    linewidth=2,
    linestyle="--",
  )

  # Plot per-action Q-values as thin lines
  n_actions = q_values.shape[1]
  colors = ["tab:red", "tab:purple", "tab:brown", "tab:gray", "tab:pink"]
  for a in range(n_actions):
    ax.plot(
      q_values[:, a],
      label=action_names.get(a, str(a)),
      color=colors[a % len(colors)],
      linewidth=0.8,
      alpha=0.5,
      linestyle="--",
    )

  # Mark current step
  ax.axvline(current_step, color="yellow", linewidth=2, linestyle="--")

  ax.set_title("Q-values over time", fontsize=14)
  ax.legend(fontsize=9, loc="upper left", ncol=2)
  ax.grid(True, alpha=0.3)
  ax.set_xlabel("Step")
  ax.set_xlim(-0.5, n_steps_so_far - 0.5)
  return fig_to_array(fig)


def render_td_error_plot(steps, current_step, gamma=0.99):
  """Render TD-error plot: Q(s_t, a_t) - G_t where G_t is the discounted return."""
  n_steps_so_far = current_step + 1
  total = len(steps)

  # Compute returns backwards over ALL steps
  rewards = np.array([steps[i]["reward"] for i in range(total)])
  returns = np.zeros(total)
  for t in reversed(range(total)):
    r_next = rewards[t + 1] if t + 1 < total else 0
    returns[t] = r_next + gamma * (returns[t + 1] if t + 1 < total else 0)

  # Q-values of taken actions
  n_plot = n_steps_so_far
  q_taken = np.array([steps[i]["q_vals"][steps[i]["action"]] for i in range(n_plot)])
  # G_T = 0 at the terminal step (already guaranteed by the backward pass above)
  td_errors = q_taken - returns[:n_plot]

  fig, ax = plt.subplots(1, 1, figsize=(8, 4))
  fig.set_dpi(120)

  ax.plot(td_errors, label="TD error (Q - G)", color="tab:red", linewidth=2)
  ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
  ax.axvline(current_step, color="yellow", linewidth=2, linestyle="--")

  ax.set_title("TD Error: Q(s,a) - Return", fontsize=14)
  ax.legend(fontsize=9, loc="upper left")
  ax.grid(True, alpha=0.3)
  ax.set_xlabel("Step")
  ax.set_xlim(-0.5, n_steps_so_far - 0.5)
  return fig_to_array(fig)


def render_frame(steps, current_step, image_dict, action_names, title="", gamma=0.99):
  """Render a single video frame:
  Row 0: env image (left) + Q-value line plot (center) + TD-error plot (right)
  Row 1: Q-value heatmap
  """
  env_img = render_env_image(steps, current_step, image_dict, title=title)
  lineplot_img = render_qvalue_lineplot(steps, current_step, action_names, gamma=gamma)
  td_img = render_td_error_plot(steps, current_step, gamma=gamma)
  heatmap_img = render_heatmap(steps, current_step, action_names)

  # Row 0: env + lineplot + td-error side by side, matched height
  target_h = max(env_img.shape[0], lineplot_img.shape[0], td_img.shape[0])

  def pad_height(img, h):
    if img.shape[0] >= h:
      return img
    padded = np.ones((h, img.shape[1], 3), dtype=np.uint8) * 255
    padded[: img.shape[0]] = img
    return padded

  env_img = pad_height(env_img, target_h)
  lineplot_img = pad_height(lineplot_img, target_h)
  td_img = pad_height(td_img, target_h)
  top_row = np.concatenate([env_img, lineplot_img, td_img], axis=1)

  # Stack top_row + heatmap vertically
  max_w = max(top_row.shape[1], heatmap_img.shape[1])
  total_h = top_row.shape[0] + heatmap_img.shape[0]
  frame = np.ones((total_h, max_w, 3), dtype=np.uint8) * 255

  # Center top row
  x_off = (max_w - top_row.shape[1]) // 2
  frame[: top_row.shape[0], x_off : x_off + top_row.shape[1]] = top_row

  # Left-align heatmap
  y_off = top_row.shape[0]
  frame[y_off : y_off + heatmap_img.shape[0], : heatmap_img.shape[1]] = heatmap_img

  return frame


def make_video(
  steps, image_dict, action_names, output_path, title="", fps=2, gamma=0.99
):
  """Create video from episode steps."""
  print(f"  Rendering {len(steps)} frames...")
  frames = []
  # Need consistent frame size - use the last frame's size for all
  # First pass: render all frames and find max size
  rendered = []
  for t in range(len(steps)):
    frame = render_frame(steps, t, image_dict, action_names, title=title, gamma=gamma)
    rendered.append(frame)

  # Pad all frames to the same size (last frame is widest due to growing heatmap)
  max_h = max(f.shape[0] for f in rendered)
  max_w = max(f.shape[1] for f in rendered)
  for frame in rendered:
    h, w = frame.shape[:2]
    padded = np.ones((max_h, max_w, 3), dtype=np.uint8) * 255
    padded[:h, :w] = frame
    frames.append(padded)

  print(f"  Writing video to {output_path}")
  imageio.mimwrite(str(output_path), frames, fps=fps, codec="libx264")
  print(f"  Done: {output_path}")
  return frames[-1]  # return last frame for verification


def render_last_frame(
  steps, image_dict, action_names, output_path, title="", gamma=0.99
):
  """Render only the final frame and save as PNG. Much faster than make_video."""
  frame = render_frame(
    steps, len(steps) - 1, image_dict, action_names, title=title, gamma=gamma
  )
  imageio.imwrite(str(output_path), frame)
  print(f"  Saved: {output_path}")
  return frame


def process_checkpoint(
  args,
  params,
  env,
  env_params,
  test_env_params,
  agent,
  image_dict,
  keys,
  action_names,
  video_dir,
  gamma,
  progress_label="",
):
  """Run episodes and render outputs for a single set of model params.

  Returns list of generated file paths.
  """
  # Find the level with rotation=(0,0) and pin to it
  rotations = jnp.asarray(test_env_params.reset_params.rotation)  # [num_levels, 2]
  level_idx = None
  for i in range(rotations.shape[0]):
    if int(rotations[i, 0]) == 0 and int(rotations[i, 1]) == 0:
      level_idx = i
      break
  if level_idx is None:
    print("  ERROR: No level with rotation=(0,0) found")
    return []

  # Create single-level env_params so random level selection is deterministic
  single_reset = jax.tree_map(
    lambda x: x[level_idx : level_idx + 1], test_env_params.reset_params
  )
  base_params = test_env_params.replace(reset_params=single_reset)

  # Get train/test objects for this specific level
  train_objects = test_env_params.reset_params.train_objects[level_idx]
  test_objects = test_env_params.reset_params.test_objects[level_idx]

  print(f"  Train objects: {[keys[int(o)] for o in train_objects]}")
  print(f"  Test objects: {[keys[int(o)] for o in test_objects]}")

  video_dir.mkdir(parents=True, exist_ok=True)

  rng = jax.random.PRNGKey(42)
  num_groups = len(train_objects)
  generated_files = []

  for group_idx in range(num_groups):
    train_goal = keys[int(train_objects[group_idx])]
    test_goal = keys[int(test_objects[group_idx])]

    train_params = base_params.replace(
      force_room=jnp.array(True),
      default_room=jnp.array(group_idx),
      p_test_sample_train=1.0,
    )
    test_params = base_params.replace(
      force_room=jnp.array(True),
      default_room=jnp.array(group_idx),
      p_test_sample_train=0.0,
    )

    for epsilon in args.epsilon:
      eps_label = f"eps{epsilon:.2f}".replace(".", "")
      prog_suffix = f" [{progress_label}]" if progress_label else ""

      def generate_outputs(ep_steps, base_name, title):
        """Generate last_frame PNG (and optionally video) for an episode."""
        png_path = video_dir / f"{base_name}.png"
        render_last_frame(
          ep_steps, image_dict, action_names, png_path, title=title, gamma=gamma
        )
        generated_files.append(png_path)
        if args.video:
          mp4_path = video_dir / f"{base_name}.mp4"
          make_video(
            ep_steps,
            image_dict,
            action_names,
            mp4_path,
            title=title,
            fps=args.fps,
            gamma=gamma,
          )
          generated_files.append(mp4_path)

      def run_and_render(ep_params, goal_name, task_type, start_pos=None):
        """Run model + optimal episodes, generate outputs for both."""
        nonlocal rng
        suffix = f"_{eps_label}"
        pos_label = ""
        if start_pos is not None:
          suffix += "_halfway"
          pos_label = " [halfway]"

        # Model actions
        rng, ep_rng = jax.random.split(rng)
        model_steps = run_episode(
          env,
          ep_params,
          agent,
          params,
          ep_rng,
          max_steps=args.max_steps,
          epsilon=epsilon,
          start_pos=start_pos,
        )
        actual = keys[int(model_steps[0]["state"].task_object)]
        print(f"    Model: task={actual}, steps={len(model_steps)}")
        generate_outputs(
          model_steps,
          f"model_{task_type}_{goal_name}{suffix}",
          f"Model {task_type}: {goal_name} (eps={epsilon}){pos_label}{prog_suffix}",
        )

        # Optimal actions
        if args.optimal:
          rng, ep_rng = jax.random.split(rng)
          opt_steps = run_episode_optimal(
            env,
            ep_params,
            agent,
            params,
            ep_rng,
            max_steps=args.max_steps,
            start_pos=start_pos,
          )
          print(f"    Optimal: steps={len(opt_steps)}")
          generate_outputs(
            opt_steps,
            f"optimal_{task_type}_{goal_name}{suffix}",
            f"Optimal {task_type}: {goal_name}{pos_label}{prog_suffix}",
          )

        return model_steps

      # Skip if outputs already exist
      check_png = video_dir / f"model_train_{train_goal}_{eps_label}.png"
      check_optimal = video_dir / f"optimal_train_{train_goal}_{eps_label}.png"
      all_exist = check_png.exists() and (not args.optimal or check_optimal.exists())
      if all_exist and not args.force:
        print(f"\n  Skipping group {group_idx} eps={epsilon} — outputs exist")
        rng, _ = jax.random.split(rng)
        rng, _ = jax.random.split(rng)
        continue

      # Train task — original start
      print(f"\n  === Group {group_idx} Train: {train_goal} (eps={epsilon}) ===")
      train_steps = run_and_render(train_params, train_goal, "train")

      # Train task — halfway start
      if args.halfway:
        state0 = train_steps[0]["state"]
        path = find_optimal_path(
          np.array(state0.grid),
          np.array(state0.agent_pos),
          int(state0.task_object),
          pass_through_objects=True,
        )
        if path is not None and len(path) > 2:
          midpoint = path[len(path) // 2]
          print(f"    Halfway: {np.array(state0.agent_pos)} -> {midpoint}")
          run_and_render(train_params, train_goal, "train", start_pos=midpoint)

      # Test task — original start
      print(f"\n  === Group {group_idx} Test: {test_goal} (eps={epsilon}) ===")
      test_steps = run_and_render(test_params, test_goal, "test")

      # Test task — halfway start
      if args.halfway:
        state0 = test_steps[0]["state"]
        path = find_optimal_path(
          np.array(state0.grid),
          np.array(state0.agent_pos),
          int(state0.task_object),
          pass_through_objects=True,
        )
        if path is not None and len(path) > 2:
          midpoint = path[len(path) // 2]
          print(f"    Halfway: {np.array(state0.agent_pos)} -> {midpoint}")
          run_and_render(test_params, test_goal, "test", start_pos=midpoint)

  return generated_files


def main_local(args):
  """Process all checkpoints in a local directory."""
  local_dir = args.local_dir.resolve()

  # Parse metadata from path: .../save_data/{group}/{wandb_name}/{seed}/
  seed_name = local_dir.name
  wandb_name = local_dir.parent.name
  group_name = local_dir.parent.parent.name

  print(f"Local mode: {group_name}/{wandb_name}/{seed_name}")

  # Discover all checkpoints
  checkpoints = discover_local_checkpoints(local_dir)
  print(f"Found {len(checkpoints)} checkpoint(s):")
  for _, label, pct in checkpoints:
    print(f"  {label} ({pct}%)")

  # Load config once
  config_files = list(local_dir.glob("*.config"))
  config = load_config(config_files[0])
  gamma = config.get("GAMMA", 0.99)

  # Setup env + agent once
  wn = wandb_name if args.wandb_name is None else args.wandb_name
  print("  Setting up environment and agent...")
  (
    env,
    env_params,
    test_env_params,
    agent,
    image_dict,
    keys,
    action_names,
    task_objects,
    _train_objects,
    _test_objects,
    train_tasks,
    test_tasks,
  ) = setup_env_and_agent(config, wandb_name=wn)

  # Process each checkpoint
  all_generated = []
  for ckpt_path, label, pct in checkpoints:
    print(f"\n{'#' * 70}")
    print(f"# Checkpoint: {label} ({pct}% trained)")
    print(f"{'#' * 70}")

    params = load_params(ckpt_path)
    video_dir = local_dir / "videos" / label

    generated = process_checkpoint(
      args,
      params,
      env,
      env_params,
      test_env_params,
      agent,
      image_dict,
      keys,
      action_names,
      video_dir,
      gamma,
      progress_label=f"{pct}% trained",
    )
    all_generated.extend(generated)
    print(f"  Outputs saved to: {video_dir}")

  # Write video_links.txt
  links_path = local_dir / "videos" / "video_links.txt"
  links_path.parent.mkdir(parents=True, exist_ok=True)
  with open(links_path, "w") as f:
    for p in all_generated:
      f.write(f"{p}\n")
  print(f"\nVideo links written to: {links_path}")

  print(f"\nDone. Processed {len(checkpoints)} checkpoint(s).")
  return 0


def main():
  parser = argparse.ArgumentParser(description="Generate preplay evaluation videos")
  parser.add_argument(
    "wandb_name", nargs="?", default=None, help="Wandb run name (the directory name)"
  )
  parser.add_argument(
    "--local-dir",
    type=Path,
    default=None,
    help="Local checkpoint directory (bypasses SSH/SCP). "
    "Processes all checkpoints in the directory.",
  )
  parser.add_argument(
    "--out",
    type=Path,
    default=Path("~/Desktop/videos/preplay").expanduser(),
    help="Output directory",
  )
  parser.add_argument(
    "--seed", type=int, default=None, help="Checkpoint seed (default: search all seeds)"
  )
  parser.add_argument("--max-steps", type=int, default=90, help="Max episode steps")
  parser.add_argument("--fps", type=int, default=2, help="Video FPS")
  parser.add_argument(
    "--epsilon",
    type=float,
    nargs="+",
    default=[0.0],
    help="Epsilon values for evaluation (default: 0.0)",
  )
  parser.add_argument(
    "--force",
    action="store_true",
    help="Regenerate outputs even if they already exist",
  )
  parser.add_argument(
    "--halfway",
    action="store_true",
    help="Also generate episodes starting from halfway point",
  )
  parser.add_argument(
    "--video",
    action="store_true",
    help="Also generate mp4 videos (default: only last-frame PNGs)",
  )
  parser.add_argument(
    "--optimal",
    action="store_true",
    help="Also generate optimal (BFS) episodes alongside model episodes",
  )
  args = parser.parse_args()

  if args.local_dir is not None:
    return main_local(args)

  if args.wandb_name is None:
    parser.error("wandb_name is required when --local-dir is not used")

  args.out = args.out.expanduser()
  args.out.mkdir(parents=True, exist_ok=True)

  # ─── Step 1: Find all matching checkpoints ──────────────────────────
  print(f"Searching for: {args.wandb_name}")
  matches = find_checkpoints_on_server(args.wandb_name, seed=args.seed)
  print(f"Found {len(matches)} matching run(s):")
  for name, remote_dir, _fname, group, seed_name in matches:
    print(f"  {name} (group: {group}, {seed_name})")

  # ─── Step 2: Process each matching run ──────────────────────────────
  for run_idx, (
    wandb_name,
    remote_dir,
    safetensors_fname,
    group_name,
    seed_name,
  ) in enumerate(matches):
    print(f"\n{'#' * 70}")
    print(f"# Run {run_idx + 1}/{len(matches)}: {wandb_name}")
    print(f"{'#' * 70}")

    video_dir = args.out / group_name / wandb_name / seed_name

    # Download checkpoint
    ckpt_dir = download_checkpoint(
      wandb_name,
      args.out,
      group_name=group_name,
      seed_name=seed_name,
      remote_dir=remote_dir,
      safetensors_fname=safetensors_fname,
    )

    # Load checkpoint
    print("  Loading checkpoint...")
    params, config = load_checkpoint(ckpt_dir)
    gamma = config.get("GAMMA", 0.99)

    # Setup env + agent
    print("  Setting up environment and agent...")
    (
      env,
      env_params,
      test_env_params,
      agent,
      image_dict,
      keys,
      action_names,
      task_objects,
      _train_objects,
      _test_objects,
      train_tasks,
      test_tasks,
    ) = setup_env_and_agent(config, wandb_name=wandb_name)

    generated_files = process_checkpoint(
      args,
      params,
      env,
      env_params,
      test_env_params,
      agent,
      image_dict,
      keys,
      action_names,
      video_dir,
      gamma,
    )

    print(f"  Outputs saved to: {video_dir}")

  print(f"\nDone. Processed {len(matches)} run(s).")


if __name__ == "__main__":
  main()
