"""Launch revision experiments on SLURM."""

import os
import subprocess
import sys

import yaml

RL_RESULTS_DIR = "/n/holylfs06/LABS/kempner_fellow_wcarvalho/jax_rl_results"
TRAINER = "craftax_multigoal_trainer.py"
STATUS_FILE = "revision_experiments_craftax_multigoal_status.yaml"

# (search_name, partition)
EXPERIMENTS = [
  # rerunning baselines
  ("her-final", "kempner_h100"),
  ("ql-final", "kempner_h100"),
  ("dyna-final", "kempner_h100"),
  ("usfa-final", "kempner_h100"),
  ("preplay-final", "kempner_h100"),
]

# partitions that require an explicit account override
PARTITION_ACCOUNTS = {
  "gpu_h200": "wcarvalho_lab",
}


def load_status():
  if os.path.exists(STATUS_FILE):
    with open(STATUS_FILE) as f:
      return yaml.safe_load(f) or {}
  return {}


def save_status(status):
  with open(STATUS_FILE, "w") as f:
    yaml.dump(status, f, default_flow_style=False)


def build_cmd(search, partition, debug=False):
  if debug:
    cmd = [
      sys.executable,
      "-m",
      "ipdb",
      "-c",
      "continue",
      TRAINER,
      "app.parallel=none",
      "app.debug=True",
      "app.wandb=False",
      f"app.search={search}",
    ]
  else:
    cmd = [
      sys.executable,
      TRAINER,
      "app.parallel=slurm",
      f"app.partition={partition}",
      f"app.search={search}",
    ]
    if partition in PARTITION_ACCOUNTS:
      cmd.append(f"app.account={PARTITION_ACCOUNTS[partition]}")
  return cmd


def build_env(debug=False):
  env = {**os.environ}
  if debug:
    env.update(
      {
        "HYDRA_FULL_ERROR": "1",
        "JAX_TRACEBACK_FILTERING": "off",
        "JAX_DEBUG_NANS": "True",
        "JAX_DISABLE_JIT": "1",
      }
    )
  else:
    env.update(
      {
        "RL_RESULTS_DIR": RL_RESULTS_DIR,
        "JAX_PLATFORMS": "cpu",
      }
    )
  return env


def main():
  dry_run = "--dry-run" in sys.argv
  debug = "--debug" in sys.argv
  skip_passed = "--no-skip" not in sys.argv

  status = load_status()

  for search, partition in EXPERIMENTS:
    if skip_passed and status.get(search) == "ok":
      print(f"\nSkipping {search} (already passed, edit {STATUS_FILE} to rerun)")
      continue

    cmd = build_cmd(search, partition, debug=debug)
    env = build_env(debug=debug)

    label = "[DRY RUN] " if dry_run else ("[DEBUG] " if debug else "")
    print(f"\n{label}Running: {' '.join(cmd)}")

    if dry_run:
      continue

    result = subprocess.run(cmd, env=env)
    if result.returncode == 0:
      status[search] = "ok"
      save_status(status)
      print(f"  {search}: OK")
    else:
      status[search] = f"failed (exit {result.returncode})"
      save_status(status)
      print(f"  WARNING: {search} exited with code {result.returncode}")
      if debug:
        print("  Stopping on first failure in debug mode.")
        break


if __name__ == "__main__":
  main()
