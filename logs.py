#!/usr/bin/env python3
"""SLURM log viewer for job .out and .err files."""

import argparse
from pathlib import Path


BASE_FOLDER = "/n/holylfs06/LABS/kempner_fellow_wcarvalho/jax_rl_results/"

TRAINER_MAP = {
  "jaxmaze": "jaxmaze_trainer",
  "craftax": "craftax_trainer",
  "craftax_multigoal": "craftax_multigoal_trainer",
}


def get_job_id(filename: str) -> int:
  """Extract job ID from filename like 'id=12345.out'."""
  try:
    # Extract the number between 'id=' and the extension
    id_part = filename.split("id=")[1].split(".")[0]
    return int(id_part)
  except (IndexError, ValueError):
    return 0


def print_file_contents(filepath: Path, full: bool, tail_lines: int = 50):
  """Print file contents with header."""
  print("\n" + "=" * 80)
  print(f"FILE: {filepath}")
  print("=" * 80)

  if not filepath.exists():
    print("[File does not exist]")
    return

  try:
    with open(filepath, "r") as f:
      if full:
        print(f.read())
      else:
        lines = f.readlines()
        if len(lines) <= tail_lines:
          print("".join(lines), end="")
        else:
          print(f"[... showing last {tail_lines} of {len(lines)} lines ...]\n")
          print("".join(lines[-tail_lines:]), end="")
  except Exception as e:
    print(f"[Error reading file: {e}]")


def main():
  parser = argparse.ArgumentParser(description="View SLURM job log files")
  parser.add_argument(
    "--trainer",
    required=True,
    choices=list(TRAINER_MAP.keys()),
    help="Trainer name (jaxmaze, craftax, craftax_multigoal)",
  )
  parser.add_argument(
    "--search",
    required=True,
    help="Search name (e.g., her, ql, usfa)",
  )
  parser.add_argument(
    "--out",
    action="store_true",
    help="Show .out files",
  )
  parser.add_argument(
    "--err",
    action="store_true",
    help="Show .err files",
  )
  parser.add_argument(
    "--all",
    action="store_true",
    help="Show all run folders and all job files (default: only latest)",
  )
  parser.add_argument(
    "--full",
    action="store_true",
    help="Show full file contents (default: tail last 50 lines)",
  )

  args = parser.parse_args()

  # Must specify at least one of --out or --err
  if not args.out and not args.err:
    parser.error("Must specify at least one of --out or --err")

  # Build path to sbatch directory
  trainer_name = TRAINER_MAP[args.trainer]
  sbatch_dir = Path(BASE_FOLDER) / trainer_name / args.search / "sbatch"

  if not sbatch_dir.exists():
    print(f"Error: sbatch directory not found: {sbatch_dir}")
    return 1

  # Find run folders (runs-YYYY.MM.DD-HH.MM)
  run_folders = sorted(
    [d for d in sbatch_dir.iterdir() if d.is_dir() and d.name.startswith("runs-")],
    key=lambda x: x.name,
    reverse=True,  # Latest first
  )

  if not run_folders:
    print(f"No run folders found in {sbatch_dir}")
    return 1

  # Select folders to process
  folders_to_process = run_folders if args.all else [run_folders[0]]

  for run_folder in folders_to_process:
    print("\n" + "#" * 80)
    print(f"# RUN FOLDER: {run_folder.name}")
    print("#" * 80)

    # Collect files to show
    extensions = []
    if args.out:
      extensions.append(".out")
    if args.err:
      extensions.append(".err")

    # Find all job files
    job_files = []
    for ext in extensions:
      job_files.extend(run_folder.glob(f"id=*{ext}"))

    # Sort by job ID (highest/most recent first)
    job_files = sorted(job_files, key=lambda x: get_job_id(x.name), reverse=True)

    if not job_files:
      print(f"No log files found in {run_folder}")
      continue

    # Select files to process
    if args.all:
      files_to_process = job_files
    else:
      # Get latest job ID and show both .out and .err for it if requested
      latest_job_id = get_job_id(job_files[0].name)
      files_to_process = [f for f in job_files if get_job_id(f.name) == latest_job_id]

    for filepath in files_to_process:
      print_file_contents(filepath, args.full)

  return 0


if __name__ == "__main__":
  exit(main())
