# Multitask Preplay (AI Simulations)


## Install

### Using uv (recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# For development dependencies (ruff, etc.)
uv sync --all-extras
```

### Running scripts

```bash
# Run any script with uv
uv run python craftax_trainer.py app.parallel=none app.debug=True

# Or activate the environment first
source .venv/bin/activate
python craftax_trainer.py app.parallel=none app.debug=True
```

### JAX with GPU (optional)

If you need GPU support, install the appropriate JAX version after `uv sync`:

```bash
# For CUDA 12
uv pip install --upgrade "jax[cuda12]"

# For CUDA 11
uv pip install --upgrade "jax[cuda11_pip]"
```
