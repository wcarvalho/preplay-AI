RL_RESULTS_DIR=/n/holylfs06/LABS/kempner_fellow_wcarvalho/jax_rl_results \
JAX_PLATFORMS=cpu python jaxmaze_trainer.py \
  app.parallel=slurm \
  app.search=her2

RL_RESULTS_DIR=/n/holylfs06/LABS/kempner_fellow_wcarvalho/jax_rl_results \
JAX_PLATFORMS=cpu python jaxmaze_trainer.py \
  app.partition=kempner_h100 \
  app.parallel=slurm \
  app.search=preplay2