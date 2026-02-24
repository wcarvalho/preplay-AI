RL_RESULTS_DIR=/n/holylfs06/LABS/kempner_fellow_wcarvalho/jax_rl_results \
JAX_PLATFORMS=cpu python jaxmaze_trainer.py \
  app.parallel=slurm \
  app.search=ql-final

RL_RESULTS_DIR=/n/holylfs06/LABS/kempner_fellow_wcarvalho/jax_rl_results \
JAX_PLATFORMS=cpu python jaxmaze_trainer.py \
  app.parallel=slurm \
  app.search=dyna-final

RL_RESULTS_DIR=/n/holylfs06/LABS/kempner_fellow_wcarvalho/jax_rl_results \
JAX_PLATFORMS=cpu python jaxmaze_trainer.py \
  app.parallel=slurm \
  app.search=usfa-final

RL_RESULTS_DIR=/n/holylfs06/LABS/kempner_fellow_wcarvalho/jax_rl_results \
JAX_PLATFORMS=cpu python jaxmaze_trainer.py \
  app.parallel=slurm \
  app.search=her-final