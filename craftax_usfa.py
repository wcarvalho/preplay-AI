from jaxneurorl.agents.usfa import *
from networks import CraftaxObsEncoder

@struct.dataclass
class UsfaR2D2LossFn(vbb.RecurrentLossFn):
    extract_cumulants: Callable = lambda data: data.timestep.observation.state_features
    extract_task: Callable = lambda data: data.timestep.observation.task_w

    def error(self, data, online_preds, online_state, target_preds, target_state, steps, **kwargs):
        # Prepare Data
        online_sf = online_preds.sf  # [T+1, B, N, A, C]
        online_w = data.timestep.observation.task_w  # [T+1, B, C]
        target_sf = target_preds.sf  # [T+1, B, N, A, C]

        # NOTE: main change was to concat these
        achievable = data.timestep.observation.achievable
        achievements = data.timestep.observation.achievements
        cumulants = jnp.concatenate((achievable, achievements), axis=-1)
        cumulants = cumulants.astype(online_sf.dtype)
        cumulants = cumulants - self.step_cost

        # Get selector actions from online Q-values for double Q-learning
        dot = lambda x, y: (x * y).sum(axis=-1)
        vdot = jax.vmap(jax.vmap(dot, (2, None), 2), (2, None), 2)
        online_q = vdot(online_sf, online_w)  # [T+1, B, N, A]
        selector_actions = jnp.argmax(online_q, axis=-1)  # [T+1, B, N]

        # Preprocess discounts & rewards
        def float(x): return x.astype(jnp.float32)
        discounts = float(data.discount) * self.discount
        lambda_ = jnp.ones_like(data.discount) * self.lambda_
        is_last = float(data.is_last)

        # Prepare loss (via vmaps)
        # vmap over batch dimension (B), return B in dim=1
        td_error_fn = jax.vmap(
            losses.q_learning_lambda_td,
            in_axes=1, out_axes=1
        )
        # vmap over policy dimension (N), return N in dim=2
        td_error_fn = jax.vmap(td_error_fn, in_axes=(
            2, None, 2, 2, None, None, None, None), out_axes=2)

        # vmap over cumulant dimension (C), return in dim=3
        td_error_fn = jax.vmap(td_error_fn, in_axes=(
            4, None, 4, None, 2, None, None, None), out_axes=3)

        # [T, B, N, C]
        sf_t, target_sf_t = td_error_fn(
            online_sf[:-1],  # [T, B, N, A, C]
            data.action[:-1],  # [T, B]
            target_sf[1:],  # [T, B, N, A, C]
            selector_actions[1:],  # [T, B, N]
            cumulants[1:],  # [T, B, C]
            discounts[1:],
            is_last[1:],
            lambda_[1:]
        )

        # Ensure target = 0 when episode terminates
        target_sf_t = target_sf_t * data.discount[:-1, :, None, None]
        batch_td_error = target_sf_t - sf_t

        # Ensure loss = 0 when episode truncates
        truncated = (data.discount + is_last) > 1
        loss_mask = (1 - truncated).astype(batch_td_error.dtype)[:-1, :, None, None]
        batch_td_error = batch_td_error * loss_mask

        # [T, B, N, C]
        batch_loss = 0.5 * jnp.square(batch_td_error)

        # [B]
        batch_loss_mean = (batch_loss * loss_mask).mean(axis=(0, 2, 3))

        metrics = {
            '0.sf_loss': batch_loss.mean(),
            '0.sf_td': jnp.abs(batch_td_error).mean(),
            '1.cumulants': cumulants.mean(),
            'z.sf_mean': online_sf.mean(),
            'z.sf_var': online_sf.var(),
        }

        if self.logger.learner_log_extra is not None:
            self.logger.learner_log_extra({
                'data': data,
                'cumulants': cumulants[1:],
                'td_errors': jnp.abs(batch_td_error),  # [T, B, N, C]
                'mask': loss_mask[:, :, 0, 0],  # [T, B]
                'sf_values': online_sf[:-1],  # [T+1, B, N, A, C]
                'sf_loss': batch_loss,  # [T, B, N, C]
                'sf_target': target_sf_t,
                'n_updates': steps,
            })

        return batch_td_error, batch_loss_mean, metrics  # [T, B, N, C], [B]

def make_loss_fn_class(config) -> vbb.RecurrentLossFn:
  return functools.partial(
      UsfaR2D2LossFn,
      discount=config['GAMMA'],
      )

def make_craftax_agent(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        example_timestep: TimeStep,
        rng: jax.random.PRNGKey) -> Tuple[nn.Module, Params, vbb.AgentResetFn]:

    # just 1 task
    import ipdb; ipdb.set_trace()
    train_tasks = jnp.expand_dims(example_timestep.observation.task_w, axis=0)

    sf_head = SfGpiHead(
        num_actions=env.num_actions(env_params),
        state_features_dim=example_timestep.observation.task_w.shape[-1],
        nsamples=config.get('NSAMPLES', 1),
        eval_task_support=config.get('EVAL_TASK_SUPPORT', 'train'),
        train_tasks=train_tasks,
        num_layers=config.get('NUM_SF_LAYERS', 2),
        hidden_dim=config.get('SF_HIDDEN_DIM', 512),
    )

    agent = UsfaAgent(
        observation_encoder=CraftaxObsEncoder(
          hidden_dim=config["MLP_HIDDEN_DIM"],
          num_layers=config['NUM_MLP_LAYERS'],
          activation=config['ACTIVATION'],
          norm_type=config.get('NORM_TYPE', 'none'),
          structured_inputs=config.get('STRUCTURED_INPUTS', False)
          ),
        rnn=vbb.ScannedRNN(hidden_dim=config["AGENT_RNN_DIM"]),
        sf_head=sf_head
    )

    rng, _rng = jax.random.split(rng)
    network_params = agent.init(
        _rng, example_timestep,
        method=agent.initialize)

    def reset_fn(params, example_timestep, reset_rng):
      batch_dims = (example_timestep.reward.shape[0],)
      return agent.apply(
          params,
          batch_dims=batch_dims,
          rng=reset_rng,
          method=agent.initialize_carry)

    return agent, network_params, reset_fn