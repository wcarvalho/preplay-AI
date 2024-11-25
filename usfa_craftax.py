
import wandb
import matplotlib.pyplot as plt
from visualizer import plot_frames
from craftax.craftax.renderer import render_craftax_pixels
from craftax.craftax import constants
from craftax.craftax.constants import Action

from jaxneurorl.agents.usfa import *
from networks import CraftaxObsEncoder

def render_fn(state):
    image = render_craftax_pixels(state, block_pixel_size=constants.BLOCK_PIXEL_SIZE_IMG)
    return image/255.0
render_fn = jax.jit(render_fn)

@struct.dataclass
class UsfaR2D2LossFn(vbb.RecurrentLossFn):
    extract_cumulants: Callable = lambda data: data.timestep.observation.state_features
    extract_task: Callable = lambda data: data.timestep.observation.task_w
    aux_coeff: float = 1.0

    def error(self, data, online_preds, online_state, target_preds, target_state, steps, **kwargs):
        # Prepare Data
        online_sf = online_preds.sf  # [T+1, B, N, A, C]
        online_w = data.timestep.observation.task_w  # [T+1, B, C]
        target_sf = target_preds.sf  # [T+1, B, N, A, C]

        # NOTE: main change was to concat these
        achievements = data.timestep.observation.achievements.astype(online_sf.dtype)
        achievable = data.timestep.observation.achievable.astype(online_sf.dtype)
        cumulants = jnp.concatenate((achievements, achievable), axis=-1)
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
            partial(losses.q_learning_lambda_td, tx_pair=self.tx_pair),
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
        loss_mask = (1 - truncated).astype(batch_td_error.dtype)

        loss_mask_td = loss_mask[:-1, :, None, None]
        batch_td_error = batch_td_error * loss_mask_td

        batch_sq_error = jnp.square(batch_td_error)

        ################
        # Compute SF loss
        ################
        # [T, B, N, C]
        batch_sf_loss = 0.5 * batch_sq_error

        # [T, B, N, C] --> [B]
        batch_sf_loss_mean = batch_sf_loss.sum(axis=3).mean(axis=(0, 2))

        ################
        # Compute Q-loss
        ################
        # NOTE: CHANGE is to weight TD errors by task weights
        task_w = data.timestep.observation.task_w[:-1]
        task_w = task_w[:, :, None]  # add policy dimension (N)

        # [T, B, N, C] --> [T, B, N]
        batch_q_loss = 0.5 * jnp.square(
            (task_w*batch_td_error).sum(axis=-1))
        # [T, B, N] --> [B]
        batch_q_loss_mean = batch_q_loss.mean(axis=(0, 2))

        batch_loss_mean = batch_q_loss_mean + self.aux_coeff*batch_sf_loss_mean

        metrics = {
            '0.sf_loss': batch_sf_loss_mean.mean(),
            '0.q_loss': batch_q_loss_mean.mean(),
            '0.sf_td': jnp.abs(batch_td_error).mean(),
            '1.cumulants': cumulants.mean(),
            'z.sf_mean': online_sf.mean(),
            'z.sf_var': online_sf.var(),
            'z.cumulants_min': cumulants[1:].min(),
            'z.cumulants_max': cumulants[1:].max(),
        }

        if self.logger.learner_log_extra is not None:
            self.logger.learner_log_extra({
                'data': data,
                'cumulants': cumulants[1:],
                'td_errors': jnp.abs(batch_td_error),  # [T, B, N, C]
                'mask': loss_mask,  # [T, B]
                'sf_values': online_sf[:-1],  # [T+1, B, N, A, C]
                'sf_loss': batch_sf_loss,  # [T, B, N, C]
                'sf_target': target_sf_t,
                'n_updates': steps,
            })

        return batch_td_error, batch_loss_mean, metrics  # [T, B, N, C], [B]

def make_loss_fn_class(config) -> vbb.RecurrentLossFn:
  return functools.partial(
      UsfaR2D2LossFn,
      discount=config['GAMMA'],
      step_cost=config.get('STEP_COST', 0.),
      aux_coeff=config.get('AUX_COEFF', 1.0),
      tx_pair=rlax.SIGNED_HYPERBOLIC_PAIR if config.get('TX_PAIR', 'none') == 'hyperbolic' else rlax.IDENTITY_PAIR,
      )

def make_craftax_agent(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        example_timestep: TimeStep,
        rng: jax.random.PRNGKey) -> Tuple[nn.Module, Params, vbb.AgentResetFn]:

    # just 1 task
    train_tasks = jnp.expand_dims(example_timestep.observation.task_w, axis=-2)

    sf_head = SfGpiHead(
        num_actions=env.action_space(env_params).n,
        state_features_dim=example_timestep.observation.task_w.shape[-1],
        nsamples=config.get('NSAMPLES', 1),
        eval_task_support=config.get('EVAL_TASK_SUPPORT', 'eval'),
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


# only redoing this
def learner_log_extra(
        data: dict,
        config: dict,
        ):
    def callback(d):
        n_updates = d.pop('n_updates')

        # Extract the relevant data
        # only use data from batch dim = 0
        # [T, B, ...] --> # [T, ...]
        d_ = jax.tree_map(lambda x: x[:, 0], d)

        mask = d_['mask']
        discounts = d_['data'].timestep.discount
        rewards = d_['data'].timestep.reward
        actions = d_['data'].action
        cumulants = d_['cumulants']  # [T, C]
        sf_values = d_['sf_values'][:, 0]  # [T, A, C]
        sf_target = d_['sf_target'][:, 0]  # [T, C]
        # [T, C]
        # vmap over cumulant dimension
        sf_values_taken = jax.vmap(rlax.batched_index, in_axes=(
            2, None), out_axes=1)(sf_values, actions[:-1])
        #sf_td_errors = d_['td_errors']  # [T, C]
        #sf_loss = d_['sf_loss']  # [T, C]

        ##############################
        # SF-learning plots
        ##############################
        # Create a figure with subplots for each cumulant
        num_cumulants = cumulants.shape[-1]
        num_cols = min(6, num_cumulants)
        num_rows = (num_cumulants + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
        axes = axes.flatten()

        for i in range(num_cumulants):
            ax = axes[i]
            time_steps = range(len(cumulants))

            ax.plot(time_steps, cumulants[:, i], label='Cumulants')
            ax.plot(time_steps, sf_values_taken[:, i], label='SF Values Taken')
            ax.plot(time_steps, sf_target[:, i], label='SF Target')
            ax.set_title(f'Cumulant {i+1}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)

        # Remove any unused subplots
        for i in range(num_cumulants, len(axes)):
            fig.delaxes(axes[i])

        # Log the Q-learning figure
        if wandb.run is not wandb.sdk.lib.disabled.RunDisabled:
            wandb.log({f"learner_example/sf-learning": wandb.Image(fig)})
        plt.close(fig)

        ##############################
        # Q-learning plots
        ##############################
        # Plot rewards and q-values in the top subplot
        width = .3
        nT = len(rewards)  # e.g. 20 --> 8

        task = d_['data'].timestep.observation.task_w[:-1]
        q_values_taken = (sf_values_taken*task).sum(-1)
        q_target = (sf_target*task).sum(-1)
        td_errors = jnp.abs(q_target - q_values_taken)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(int(width*nT), 16))
        def format(ax):
            ax.set_xlabel('Time')
            ax.grid(True)
            ax.set_xticks(range(0, len(rewards), 1))
        # Set the same x-limit for all subplots
        x_max = len(rewards)
        
        ax1.plot(rewards, label='Rewards')
        ax1.plot(q_values_taken, label='Q-Values')
        ax1.plot(q_target, label='Q-Targets')
        ax1.set_title('Rewards and Q-Values')
        ax1.set_xlim(0, x_max)
        format(ax1)
        ax1.legend()

        # Plot TD errors in the middle subplot
        ax2.plot(td_errors)
        ax2.set_xlim(0, x_max)
        format(ax2)
        ax2.set_title('TD Errors')

        # Plot episode quantities
        is_last = d_['data'].timestep.last()
        ax3.plot(discounts, label='Discounts')
        ax3.plot(mask, label='mask')
        ax3.plot(is_last, label='is_last')
        ax3.set_xlim(0, x_max)
        format(ax3)
        ax3.set_title('Episode markers')
        ax3.legend()

        # Ensure all subplots have the same x-axis range
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)

        # Adjust the spacing between subplots
        #plt.tight_layout()
        # log
        if wandb.run is not wandb.sdk.lib.disabled.RunDisabled:
            wandb.log({f"learner_example/q-values": wandb.Image(fig)})
        plt.close(fig)

        ##############################
        # plot images of env
        ##############################
        #timestep = jax.tree_map(lambda x: jnp.array(x), d_['data'].timestep)
        timesteps: TimeStep = d_['data'].timestep

        # ------------
        # get images
        # ------------

        #state_images = []
        obs_images = []
        max_len = min(config.get("MAX_EPISODE_LOG_LEN", 40), len(rewards))
        for idx in range(max_len):
            index = lambda y: jax.tree_map(lambda x: x[idx], y)
            obs_image = render_fn(index(d_['data'].timestep.state))
            obs_images.append(obs_image)

        # ------------
        # plot
        # ------------
        actions_taken = [Action(a).name for a in actions]

        def index(t, idx): return jax.tree_map(lambda x: x[idx], t)
        def panel_title_fn(timesteps, i):
            title = f't={i}\n'
            title += f'{actions_taken[i]}\n'
            title += f'r={timesteps.reward[i]}, $\\gamma={timesteps.discount[i]}$'
            return title

        fig = plot_frames(
            timesteps=timesteps,
            frames=obs_images,
            panel_title_fn=panel_title_fn,
            ncols=6)
        if wandb.run is not wandb.sdk.lib.disabled.RunDisabled:
            wandb.log(
                {f"learner_example/trajecotry": wandb.Image(fig)})
        plt.close(fig)

    # this will be the value after update is applied
    n_updates = data['n_updates'] + 1
    is_log_time = n_updates % config["LEARNER_EXTRA_LOG_PERIOD"] == 0

    jax.lax.cond(
        is_log_time,
        lambda d: jax.debug.callback(callback, d),
        lambda d: None,
        data)
