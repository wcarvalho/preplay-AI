"""
Recurrent Q-learning for Craftax.
Add an auxiliary task for predicting achievements. This predicts both which has been achieved at this timestep and which is achievable.
"""



import jax
from typing import Callable, NamedTuple


import flax.linen as nn

import flax
from flax import struct
import jax.numpy as jnp
from gymnax.environments import environment
import matplotlib.pyplot as plt


from jaxneurorl.agents.basics import TimeStep
from jaxneurorl.agents import value_based_basics as vbb
from jaxneurorl.agents.qlearning import *
from jaxneurorl import losses
from networks import CraftaxObsEncoder


Agent = nn.Module
Params = flax.core.FrozenDict
AgentState = flax.struct.PyTreeNode
RNNInput = vbb.RNNInput

class Predictions(NamedTuple):
    q_vals: jax.Array
    achievements: jax.Array
    rnn_states: jax.Array

class RnnAgent(nn.Module):
    """_summary_

    - observation encoder: CNN
    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """

    observation_encoder: nn.Module
    rnn: vbb.ScannedRNN
    q_fn: nn.Module
    achieve_fn: nn.Module


    def initialize(self, x: TimeStep):
        """Only used for initialization."""

        rng = jax.random.PRNGKey(0)
        batch_dims = (x.reward.shape[0],)
        rnn_state = self.initialize_carry(rng, batch_dims)

        return self(rnn_state, x, rng)

    def __call__(self, rnn_state, x: TimeStep, rng: jax.random.PRNGKey):

        embedding = self.observation_encoder(x.observation)

        rnn_in = RNNInput(obs=embedding, reset=x.first())
        rng, _rng = jax.random.split(rng)
        new_rnn_state, rnn_out = self.rnn(rnn_state, rnn_in, _rng)

        q_vals = self.q_fn(rnn_out)
        achieve_vals = self.achieve_fn(rnn_out)
        return Predictions(q_vals, achieve_vals, rnn_out), new_rnn_state

    def unroll(self, rnn_state, xs: TimeStep, rng: jax.random.PRNGKey):
        # rnn_state: [B]
        # xs: [T, B]

        embedding = nn.BatchApply(self.observation_encoder)(xs.observation)

        rnn_in = RNNInput(obs=embedding, reset=xs.first())
        rng, _rng = jax.random.split(rng)
        new_rnn_state, rnn_out = self.rnn.unroll(rnn_state, rnn_in, _rng)

        q_vals = nn.BatchApply(self.q_fn)(rnn_out)
        achieve_vals = nn.BatchApply(self.achieve_fn)(rnn_out)
        return Predictions(q_vals, achieve_vals, rnn_out), new_rnn_state

    def initialize_carry(self, *args, **kwargs):
        """Initializes the RNN state."""
        return self.rnn.initialize_carry(*args, **kwargs)

@struct.dataclass
class R2D2LossFn(vbb.RecurrentLossFn):

  """Loss function of R2D2.
  
  https://openreview.net/forum?id=r1lyTjAqYX
  """

  extract_q: Callable[[jax.Array], jax.Array] = lambda preds: preds.q_vals
  aux_coeff: float = 1.0


  def error(self, data, online_preds, online_state, target_preds, target_state, steps, **kwargs):
    """R2D2 learning.
    """

    float = lambda x: x.astype(jnp.float32)
    # Get value-selector actions from online Q-values for double Q-learning.
    selector_actions = jnp.argmax(self.extract_q(online_preds), axis=-1)  # [T+1, B]

    # Preprocess discounts & rewards.
    discounts = float(data.discount)*self.discount
    lambda_ = jnp.ones_like(data.discount)*self.lambda_
    rewards = float(data.reward)
    rewards = rewards - self.step_cost
    is_last = float(data.is_last)

    # Get N-step transformed TD error and loss.
    batch_td_error_fn = jax.vmap(
        partial(losses.q_learning_lambda_td, tx_pair=self.tx_pair),
        in_axes=1,
        out_axes=1)

    # [T, B]
    q_t, target_q_t = batch_td_error_fn(
        self.extract_q(online_preds)[:-1],  # [T+1] --> [T]
        data.action[:-1],    # [T+1] --> [T]
        self.extract_q(target_preds)[1:],  # [T+1] --> [T]
        selector_actions[1:],  # [T+1] --> [T]
        rewards[1:],        # [T+1] --> [T]
        discounts[1:],
        is_last[1:],
        lambda_[1:])      # [T+1] --> [T]

    # ensure target = 0 when episode terminates
    target_q_t = target_q_t*data.discount[:-1]
    batch_td_error = target_q_t - q_t

    # ensure loss = 0 when episode truncates
    # truncated if FINAL time-step but data.discount = 1.0, something like [1,1,2,1,1]
    truncated = (data.discount+is_last) > 1  # truncated is discount on AND is last
    loss_mask = (1-truncated).astype(batch_td_error.dtype)[:-1]
    batch_td_error = batch_td_error*loss_mask

    # [T, B]
    batch_loss = 0.5 * jnp.square(batch_td_error)

    # [B]
    batch_loss_mean = (batch_loss*loss_mask).mean(0)

    metrics = {
        '0.q_loss': batch_loss.mean(),
        '0.q_td': jnp.abs(batch_td_error).mean(),
        '1.reward': rewards[1:].mean(),
        '1.reward_min': rewards[1:].min(),
        '1.reward_max': rewards[1:].max(),
        'z.q_mean': self.extract_q(online_preds).mean(),
        'z.q_var': self.extract_q(online_preds).var(),
        }
    
    ###########
    # Add auxiliary for achievements and achievable
    ###########
    if self.aux_coeff > 0:
      # [T, B, D]
      achievements = data.timestep.observation.achievements.astype(jnp.float32)
      achievable = data.timestep.observation.achievable.astype(jnp.float32)

      # current prediction is for next timestep
      achieve_predictions = online_preds.achievements[:-1]
      achieve = jnp.concatenate((achievements, achievable), axis=-1)[1:]

      # [T, B, D], sum over achievements
      achieve_loss = 0.5 * jnp.square(achieve - achieve_predictions).sum(-1)

      achieve_loss = achieve_loss.mean(axis=0)  # [B]
      metrics['0.achieve_loss'] = achieve_loss
      batch_loss_mean = batch_loss_mean + self.aux_coeff*achieve_loss

    if self.logger.learner_log_extra is not None:
        self.logger.learner_log_extra({
        'data': data,
        'td_errors': jnp.abs(batch_td_error),        # [T]
        'mask': loss_mask,                 # [T]
        'q_values': self.extract_q(online_preds),    # [T, B]
        'q_loss': batch_loss,                        #[ T, B]
        'q_target': target_q_t,
        'n_updates': steps,
        })

    return batch_td_error, batch_loss_mean, metrics  # [T-1, B], [B]#

def make_loss_fn_class(config) -> vbb.RecurrentLossFn:
  return functools.partial(
     R2D2LossFn,
     discount=config['GAMMA'],
     tx_pair=rlax.SIGNED_HYPERBOLIC_PAIR if config.get('TX_PAIR', 'none') == 'hyperbolic' else rlax.IDENTITY_PAIR,
     step_cost=config.get('STEP_COST', 0.),
     aux_coeff=config.get('AUX_COEFF', 1.0))

def make_craftax_agent(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        example_timestep: TimeStep,
        rng: jax.random.PRNGKey):

    cell_type = config.get('RNN_CELL_TYPE', 'OptimizedLSTMCell')
    if cell_type.lower() == 'none':
        rnn = vbb.DummyRNN()
    else:
        rnn = vbb.ScannedRNN(
            hidden_dim=config["AGENT_RNN_DIM"],
            cell_type=cell_type,
            )
    achievable = example_timestep.observation.achievable
    achievements = example_timestep.observation.achievements
    n_achieve = achievable.shape[-1] + achievements.shape[-1]


    agent = RnnAgent(
        observation_encoder=CraftaxObsEncoder(
          hidden_dim=config["MLP_HIDDEN_DIM"],
          num_layers=config['NUM_MLP_LAYERS'],
          activation=config['ACTIVATION'],
          norm_type=config.get('NORM_TYPE', 'none'),
          structured_inputs=config.get('STRUCTURED_INPUTS', False)
          ),
        rnn=rnn,
        q_fn=MLP(
           hidden_dim=config.get('Q_HIDDEN_DIM', 512),
           num_layers=config.get('NUM_Q_LAYERS', 2),
           out_dim=env.action_space(env_params).n),
        achieve_fn=MLP(
           hidden_dim=config.get('Q_HIDDEN_DIM', 512),
           num_layers=config.get('NUM_Q_LAYERS', 2),
           out_dim=n_achieve)
    )

    rng, _rng = jax.random.split(rng)
    network_params = agent.init(
        _rng, example_timestep, method=agent.initialize)

    def reset_fn(params, example_timestep, reset_rng):
      batch_dims = (example_timestep.reward.shape[0],)
      return agent.apply(
          params,
          batch_dims=batch_dims,
          rng=reset_rng,
          method=agent.initialize_carry)

    return agent, network_params, reset_fn