
from jaxneurorl.agents.alphazero import *
from networks import CraftaxObsEncoder

class AlphaZeroAgent(nn.Module):

    action_dim: int

    observation_encoder: nn.Module
    rnn: vbb.ScannedRNN
    env: environment.Environment
    env_params: environment.EnvParams
    test_env_params: environment.EnvParams

    num_bins: int = 101

    def setup(self):

        self.policy_fn = MLP(hidden_dim=512, num_layers=1, out_dim=self.action_dim)
        self.value_fn = MLP(hidden_dim=512, num_layers=1, out_dim=self.num_bins)

    def initialize(self, x: TimeStep):
        """Only used for initialization."""
        # [B, D]
        rng = jax.random.PRNGKey(0)
        batch_dims = (x.reward.shape[0],)
        rnn_state = self.initialize_carry(rng, batch_dims)
        predictions, rnn_state = self.__call__(rnn_state, x, rng)
        dummy_action = jnp.zeros(batch_dims, dtype=jnp.int32)

        state = jax.tree_map(lambda x: x[:, None], predictions.state)
        dummy_action = jax.tree_map(lambda x: x[:, None], dummy_action)
        jax.vmap(self.apply_model, (0,0,None), 0)(state, dummy_action, rng)

    def initialize_carry(self, *args, **kwargs):
        """Initializes the RNN state."""
        return self.rnn.initialize_carry(*args, **kwargs)

    def __call__(self, rnn_state, x: TimeStep, rng: jax.random.PRNGKey) -> Tuple[Predictions, RnnState]:

        embedding = self.observation_encoder(x.observation)

        rnn_in = vbb.RNNInput(obs=embedding, reset=x.first())
        rng, _rng = jax.random.split(rng)
        new_rnn_state, rnn_out = self.rnn(rnn_state, rnn_in, _rng)

        policy_logits = self.policy_fn(rnn_out)
        value_logits = self.value_fn(rnn_out)
        predictions = Predictions(
            policy_logits=policy_logits,
            value_logits=value_logits,
            state=AgentState(
                timestep=x,
                rnn_state=new_rnn_state)
            )

        return predictions, new_rnn_state

    def unroll(self, rnn_state, xs: TimeStep, rng: jax.random.PRNGKey) -> Tuple[Predictions, RnnState]:
        # rnn_state: [B]
        # xs: [T, B]

        embedding = jax.vmap(self.observation_encoder)(xs.observation)

        rnn_in = vbb.RNNInput(obs=embedding, reset=xs.first())
        rng, _rng = jax.random.split(rng)
        new_rnn_state, new_rnn_states = self.rnn.unroll(rnn_state, rnn_in, _rng)

        rnn_out = self.rnn.output_from_state(new_rnn_states)
        policy_logits = nn.BatchApply(self.policy_fn)(rnn_out)
        value_logits = nn.BatchApply(self.value_fn)(rnn_out)
        predictions = Predictions(
            policy_logits=policy_logits,
            value_logits=value_logits,
            state=AgentState(
                timestep=xs,
                rnn_state=new_rnn_states)
            )
        return predictions, new_rnn_state

    def apply_model(
          self,
          state: AgentState,
          action: jnp.ndarray,
          rng: jax.random.PRNGKey,
          evaluation: bool = False,
      ) -> Tuple[Predictions, RnnState]:
        """This applies the model to each element in the state, action vectors.

        Args:
            state (State): states. [1, D]
            action (jnp.ndarray): actions to take on states. [1]

        Returns:
            Tuple[ModelOutput, State]: muzero outputs and new states for 
              each state state action pair.
        """
        assert action.shape[0] == 1, 'function only accepts batchsize=1 due to inability to vmap over environment. please use vmap to get these dimensions.'
        rng, rng_ = jax.random.split(rng)
        env_params = self.test_env_params if evaluation else self.env_params
        timestep = jax.tree_map(lambda x: x[0], state.timestep)
        next_timestep = self.env.step(rng_, timestep, action[0], env_params)
        next_timestep = jax.tree_map(lambda x: x[None], next_timestep)

        rng, rng_ = jax.random.split(rng)
        return self.__call__(state.rnn_state, next_timestep, rng_)


def make_agent(
        config: dict,
        env: environment.Environment,
        env_params: environment.EnvParams,
        example_timestep: TimeStep,
        rng: jax.random.PRNGKey,
        model_env: environment.Environment,
        ) -> Tuple[nn.Module, Params, vbb.AgentResetFn]:


    agent = AlphaZeroAgent(
        action_dim=env.action_space(env_params).n,
        observation_encoder=CraftaxObsEncoder(
          hidden_dim=config["MLP_HIDDEN_DIM"],
          num_layers=config['NUM_MLP_LAYERS'],
          activation=config['ACTIVATION'],
          norm_type=config.get('NORM_TYPE', 'none'),
          structured_inputs=config.get('STRUCTURED_INPUTS', False)
          ),
        rnn=vbb.ScannedRNN(
            hidden_dim=config["AGENT_RNN_DIM"],
            unroll_output_state=True),
        num_bins=config['NUM_BINS'],
        env=model_env,
        env_params=env_params,
        test_env_params=env_params,
    )

    rng, _rng = jax.random.split(rng)
    network_params = agent.init(
        _rng, example_timestep,
        method=agent.initialize)

    def reset_fn(params, example_timestep, reset_rng):
      batch_dims = example_timestep.reward.shape
      return agent.apply(
          params,
          batch_dims=batch_dims,
          rng=reset_rng,
          method=agent.initialize_carry)

    return agent, network_params, reset_fn