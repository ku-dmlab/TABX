from flax import nnx
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

LOG_STD_MAX = 0.0
LOG_STD_MIN = -3.0


class PQN_Critic(nnx.Module):
    def __init__(self, state_dim, action_dim, layer_dim=64, rngs=nnx.Rngs(0), batch_norm=False):
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm_layer = nnx.BatchNorm(state_dim, rngs=rngs)
        self.layer = nnx.Sequential(
            nnx.Linear(state_dim, layer_dim, rngs=rngs),
            nnx.LayerNorm(layer_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(layer_dim, layer_dim, rngs=rngs),
            nnx.LayerNorm(layer_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(
                layer_dim,
                action_dim,
                rngs=rngs,
            ),
        )

    def __call__(self, state):
        if self.batch_norm:
            state = self.batch_norm_layer(state)
        return self.layer(state)


class QNetwork(nnx.Module):
    def __init__(self, state_dim, action_dim, layer_dim=64, rngs=nnx.Rngs(0)):
        self.layer = nnx.Sequential(
            nnx.Linear(state_dim, layer_dim, rngs=rngs),
            nnx.LayerNorm(layer_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(layer_dim, layer_dim, rngs=rngs),
            nnx.LayerNorm(layer_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(
                layer_dim,
                action_dim,
                rngs=rngs,
                kernel_init=nnx.initializers.orthogonal(jnp.sqrt(0.001)),
            ),
        )

    def __call__(self, state):
        return self.layer(state)


class Critic(nnx.Module):
    def __init__(self, state_dim, layer_dim=64, rngs=nnx.Rngs(0)):
        self.layer = nnx.Sequential(
            nnx.Linear(
                state_dim,
                layer_dim,
                rngs=rngs,
                kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2.0)),
                bias_init=nnx.initializers.zeros,
            ),
            nnx.LayerNorm(layer_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(
                layer_dim,
                layer_dim,
                rngs=rngs,
                kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2.0)),
                bias_init=nnx.initializers.zeros,
            ),
            nnx.LayerNorm(layer_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(
                layer_dim,
                1,
                rngs=rngs,
                kernel_init=nnx.initializers.orthogonal(jnp.sqrt(0.001)),
                bias_init=nnx.initializers.zeros,
            ),
        )

    def __call__(self, state):
        return self.layer(state)


class Policy(nnx.Module):
    def __init__(self, state_dim, action_dim, layer_dim=64, rngs=nnx.Rngs(0)):
        self.layer = nnx.Sequential(
            nnx.Linear(
                state_dim,
                layer_dim,
                rngs=rngs,
                kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2.0)),
                bias_init=nnx.initializers.zeros,
            ),
            nnx.LayerNorm(layer_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(
                layer_dim,
                layer_dim,
                rngs=rngs,
                kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2.0)),
                bias_init=nnx.initializers.zeros,
            ),
            nnx.LayerNorm(layer_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(
                layer_dim,
                action_dim,
                rngs=rngs,
                kernel_init=nnx.initializers.orthogonal(jnp.sqrt(0.01)),
                bias_init=nnx.initializers.zeros,
            ),
        )

    def __call__(self, state):
        return self.layer(state)


class RNNCritic(nnx.Module):
    def __init__(self, state_dim, layer_dim=64, rngs=nnx.Rngs(0)):
        self.layer = nnx.Sequential(
            nnx.Linear(state_dim, layer_dim, rngs=rngs), nnx.LayerNorm(layer_dim, rngs=rngs)
        )
        self.gru = nnx.GRUCell(layer_dim, layer_dim, rngs=rngs)
        self.critic = nnx.Sequential(
            nnx.relu,
            nnx.Linear(layer_dim, layer_dim, rngs=rngs),
            nnx.LayerNorm(layer_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(
                layer_dim,
                1,
                rngs=rngs,
                kernel_init=nnx.initializers.orthogonal(jnp.sqrt(0.001)),
            ),
        )

    def __call__(self, hidden_state, observation):
        # observation shape = (batch_size, observation_dim)
        obs_emb = self.layer(observation)  # (batch_size, layer_dim)
        next_state, output = self.gru(hidden_state, obs_emb)
        return next_state, self.critic(output)

    def initialize_carry(self, shape):
        return self.gru.initialize_carry(shape)


class RNNHybridPolicy(nnx.Module):
    def __init__(self, obs_dim, action_dim, layer_dim=64, rngs=nnx.Rngs(0)):
        self.layer = nnx.Sequential(
            nnx.Linear(
                obs_dim,
                layer_dim,
                rngs=rngs,
                kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2.0)),
                bias_init=nnx.initializers.zeros,
            ),
            nnx.LayerNorm(layer_dim, rngs=rngs),
        )
        self.gru = nnx.GRUCell(
            layer_dim,
            layer_dim,
            rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(1.0)),
            bias_init=nnx.initializers.zeros,
        )
        self.policy = nnx.Sequential(
            nnx.relu,
            nnx.Linear(
                layer_dim,
                layer_dim,
                rngs=rngs,
                kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2.0)),
                bias_init=nnx.initializers.zeros,
            ),
            nnx.LayerNorm(layer_dim, rngs=rngs),
            nnx.relu,
        )

        self.discrete_policy = nnx.Linear(
            layer_dim,
            action_dim,
            rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(0.01)),
            bias_init=nnx.initializers.zeros,
        )
        self.policy_mu = nnx.Linear(
            layer_dim,
            1,
            rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(0.01)),
            bias_init=nnx.initializers.zeros,
        )
        self.policy_std = nnx.Linear(
            layer_dim,
            1,
            rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(0.01)),
            bias_init=nnx.initializers.zeros,
        )

    def __call__(self, hidden_state, observation):
        # observation shape = (batch_size, observation_dim)
        obs_emb = self.layer(observation)  # (batch_size, layer_dim)
        next_state, output = self.gru(hidden_state, obs_emb)
        output = self.policy(output)

        logits = self.discrete_policy(output)
        mean = self.policy_mu(output)
        log_std = jnp.tanh(self.policy_std(output))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return next_state, logits, mean, log_std

    def get_distribution(self, logits, mean, log_std):
        std = jnp.exp(log_std)
        continuous_distribution = tfd.Normal(mean, std)
        continuous_distribution = tfd.TransformedDistribution(
            continuous_distribution, tfb.Chain([tfb.Scale(jnp.pi / 12.0), tfb.Tanh()])
        )[..., 0]
        discrete_distribution = tfd.Categorical(logits=logits)
        return continuous_distribution, discrete_distribution

    def initialize_carry(self, shape):
        return self.gru.initialize_carry(shape)


class RNNActorCritic(nnx.Module):
    def __init__(self, obs_dim, action_dim, layer_dim, rngs=nnx.Rngs(0)):
        self.layer = nnx.Sequential(
            nnx.Linear(obs_dim, layer_dim, rngs=rngs),
            nnx.LayerNorm(layer_dim, rngs=rngs),
            nnx.relu,
        )
        self.gru = nnx.GRUCell(layer_dim, layer_dim, rngs=rngs)
        self.actor = nnx.Sequential(
            nnx.relu,
            nnx.Linear(layer_dim, layer_dim, rngs=rngs),
            nnx.LayerNorm(layer_dim, rngs=rngs),
            nnx.relu,
        )
        self.actor_discrete = nnx.Linear(
            layer_dim,
            action_dim,
            rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(0.001)),
        )
        self.actor_mu = nnx.Linear(
            layer_dim, 1, rngs=rngs, kernel_init=nnx.initializers.orthogonal(jnp.sqrt(0.001))
        )
        self.actor_std = nnx.Linear(
            layer_dim, 1, rngs=rngs, kernel_init=nnx.initializers.orthogonal(jnp.sqrt(0.001))
        )
        self.critic = nnx.Sequential(
            nnx.relu,
            nnx.Linear(layer_dim, layer_dim, rngs=rngs),
            nnx.LayerNorm(layer_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(
                layer_dim,
                1,
                rngs=rngs,
                kernel_init=nnx.initializers.orthogonal(jnp.sqrt(0.001)),
            ),
        )

    def __call__(self, hidden_state, observation):
        obs_emb = self.layer(observation)
        next_state, output = self.gru(hidden_state, obs_emb)

        # Actor
        pi_out = self.actor(output)
        logits = self.actor_discrete(pi_out)
        mean = self.actor_mu(pi_out)
        std = self.actor_mu(pi_out)

        # Critic
        value = self.critic(output)

        return next_state, logits, mean, std, value

    def get_distribution(self, logits, mean, log_std):
        std = jnp.exp(log_std)
        continuous_distribution = tfd.Normal(mean, std)
        continuous_distribution = tfd.TransformedDistribution(
            continuous_distribution, tfb.Chain([tfb.Scale(jnp.pi / 12.0), tfb.Tanh()])
        )
        discrete_distribution = tfd.Categorical(logits=logits)
        return continuous_distribution, discrete_distribution

    def initialize_carry(self, shape):
        return self.gru.initialize_carry(shape)
