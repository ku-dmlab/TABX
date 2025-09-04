from flax import nnx
import jax.numpy as jnp


class Critic(nnx.Module):
    def __init__(self, state_dim, layer_dim=64, rngs=nnx.Rngs(0)):
        self.layer = nnx.Sequential(
            nnx.Linear(state_dim, layer_dim, rngs=rngs),
            nnx.LayerNorm(layer_dim, rngs=rngs),
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

    def __call__(self, state):
        return self.layer(state)


class Policy(nnx.Module):
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


class RNNCritic(nnx.Module):
    def __init__(self, state_dim, layer_dim=64, rngs=nnx.Rngs(0)):
        self.layer = nnx.Sequential(
            nnx.Linear(state_dim, layer_dim, rngs=rngs), nnx.LayerNorm(layer_dim, rngs=rngs)
        )
        self.gru = nnx.GRUCell(layer_dim, layer_dim, rngs=rngs)
        self.value = nnx.Sequential(
            nnx.relu,
            nnx.Linear(
                layer_dim,
                layer_dim,
                rngs=rngs,
            ),
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
        return next_state, self.value(output)

    def initialize_carry(self, shape):
        return self.gru.initialize_carry(shape)


class RNNHybridPolicy(nnx.Module):
    def __init__(self, obs_dim, action_dim, layer_dim=64, rngs=nnx.Rngs(0)):
        self.layer = nnx.Sequential(
            nnx.Linear(obs_dim, layer_dim, rngs=rngs),
            nnx.LayerNorm(layer_dim, rngs=rngs),
        )
        self.gru = nnx.GRUCell(layer_dim, layer_dim, rngs=rngs)
        self.policy = nnx.Sequential(
            nnx.relu,
            nnx.Linear(
                layer_dim,
                layer_dim,
                rngs=rngs,
            ),
            nnx.LayerNorm(layer_dim, rngs=rngs),
            nnx.relu,
        )

        self.discrete_policy = nnx.Linear(
            layer_dim,
            action_dim,
            rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(0.001)),
        )

        self.policy_mu = nnx.Linear(
            layer_dim,
            1,
            rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(0.001)),
        )

        self.policy_std = nnx.Linear(
            layer_dim,
            1,
            rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(0.001)),
        )

    def __call__(self, hidden_state, observation):
        # observation shape = (batch_size, observation_dim)
        obs_emb = self.layer(observation)  # (batch_size, layer_dim)
        next_state, output = self.gru(hidden_state, obs_emb)
        output = self.policy(output)

        logits = self.discrete_policy(output)
        mean = self.policy_mu(output)
        std = self.policy_std(output)

        return next_state, logits, mean, std

    def initialize_carry(self, shape):
        return self.gru.initialize_carry(shape)
