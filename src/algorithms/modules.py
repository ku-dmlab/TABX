from flax import nnx
import jax.numpy as jnp


class Qmixer(nnx.Module):
    def __init__(self, state_dim, embed_dim, n_agents, rngs):
        self.hyper_w_1 = nnx.Linear(state_dim, embed_dim * n_agents, rngs=rngs)
        self.hyper_w_final = nnx.Linear(state_dim, embed_dim, rngs=rngs)
        self.hyper_b_1 = nnx.Linear(state_dim, embed_dim, rngs=rngs)
        self.layer_norm = nnx.LayerNorm(embed_dim * n_agents, rngs=rngs)
        self.layer_norm_2 = nnx.LayerNorm(embed_dim, rngs=rngs)

        self.value = nnx.Sequential(
            nnx.Linear(state_dim, embed_dim, rngs=rngs),
            nnx.LayerNorm(embed_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(embed_dim, 1, rngs=rngs),
        )

    def __call__(self, agent_qs, states):
        states = states + 1e-6
        batch_size = agent_qs.shape[0]
        state_dim = states.shape[1]
        n_agents = agent_qs.shape[2]
        w1 = jnp.abs(self.layer_norm(self.hyper_w_1(states)))
        b1 = self.hyper_b_1(states)
        w1 = w1.reshape(batch_size, n_agents, -1)
        b1 = b1.reshape(batch_size, 1, -1)
        hidden = nnx.elu(agent_qs @ w1 + b1)
        w_final = jnp.abs(self.layer_norm_2(self.hyper_w_final(states)))
        w_final = w_final.reshape(batch_size, -1, 1)

        v = self.value(states).reshape(batch_size, 1, 1)

        y = hidden @ w_final + v

        q_tot = y.reshape(batch_size, -1, 1)

        return q_tot


class Qnetwork(nnx.Module):
    def __init__(self, state_dim, action_dim, layer_dim=64, rngs=nnx.Rngs(0)):
        self.layer = nnx.Sequential(
            nnx.Linear(state_dim, layer_dim, rngs=rngs),
            nnx.LayerNorm(layer_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(layer_dim, layer_dim, rngs=rngs),
            nnx.LayerNorm(layer_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(layer_dim, action_dim, rngs=rngs),
        )

    def __call__(self, state):
        return self.layer(state)


class ValueNetwork(nnx.Module):
    def __init__(self, state_dim, layer_dim=64, rngs=nnx.Rngs(0)):
        self.layer = nnx.Sequential(
            nnx.Linear(state_dim, layer_dim, rngs=rngs),
            nnx.LayerNorm(layer_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(layer_dim, layer_dim, rngs=rngs),
            nnx.LayerNorm(layer_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(layer_dim, 1, rngs=rngs),
        )

    def __call__(self, state):
        return self.layer(state)
