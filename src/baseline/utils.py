import jax
import jax.numpy as jnp
from flax.struct import dataclass
from flax import nnx
import os


def get_abs_path(path):
    if not os.path.isdir(path):
        path = os.getcwd() + path
    return path


@dataclass
class NetworkState:
    graphdef: nnx.GraphDef
    state: nnx.State


def rnn_result(model, init_shape, feature, done):
    def rnn_scan_body(carry, xs):
        (hidden_state,) = carry
        feature, done = xs
        print(feature.shape, done.shape, hidden_state.shape)
        model_output = model(hidden_state, feature)
        next_hidden_state = model_output[0]
        output = model_output[1:]
        return (next_hidden_state * ~done,), output

    _, output = jax.lax.scan(rnn_scan_body, (jnp.zeros(init_shape),), (feature, done))

    return output


def get_gae(common_reward, done, values, last_value, gamma, lamda):
    def calculate_gae(carry, xs):
        last_gae, v_t1, returns = carry
        reward, done, v = xs
        delta = reward + gamma * v_t1 * (1 - done) - v
        last_gae = delta + gamma * lamda * last_gae * (1 - done)
        returns = reward + gamma * returns * (1 - done)
        return (last_gae, v, returns), (
            last_gae,
            returns,
        )

    _, outputs = jax.lax.scan(
        calculate_gae,
        (jnp.array([0.0]), last_value, jnp.array([0.0])),
        (common_reward, done, values),
        reverse=True,
        unroll=16,
    )
    gae, _ = outputs
    return gae, gae + values


def get_model(state: NetworkState):
    network, optimizer = nnx.merge(state.graphdef, state.state)
    return network, optimizer
