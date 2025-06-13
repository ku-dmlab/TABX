from collections import namedtuple
from src.algorithms.modules import Qnetwork, ValueNetwork
import tensorflow_probability.substrates.jax as tfp
import jax
import jax.numpy as jnp
from flax import nnx
import optax

tfd = tfp.distributions
tfb = tfp.bijectors


def sample_action(policy, states, key):
    logits = policy(states)
    sample = tfd.Categorical(logits=logits).sample(seed=key)
    log_probs = tfd.Categorical(logits=logits).log_prob(sample)
    return sample, log_probs


NetworkState = namedtuple("NetworkState", ["graphdef", "state"])
TrainState = namedtuple("TrainState", ["pi1_state", "pi2_state", "value_state", "key"])
Model = namedtuple("Model", ["network", "optimizer"])


def init_train_state(config):
    pi1 = Qnetwork(config.state_dim, config.action_dim, rngs=nnx.Rngs(config.seed))
    pi2 = Qnetwork(config.state_dim, config.action_dim, rngs=nnx.Rngs(config.seed))
    value = ValueNetwork(config.state_dim, rngs=nnx.Rngs(config.seed))

    pi1_optimizer = nnx.Optimizer(pi1, optax.adam(learning_rate=config.lr))
    pi2_optimizer = nnx.Optimizer(pi2, optax.adam(learning_rate=config.lr))
    value_optimizer = nnx.Optimizer(value, optax.adam(learning_rate=config.lr))

    (pi1_gd, pi1_state) = nnx.split((pi1, pi1_optimizer))
    (pi2_gd, pi2_state) = nnx.split((pi2, pi2_optimizer))
    (value_gd, value_state) = nnx.split((value, value_optimizer))

    return TrainState(
        pi1_state=NetworkState(pi1_gd, pi1_state),
        pi2_state=NetworkState(pi2_gd, pi2_state),
        value_state=NetworkState(value_gd, value_state),
        key=jax.random.key(config.seed),
    )


def get_model(state: NetworkState) -> Model:
    network, optimizer = nnx.merge(state.graphdef, state.state)
    return network, optimizer


Batch = namedtuple(
    "batch",
    [
        "observation",
        "reward",
        "done",
        "log_pi1",
        "log_pi2",
        "a1",
        "a2",
        "advantages",
        "returns",
        "values",
    ],
)


def train_step(config, train_state: TrainState, batch: Batch):
    value, value_optimizer = get_model(train_state.value_state)

    def value_loss(value):
        values = value(batch.observation)
        clip_value = jnp.clip(
            values, batch.values - config.clip_value, batch.values + config.clip_value
        )
        diff = (batch.returns.reshape(values.shape) - values) ** 2
        clip_diff = (batch.returns.reshape(values.shape) - clip_value) ** 2
        loss = jnp.maximum(diff, clip_diff)
        return jnp.mean(loss), diff

    (v_loss, diff), grads = nnx.value_and_grad(value_loss, has_aux=True)(value)
    value_optimizer.update(grads)
    value_state_ = nnx.state((value, value_optimizer))

    pi1, pi1_optimizer = get_model(train_state.pi1_state)
    pi2, pi2_optimizer = get_model(train_state.pi2_state)

    def ppo_loss(pi1, pi2):
        logits1 = pi1(batch.observation)
        logits2 = pi2(batch.observation)

        dist1 = tfd.Categorical(logits=logits1)
        dist2 = tfd.Categorical(logits=logits2)

        log_pi1 = dist1.log_prob(batch.a1.flatten())
        log_pi2 = dist2.log_prob(batch.a2.flatten())

        ratio1 = jnp.exp(log_pi1 - batch.log_pi1.reshape(log_pi1.shape))
        ratio2 = jnp.exp(log_pi2 - batch.log_pi2.reshape(log_pi2.shape))

        pi1_loss = jnp.minimum(
            ratio1 * batch.advantages.reshape(-1),
            jnp.clip(ratio1, 1 - config.clip_ratio, 1 + config.clip_ratio)
            * batch.advantages.reshape(-1),
        )
        pi2_loss = jnp.minimum(
            ratio2 * batch.advantages.reshape(-1),
            jnp.clip(ratio2, 1 - config.clip_ratio, 1 + config.clip_ratio)
            * batch.advantages.reshape(-1),
        )
        loss = pi1_loss.mean() + pi2_loss.mean()
        return -loss - config.entropy_coef * (dist1.entropy().mean() + dist2.entropy().mean())

    (policy_loss, grads) = nnx.value_and_grad(ppo_loss, argnums=[0, 1])(pi1, pi2)
    pi1_optimizer.update(grads[0])
    pi2_optimizer.update(grads[1])

    pi1_state_ = nnx.state((pi1, pi1_optimizer))
    pi2_state_ = nnx.state((pi2, pi2_optimizer))

    train_state = train_state._replace(
        pi1_state=train_state.pi1_state._replace(state=pi1_state_),
        pi2_state=train_state.pi2_state._replace(state=pi2_state_),
        value_state=train_state.value_state._replace(state=value_state_),
    )

    return train_state, (v_loss, policy_loss, batch.reward.sum())
