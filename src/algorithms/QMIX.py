
from collections import namedtuple
import jax.numpy as jnp
from modules import Qmixer, Qnetwork
NetworkState = namedtuple('NetworkState', ['graphdef', 'state', 'target_params'])
TrainState = namedtuple('TrainState', ['q1_state', 'q2_state', 'q_mix_state', 'step'])
Model = namedtuple('Model', ['network', 'optimizer', 'target_network'])

from flax import nnx
import optax





def get_model(state: NetworkState) -> Model:
    network, optimizer = nnx.merge(state.graphdef, state.state)
    _, other_variables = state.state.split(nnx.Param, ...)
    target_network, _ = nnx.merge(state.graphdef, state.target_params, other_variables)
    
    return Model(network, optimizer, target_network)

def train_step(config, train_state: TrainState, batch):
    step = train_state.step
    
    q1, q1_optimizer, q1_target = get_model(train_state.q1_state)
    q2, q2_optimizer, q2_target = get_model(train_state.q2_state)
    q_mix, q_mix_optimizer, q_mix_target = get_model(train_state.q_mix_state)
    argmax_next_q1 = q1(batch.q1_obs).argmax(axis=-1, keepdims=True)  
    argmax_next_q2 = q2(batch.q2_obs).argmax(axis=-1, keepdims=True)

    next_target_q1 = jnp.take_along_axis(q1_target(batch.q1_next_obs), argmax_next_q1, axis = -1)
    next_target_q2 = jnp.take_along_axis(q2_target(batch.q2_next_obs), argmax_next_q2, axis = -1)

    next_agent_qs = jnp.stack([next_target_q1, next_target_q2], axis = -1)
    target_next_q_tot = q_mix_target(next_agent_qs, batch.q1_obs[:, None]).reshape(batch.reward.shape)
    target_next_q_tot = batch.reward + (1-batch.done) * target_next_q_tot

    def q_loss(q1, q2, q_mixer):
        q1_vals = jnp.take_along_axis(q1(batch.q1_obs), batch.a1[:, None], axis = 1)
        q2_vals = jnp.take_along_axis(q2(batch.q2_obs), batch.a2[:, None], axis = 1)
        agent_qs = jnp.stack([q1_vals, q2_vals], axis = -1)
        q_tot = q_mixer(agent_qs, batch.q1_obs[:, None])

        return jnp.mean((q_tot.reshape(target_next_q_tot.shape) - target_next_q_tot) ** 2)
    (loss, grads) = nnx.value_and_grad(q_loss, argnums = [0, 1, 2])(q1, q2, q_mix)

    q1_optimizer.update(grads[0])
    q2_optimizer.update(grads[1])
    q_mix_optimizer.update(grads[2])

    q1_state = nnx.state((q1, q1_optimizer))
    q2_state = nnx.state((q2, q2_optimizer))
    q_mix_state = nnx.state((q_mix, q_mix_optimizer))


    q1_params = q1_state.filter(nnx.Param)
    q2_params = q2_state.filter(nnx.Param)
    q_mix_params = q_mix_state.filter(nnx.Param)


    tau = ((step % config.sync_period) == 0)* config.tau

    target_q1_params = optax.incremental_update(q1_params, train_state.q1_state.target_params, tau)
    target_q2_params = optax.incremental_update(q2_params, train_state.q2_state.target_params, tau)
    target_q_mix_params = optax.incremental_update(q_mix_params, train_state.q_mix_state.target_params, tau)

    train_state = train_state._replace(step = step + 1, q1_state = train_state.q1_state._replace(target_params = target_q1_params, state = q1_state), q2_state = train_state.q2_state._replace(target_params = target_q2_params, state = q2_state), q_mix_state = train_state.q_mix_state._replace(target_params = target_q_mix_params, state = q_mix_state))
    
    return train_state, loss
        
def init_train_state(config):
    

    q_mixer = Qmixer(config.state_dim, config.embed_dim, config.n_agents, nnx.Rngs(0))
    q_mixer_optimizer = nnx.Optimizer(q_mixer, optax.adam(learning_rate = 3e-4))
    q1 = Qnetwork(config.state_dim, config.action_dim, layer_dim = config.layer_dim, rngs = nnx.Rngs(0))
    q1_optimizer = nnx.Optimizer(q1, optax.adam(learning_rate = 3e-4))
    q2 = Qnetwork(config.state_dim, config.action_dim, layer_dim = config.layer_dim, rngs = nnx.Rngs(1))
    q2_optimizer = nnx.Optimizer(q2, optax.adam(learning_rate = 3e-4))


    (q_mix_gd, q_mix_state) = nnx.split((q_mixer, q_mixer_optimizer))
    (q1_gd, q1_state) = nnx.split((q1, q1_optimizer))
    (q2_gd, q2_state) = nnx.split((q2, q2_optimizer))

    q1_target = q1_state.filter(nnx.Param)
    q2_target = q2_state.filter(nnx.Param)
    q_mix_target = q_mix_state.filter(nnx.Param)

    
    return TrainState(
    q1_state = NetworkState(q1_gd, q1_state, q1_target),
    q2_state = NetworkState(q2_gd, q2_state, q2_target),
    q_mix_state = NetworkState(q_mix_gd, q_mix_state, q_mix_target),
    step = jnp.array(0)
)