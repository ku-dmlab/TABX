
import jax
import jax.numpy as jnp
from collections import namedtuple
from functools import partial
dataset_state = namedtuple('dataset_state', ['data', 'length', 'key', 'max_length'])

def make_buffer(data, max_length, key):
    data =jax.tree_util.tree_map(lambda value: jnp.repeat(value, max_length).reshape((max_length,) + value.shape), data)
    return dataset_state(data, 0, key, max_length)


def add(dataset_state, target_data, length):
    data = dataset_state.data
    data = jax.tree_util.tree_map(lambda value: jnp.roll(value, axis=0, shift=length), data)
    def update(value, target_value):
        target_value = target_value.reshape((length,) + value.shape[1:])
        value = value.at[:length]
        return value.set(target_value)
    data = jax.tree_util.tree_map(update, data, target_data)
    updated_length = dataset_state.length + length
    updated_length = jnp.clip(updated_length, 0, dataset_state.max_length)
    return dataset_state._replace(data=data, length=updated_length)

def sample(dataset_state, batch_size):
    key, subkey = jax.random.split(dataset_state.key)
    data = dataset_state.data
    indices = jax.random.randint(subkey, shape=(batch_size,), minval=0, maxval=dataset_state.length)
    return dataset_state._replace(key=key), jax.tree_util.tree_map(lambda value: value[indices], data)