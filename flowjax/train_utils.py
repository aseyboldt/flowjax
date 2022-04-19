from flowjax.flows import Flow
from jax import random
import jax.numpy as jnp
import equinox as eqx
import optax
from tqdm import tqdm
from typing import Optional


def train_flow(
    key: random.PRNGKey,
    flow: Flow,
    x: jnp.ndarray,
    condition: Optional[jnp.ndarray] = None,
    max_epochs: int = 50,
    max_patience: int = 5,
    learning_rate: float = 5e-4,
    batch_size: int = 256,
    val_prop: float = 0.1,
    show_progress: bool = True,
):
    def loss(flow, x, condition=None):
        return -flow.log_prob(x, condition).mean()

    @eqx.filter_jit
    def step(flow, optimizer, opt_state, x, condition=None):
        loss_val, grads = eqx.filter_value_and_grad(loss)(flow, x, condition)
        updates, opt_state = optimizer.update(grads, opt_state)
        flow = eqx.apply_updates(flow, updates)
        return flow, opt_state, loss_val

    key, subkey = random.split(key)

    inputs = (x,) if condition is None else (x, condition)
    train_args, val_args = train_val_split(subkey, inputs, val_prop=val_prop)

    optimizer = optax.adam(learning_rate=learning_rate)
    best_params, static = eqx.partition(flow, eqx.is_array)

    opt_state = optimizer.init(best_params)
    losses = []

    losses = {"train": [], "val": []}

    loop = tqdm(range(max_epochs)) if show_progress is True else range(max_epochs)
    for epoch in loop:
        key, subkey = random.split(key)
        train_args = random_permutation_multiple(subkey, train_args)
        batches = range(0, train_args[0].shape[0] - batch_size, batch_size)

        epoch_train_loss = 0
        for i in batches:
            batch = tuple(a[i : i + batch_size] for a in train_args)

            flow, opt_state, loss_val = step(flow, optimizer, opt_state, *batch)
            epoch_train_loss += loss_val.item()

        val_loss = loss(flow, *val_args).item()
        losses["train"].append(epoch_train_loss / len(batches))
        losses["val"].append(val_loss)

        if val_loss == min(losses["val"]):
            best_params, _ = eqx.partition(flow, eqx.is_array)

        elif count_fruitless(losses["val"]) > max_patience:
            print("Max patience reached.")
            break

        if show_progress:
            loop.set_postfix({k: v[-1] for k, v in losses.items()})

    flow = eqx.combine(best_params, static)
    return flow, losses


def train_val_split(key: random.PRNGKey, arrays, val_prop: float = 0.1):
    "Returns ((train_x, train_y), (val_x, val_y), ...)). Split on axis 0."
    assert 0 <= val_prop <= 1
    key, subkey = random.split(key)
    arrays = random_permutation_multiple(subkey, arrays)
    n_val = round(val_prop * arrays[0].shape[0])
    train = tuple(a[:-n_val] for a in arrays)
    val = tuple(a[-n_val:] for a in arrays)
    return train, val


def random_permutation_multiple(key, arrays):
    "Randomly permute multiple arrays on axis 0 (consistent between arrays)."
    n = arrays[0].shape[0]
    shuffle = random.permutation(key, jnp.arange(n))
    arrays = tuple(a[shuffle] for a in arrays)
    return arrays


def count_fruitless(losses: list):
    """Given a list of losses from each epoch, count the number of epochs since
    the minimum loss"""
    min_idx = jnp.array(losses).argmin().item()
    return len(losses) - min_idx - 1
