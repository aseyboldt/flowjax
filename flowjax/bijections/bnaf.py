from typing import Callable
import equinox as eqx
import jax
import jax.numpy as jnp
from flowjax.bijections.abc import Bijection
from jax import random
from jax.nn.initializers import glorot_uniform
from jax import lax


def b_diag_mask(block_shape: tuple, n_blocks: int):
    "Block diagonal mask."
    return jax.scipy.linalg.block_diag(
        *[jnp.ones(block_shape, int) for _ in range(n_blocks)]
    )


def b_tril_mask(block_shape: tuple, n_blocks: int):
    "Upper triangular block mask, excluding diagonal blocks."
    mask = jnp.zeros((block_shape[0] * n_blocks, block_shape[1] * n_blocks))

    for i in range(n_blocks):
        mask = mask.at[
            (i + 1) * block_shape[0] :, i * block_shape[1] : (i + 1) * block_shape[1]
        ].set(1)
    return mask


class BlockAutoregressiveLinear(eqx.Module):
    n_blocks: int
    block_shape: tuple
    W: jnp.ndarray
    bias: jnp.ndarray
    W_log_scale: jnp.ndarray
    in_features: int
    out_features: int
    _b_diag_mask: jnp.ndarray
    _b_diag_mask_idxs: jnp.ndarray
    _b_tril_mask: jnp.ndarray

    def __init__(
        self,
        key: random.PRNGKey,
        n_blocks: int,
        block_shape: tuple,
        init: Callable = glorot_uniform(),
    ):
        """Block autoregressive neural netork layer (https://arxiv.org/abs/1904.04676).

        Args:
            key (random.PRNGKey): Random key
            n_blocks (int): Number of diagonal blocks (dimension of input layer).
            block_shape (tuple): The shape of the blocks.
            init (Callable, optional): Default initialisation method for the weight matrix. Defaults to glorot_uniform().
        """
        self.block_shape = block_shape
        self.n_blocks = n_blocks

        self._b_diag_mask = b_diag_mask(block_shape, n_blocks)
        self._b_diag_mask_idxs = jnp.where(self._b_diag_mask)
        self._b_tril_mask = b_tril_mask(block_shape, n_blocks)

        in_features, out_features = (
            block_shape[1] * n_blocks,
            block_shape[0] * n_blocks,
        )

        *w_key, bias_key, scale_key = random.split(key, n_blocks + 2)

        self.W = init(w_key[0], (out_features, in_features)) * (
            self.b_tril_mask + self.b_diag_mask
        )
        self.bias = (random.uniform(bias_key, (out_features,)) - 0.5) * (
            2 / jnp.sqrt(out_features)
        )
        self.W_log_scale = jnp.log(random.uniform(scale_key, (out_features, 1)))
        self.in_features = in_features
        self.out_features = out_features

    def get_normalised_weights(self):
        "Carries out weight normalisation."
        W = jnp.exp(self.W) * self.b_diag_mask + self.W * self.b_tril_mask
        W_norms = jnp.linalg.norm(W, axis=-1, keepdims=True)
        return jnp.exp(self.W_log_scale) * W / W_norms

    def __call__(self, x):
        "returns output y, and components of weight matrix needed log_det component (n_blocks, block_shape[0], block_shape[1])"
        W = self.get_normalised_weights()
        y = W @ x + self.bias
        jac_3d = W[self.b_diag_mask_idxs].reshape(self.n_blocks, *self.block_shape)
        return y, jnp.log(jac_3d)

    @property
    def b_diag_mask(self):
        return jax.lax.stop_gradient(self._b_diag_mask)

    @property
    def b_diag_mask_idxs(self):
        return jax.lax.stop_gradient(self._b_diag_mask_idxs)

    @property
    def b_tril_mask(self):
        return jax.lax.stop_gradient(self._b_tril_mask)


def logmatmulexp(x, y):
    """
    Numerically stable version of ``(x.log() @ y.log()).exp()``. From numpyro https://github.com/pyro-ppl/numpyro/blob/f2ff89a3a7147617e185eb51148eb15d56d44661/numpyro/distributions/util.py#L387
    """
    x_shift = lax.stop_gradient(jnp.amax(x, -1, keepdims=True))
    y_shift = lax.stop_gradient(jnp.amax(y, -2, keepdims=True))
    xy = jnp.log(jnp.matmul(jnp.exp(x - x_shift), jnp.exp(y - y_shift)))
    return xy + x_shift + y_shift


class _TanhBNAF:
    """
    Tanh transformation compatible with BNAF (log_abs_det provided as 3D array).
    Condition is ignored. Output shape is (n_blocks, *block_size), where
    output[i] is the log jacobian for the iith block.
    """

    def __init__(self, n_blocks: int):
        self.n_blocks = n_blocks

    def __call__(self, x, condition=None):
        d = x.shape[0] // self.n_blocks
        log_det_vals = -2 * (x + jax.nn.softplus(-2 * x) - jnp.log(2.0))
        log_det = jnp.full((self.n_blocks, d, d), -jnp.inf)
        log_det = log_det.at[:, jnp.arange(d), jnp.arange(d)].set(
            log_det_vals.reshape(self.n_blocks, d)
        )
        return jnp.tanh(x), log_det


class BlockAutoregressiveNetwork(eqx.Module, Bijection):
    n_layers: int
    layers: list
    activation: Callable

    def __init__(
        self,
        key: random.PRNGKey,
        dim: int,
        n_layers: int = 3,
        block_size: tuple = (8, 8),
        activation=_TanhBNAF,
    ):

        self.n_layers = n_layers

        layers = []

        block_sizes = [
            (block_size[0], 1),
            *[block_size] * (n_layers - 2),
            (1, block_size[1]),
        ]
        for size in block_sizes:
            key, subkey = random.split(key)
            layers.extend(
                [BlockAutoregressiveLinear(subkey, dim, size), activation(dim)]
            )
        self.layers = layers[:-1]
        self.activation = activation

    def transform(self, x: jnp.ndarray, condition=None):
        y = x
        for layer in self.layers:
            y = layer(y)[0]
        return y

    def transform_and_log_abs_det_jacobian(self, x: jnp.ndarray, condition=None):
        y = x
        log_det_3ds = []

        for layer in self.layers:
            y, log_det_3d = layer(y)
            log_det_3ds.append(log_det_3d)

        logdet = log_det_3ds[-1]
        for ld in reversed(log_det_3ds[:-1]):
            logdet = logmatmulexp(logdet, ld)
        return y, logdet.sum()

    def inverse(*args, **kwargs):
        return NotImplementedError(
            """
        This transform would require numerical methods for inversion..
        """
        )
