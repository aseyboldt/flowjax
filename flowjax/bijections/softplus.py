"""SoftPlus bijection."""

from typing import ClassVar

import jax
import jax.numpy as jnp
from jax.nn import softplus
from jax import Array

from flowjax.bijections.bijection import AbstractBijection


class SoftPlus(AbstractBijection):
    r"""Transforms to positive domain using softplus :math:`y = \log(1 + \exp(x))`."""

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def transform_and_log_det(self, x, condition=None):
        return softplus(x), -softplus(-x).sum()

    def inverse_and_log_det(self, y, condition=None):
        x = jnp.log(-jnp.expm1(-y)) + y
        return x, softplus(-x).sum()


class SoftPlusX(AbstractBijection):
    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def transform_and_log_det(self, x, condition=None):
        val = x - 0.5 * jax.nn.softplus(x)
        logdet = -jnp.log(2) + jnp.log1p(jax.nn.sigmoid(-x))
        return val, logdet

    def _inverse_and_log_det(self, y, condition=None):
        y_min = -20.0
        y_max = 20.0

        term = jnp.sqrt(1 + 4 * jnp.exp(-2*y))
        val = 2 * y + jnp.log1p(term) - jnp.log(2)
        logdet = -jnp.log1p(-0.5 * jax.nn.sigmoid(val))

        val = jnp.where(y < y_min, y, val)
        val = jnp.where(y > y_max, 2 * y, val)
        logdet = jnp.where(y < y_min, 0.0, logdet)
        logdet = jnp.where(y < y_max, 0.0, -jnp.log1p(-0.5 * jax.nn.sigmoid(2 * y)))

        return val, logdet

    def inverse_and_log_det(self, y, condition=None):
        y_min = -20.0
        y_max = 20.0

        # Compute values for y < y_min
        val_low = y
        logdet_low = 0.0

        # Compute values for y > y_max
        val_high = 2 * y
        sigmoid_val_high = jax.nn.sigmoid(2 * y)
        logdet_high = -jnp.log1p(-0.5 * sigmoid_val_high)

        # Compute values for y within [y_min, y_max]
        y_ = jnp.clip(y, y_min, y_max)
        exp_neg_2y = jnp.exp(-2 * y_)
        term = jnp.sqrt(1.0 + 4.0 * exp_neg_2y)
        val_mid = 2 * y_ + jnp.log1p(term) - jnp.log(2.0)
        sigmoid_val_mid = jax.nn.sigmoid(val_mid)
        logdet_mid = -jnp.log1p(-0.5 * sigmoid_val_mid)

        # Use jnp.where to select appropriate values
        val = jnp.where(
            y < y_min, val_low, jnp.where(y > y_max, val_high, val_mid)
        )
        logdet = jnp.where(
            y < y_min, logdet_low, jnp.where(y > y_max, logdet_high, logdet_mid)
        )

        return val, logdet

    def inverse_gradient_and_val(
        self,
        y: Array,
        y_grad: Array,
        y_logp: Array,
        condition: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        x, _ = self.inverse_and_log_det(y, condition=condition)

        # Step 2: Compute derivatives
        s_x = jax.nn.sigmoid(x)
        s_neg_x = jax.nn.sigmoid(-x)
        dval_dx = 1 - 0.5 * s_x
        numerator = -s_neg_x * s_x
        denominator = 1 + s_neg_x
        dlogdet_dx = numerator / denominator

        # Step 3: Compute x_grad
        x_grad = y_grad * dval_dx + dlogdet_dx

        # Step 4: Compute y_logp + fwd_log_det
        logdet = -jnp.log(2) + jnp.log1p(s_neg_x)
        total_logp = y_logp + logdet

        return x, x_grad, total_logp

