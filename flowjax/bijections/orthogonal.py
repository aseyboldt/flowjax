from flowjax.bijections.bijection import AbstractBijection
from jax import Array
import jax.numpy as jnp
import jax.nn as jnn
from jax.scipy import fft
from paramax.utils import inv_softplus
from jax.nn import softplus
from paramax import AbstractUnwrappable, Parameterize

class Neg(AbstractBijection):
    """A bijection that negates its input (multiplies by -1).

    This is a simple bijection that flips the sign of all elements in the input array.

    Attributes:
        shape: Shape of the input/output arrays
        cond_shape: Shape of conditional inputs (None as this bijection is unconditional)
    """
    shape: tuple[int, ...]
    cond_shape = None

    def __init__(self, shape):
        self.shape = shape

    def transform_and_log_det(self, x: jnp.ndarray, condition: Array | None = None):
        return -x, jnp.zeros(())

    def inverse_and_log_det(self, y: Array, condition: Array | None = None):
        return -y, jnp.zeros(())

class MvScale(AbstractBijection):
    shape: tuple[int, ...]
    params: Array
    cond_shape = None

    def __init__(self, params: Array):
        self.shape = (params.shape[-1],)
        self.params = params


    def _exp_map_sphere(self, v):
        """Riemannian exponential map on the n-sphere S^n

        Compute the point on the sphere reached by the exponential
        map from point p with tangent vector v (shape: (n+1,))

        Parameters:
        p: Point on the sphere (shape: (n+1,))
        v: Tangent vector in R^n (shape: (n,))
        """
        # Project v into the correct tangent space using Householder transformation
        #v_proj = householder_transform(p, v)
        #p = jnp.ones_like(v)
        #p = p / jnp.linalg.norm(p)
        p = jnp.zeros_like(v).at[self.base_index].set(1.0)
        v_proj = v - (v @ p) * p

        norm_v_raw = jnp.linalg.norm(v_proj)
        # Avoid NaNs in the gradient
        norm_v = jnp.where(norm_v_raw > 1e-6, norm_v_raw, 1.0)

        direction = v_proj / norm_v

        # General case: Compute the exponential map
        exp_general = jnp.cos(norm_v) * p + jnp.sin(norm_v) * direction

        # Small v case: Use a Taylor expansion and re-normalize to stay on the sphere
        exp_taylor = p + v_proj
        norm_exp_taylor_raw = jnp.linalg.norm(exp_taylor)
        # Prevent nans in the gradient
        norm_exp_taylor = jnp.where(norm_exp_taylor_raw > 1e-6, norm_exp_taylor_raw, 1.0)
        exp_taylor = exp_taylor / norm_exp_taylor
        return jnp.where(norm_v_raw > 1e-6, exp_general, exp_taylor)

    def transform_and_log_det(self, x: jnp.ndarray, condition: Array | None = None):
        p = self.params
        norm = jnp.linalg.norm(p)
        v = p / norm

        y = x + ((v @ x) * (norm - 1)) * v
        return y, jnp.log(norm)

    def inverse_and_log_det(self, y: Array, condition: Array | None = None):
        p = self.params
        norm = jnp.linalg.norm(p)
        v = p / norm
        x = y + ((v @ y) * (1 / norm - 1)) * v
        return x, -jnp.log(norm)


class Householder(AbstractBijection):
    """A Householder reflection bijection.

    This bijection implements a Householder reflection, which is a linear
    transformation that reflects vectors across a hyperplane defined by a normal
    vector (params). The transformation is its own inverse and volume-preserving
    (determinant = ±1).

    Given a unit vector v, the transformation is:
    x → x - 2(x·v)v

    Attributes:
        shape: Shape of the input/output vectors
        cond_shape: Shape of conditional inputs (None as this bijection is unconditional)
        params: Normal vector defining the reflection hyperplane. The vector is
            normalized in the transformation, so scaling params will have no effect
            on the bijection.
    """
    shape: tuple[int, ...]
    params: Array
    cond_shape = None
    base_index: int

    def __init__(self, params: Array, base_index: int = 0):
        self.shape = (params.shape[-1],)
        self.params = params
        self.base_index = base_index

    def _householder(self, x: Array, params: Array) -> Array:
        def exp_map_sphere(p, v):
            """Riemannian exponential map on the n-sphere S^n

            Compute the point on the sphere reached by the exponential
            map from point p with tangent vector v (shape: (n+1,))

            Parameters:
            p: Point on the sphere (shape: (n+1,))
            v: Tangent vector in R^n (shape: (n,))
            """
            # Project v into the correct tangent space using Householder transformation
            #v_proj = householder_transform(p, v)
            #p = jnp.ones_like(v)
            #p = p / jnp.linalg.norm(p)
            p = jnp.zeros_like(v).at[self.base_index].set(1.0)
            v_proj = v - (v @ p) * p

            norm_v_raw = jnp.linalg.norm(v_proj)
            # Avoid NaNs in the gradient
            norm_v = jnp.where(norm_v_raw > 1e-6, norm_v_raw, 1.0)

            direction = v_proj / norm_v

            # General case: Compute the exponential map
            exp_general = jnp.cos(norm_v) * p + jnp.sin(norm_v) * direction

            # Small v case: Use a Taylor expansion and re-normalize to stay on the sphere
            exp_taylor = p + v_proj
            norm_exp_taylor_raw = jnp.linalg.norm(exp_taylor)
            # Prevent nans in the gradient
            norm_exp_taylor = jnp.where(norm_exp_taylor_raw > 1e-6, norm_exp_taylor_raw, 1.0)
            exp_taylor = exp_taylor / norm_exp_taylor
            return jnp.where(norm_v_raw > 1e-6, exp_general, exp_taylor)

        #norm_sq = params @ params
        #norm = jnp.sqrt(norm_sq)
        #vec = params / norm

        if self.shape == (1,):
            return -x

        vec = exp_map_sphere(jnp.zeros_like(params), params)

        return x - 2 * vec * (x @ vec)

    def transform_and_log_det(self, x: jnp.ndarray, condition: Array | None = None):
        return self._householder(x, self.params), jnp.zeros(())

    def inverse_and_log_det(self, y: Array, condition: Array | None = None):
        return self._householder(y, self.params), jnp.zeros(())

    def inverse_gradient_and_val(
        self,
        y: Array,
        y_grad: Array,
        y_logp: Array,
        condition: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        x, logdet = self.inverse_and_log_det(y)
        x_grad = self._householder(y_grad, params=self.params)
        return (x, x_grad, y_logp - logdet)


class DCT(AbstractBijection):
    """Discrete Cosine Transform (DCT) bijection.

    This bijection applies the DCT or its inverse along a specified axis.

    Attributes:
        shape: Shape of the input/output arrays
        cond_shape: Shape of conditional inputs (None as this bijection is unconditional)
        axis: Axis along which to apply the DCT
        norm: Normalization method, fixed to 'ortho' to ensure bijectivity
    """

    shape: tuple[int, ...]
    cond_shape = None
    axis: int
    norm: str

    def __init__(self, shape, *, axis: int = -1):
        self.shape = shape
        self.axis = axis
        self.norm = "ortho"

    def _dct(self, x: Array, inverse: bool = False) -> Array:
        if inverse:
            z = fft.idct(x, norm=self.norm, axis=self.axis)
        else:
            z = fft.dct(x, norm=self.norm, axis=self.axis)

        return z

    def transform_and_log_det(self, x: jnp.ndarray, condition: Array | None = None):
        y = self._dct(x)
        return y, jnp.zeros(())

    def inverse_and_log_det(self, y: Array, condition: Array | None = None):
        x = self._dct(y, inverse=True)
        return x, jnp.zeros(())
