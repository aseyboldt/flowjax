from flowjax.bijections.bijection import AbstractBijection
from jax import Array
import jax.numpy as jnp
import jax.nn as jnn
from jax.scipy import fft


class Rescale(AbstractBijection):
    """A bijection that scales the target space in one direction.

    Attributes:
        shape: The shape of the bijection, defined by the size of params[1:].
        params: Array containing scaling and direction parameters.
    """

    shape: tuple[int, ...]
    cond_shape = None

    def __init__(self, vals):
        """Initialize the MvScale bijection with `params`."""
        super().__init__()
        self.shape = vals.shape

    def transform(self, x: jnp.ndarray, condition: Array | None = None) -> Array:
        """Apply the transformation (scaling in a specific direction) on the input vector.

        Args:
            x: Input vector to transform.
            condition: Not used, present for API compatibility.

        Returns:
            The transformed vector.
        """
        return self.transform_and_log_det(x, condition=condition)[0]

    def transform_and_log_det(self, x: jnp.ndarray, condition: Array | None = None):
        """Apply the transformation and return the log-determinant of the Jacobian.

        The log-determinant is simply the logarithm of the scaling factor `params[0]`.

        Args:
            x: Input vector.
            condition: Not used, present for API compatibility.

        Returns:
            A tuple of (transformed vector, log-determinant).
        """
        (n,) = x.shape
        var = (x @ x) / n
        new_var = jnp.sqrt(var)
        scale = jnp.sqrt(new_var / var)
        y = scale * x
        logdet = n / 4 * (jnp.log(n) - jnp.log(x @ x)) - jnp.log(2)
        return y, logdet

    def inverse(self, y: Array, condition: Array | None = None) -> Array:
        """Inverse the transformation by inverting the scaling factor.

        Args:
            y: Transformed vector.
            condition: Not used, present for API compatibility.

        Returns:
            The inverse transformed vector.
        """
        return self.inverse_and_log_det(y, condition=condition)[0]

    def inverse_and_log_det(self, y: Array, condition: Array | None = None):
        """Invert the transformation and return the log-determinant of the Jacobian.

        Args:
            y: Transformed vector.
            condition: Not used, present for API compatibility.

        Returns:
            A tuple of (inverse transformed vector, negative log-determinant).
        """
        (n,) = y.shape
        var = (y @ y) / n
        new_var = var**2
        scale = jnp.sqrt(new_var / var)
        x = scale * y
        inv_logdet = n / 4 * (jnp.log(n) - jnp.log(x @ x)) - jnp.log(2)
        return x, -inv_logdet


class MvScale3(AbstractBijection):
    """A bijection that scales the target space in one direction.

    The scaling factor is `exp(params[0])`, while the direction is determined
    by `params[1:]`.

    Attributes:
        shape: The shape of the bijection, defined by the size of params[1:].
        params: Array containing scaling and direction parameters.
    """

    shape: tuple[int, ...]
    scale: Array
    cond_shape = None

    def __init__(self, scale: Array):
        """Initialize the MvScale bijection with `params`."""
        super().__init__()
        self.shape = (scale.shape[-1],)
        self.scale = scale

    @staticmethod
    def householder_reflection(x: Array) -> Array:
        """Compute the Householder reflection that maps (1, 0, 0, ...) to (1, 1, 1, ...)/sqrt(n).

        This reflection is useful for changing the direction of the vector to be aligned
        with the uniform direction. It avoids storing the full reflection matrix, and
        instead directly computes the result using vector operations.

        Args:
            x: Input vector to apply the Householder reflection.

        Returns:
            Reflected vector based on the Householder transformation.
        """
        n = x.shape[0]

        # Define the target vector: (1/sqrt(n), 1/sqrt(n), ..., 1/sqrt(n))
        target = jnp.ones(n) / jnp.sqrt(n)

        # Define the initial vector a = e1 = (1, 0, 0, ...)
        e1 = jnp.zeros_like(x)
        e1 = e1.at[0].set(1.0)

        # Compute lambda = ±||a||, in our case lambda = ±1 since ||e1|| = 1
        lambda_ = 1

        # Compute the Householder vector v = (a - lambda * target) / ||a - lambda * target||
        v = e1 - lambda_ * target
        v = -v / jnp.linalg.norm(v)  # Normalizing the vector

        # Compute the Householder reflection: Hx = x - 2 * v * (v.T @ x)
        v_dot_x = v @ x
        return x - 2.0 * v * v_dot_x

    def scale_vals(self, x: Array, params: Array) -> Array:
        """Apply the scaling operation in the direction determined by the parameters.

        Args:
            x: Input vector to scale.
            params: Array where params[0] controls the scale, and params[1:] control the direction.

        Returns:
            The scaled vector.
        """
        vec = self.scale

        # Apply the scaling in the direction of vec
        # return vec * ((vec @ x) * (eig - 1)) + x
        return x - 2 * vec * (vec @ x)

    def transform(self, x: jnp.ndarray, condition: Array | None = None) -> Array:
        """Apply the transformation (scaling in a specific direction) on the input vector.

        Args:
            x: Input vector to transform.
            condition: Not used, present for API compatibility.

        Returns:
            The transformed vector.
        """
        return self.scale_vals(x, self.scale)

    def transform_and_log_det(self, x: jnp.ndarray, condition: Array | None = None):
        """Apply the transformation and return the log-determinant of the Jacobian.

        The log-determinant is simply the logarithm of the scaling factor `params[0]`.

        Args:
            x: Input vector.
            condition: Not used, present for API compatibility.

        Returns:
            A tuple of (transformed vector, log-determinant).
        """
        return self.transform(x), jnp.log(jnp.abs(2 * self.scale @ self.scale - 1))

    def inverse(self, y: Array, condition: Array | None = None) -> Array:
        """Inverse the transformation by inverting the scaling factor.

        Args:
            y: Transformed vector.
            condition: Not used, present for API compatibility.

        Returns:
            The inverse transformed vector.
        """
        return self.scale_vals(y, self.scale)

    def inverse_and_log_det(self, y: Array, condition: Array | None = None):
        """Invert the transformation and return the log-determinant of the Jacobian.

        Args:
            y: Transformed vector.
            condition: Not used, present for API compatibility.

        Returns:
            A tuple of (inverse transformed vector, negative log-determinant).
        """
        return self.inverse(y), -jnp.log(jnp.abs(2 * self.scale @ self.scale - 1))


class MvScale2(AbstractBijection):
    """A bijection that scales the target space in one direction.

    The scaling factor is `exp(params[0])`, while the direction is determined
    by `params[1:]`.

    Attributes:
        shape: The shape of the bijection, defined by the size of params[1:].
        params: Array containing scaling and direction parameters.
    """

    shape: tuple[int, ...]
    scale: Array
    cond_shape = None

    def __init__(self, scale: Array):
        """Initialize the MvScale bijection with `params`."""
        super().__init__()
        self.shape = (scale.shape[-1],)
        self.scale = scale

    @staticmethod
    def householder_reflection(x: Array) -> Array:
        """Compute the Householder reflection that maps (1, 0, 0, ...) to (1, 1, 1, ...)/sqrt(n).

        This reflection is useful for changing the direction of the vector to be aligned
        with the uniform direction. It avoids storing the full reflection matrix, and
        instead directly computes the result using vector operations.

        Args:
            x: Input vector to apply the Householder reflection.

        Returns:
            Reflected vector based on the Householder transformation.
        """
        n = x.shape[0]

        # Define the target vector: (1/sqrt(n), 1/sqrt(n), ..., 1/sqrt(n))
        target = jnp.ones(n) / jnp.sqrt(n)

        # Define the initial vector a = e1 = (1, 0, 0, ...)
        e1 = jnp.zeros_like(x)
        e1 = e1.at[0].set(1.0)

        # Compute lambda = ±||a||, in our case lambda = ±1 since ||e1|| = 1
        lambda_ = 1

        # Compute the Householder vector v = (a - lambda * target) / ||a - lambda * target||
        v = e1 - lambda_ * target
        v = -v / jnp.linalg.norm(v)  # Normalizing the vector

        # Compute the Householder reflection: Hx = x - 2 * v * (v.T @ x)
        v_dot_x = v @ x
        return x - 2.0 * v * v_dot_x

    def scale_vals(self, x: Array, params: Array) -> Array:
        """Apply the scaling operation in the direction determined by the parameters.

        Args:
            x: Input vector to scale.
            params: Array where params[0] controls the scale, and params[1:] control the direction.

        Returns:
            The scaled vector.
        """
        eig = jnp.exp(params[0])  # Scaling factor controlled by params[0]
        vec_params = params.at[0].set(0.0)  # Set the first param to 0 for direction

        # Apply the Householder reflection to the direction parameters
        transformed_params = self.householder_reflection(vec_params)

        # Apply softmax to the reflected parameters to get the Householder vector
        vec = jnp.sqrt(jnn.softmax(transformed_params))

        # Ensure numerical stability by normalizing the vector
        vec = vec / jnp.linalg.norm(vec)

        # Apply the scaling in the direction of vec
        # return vec * ((vec @ x) * (eig - 1)) + x
        return x - 2 * vec * (vec @ x)

    def transform(self, x: jnp.ndarray, condition: Array | None = None) -> Array:
        """Apply the transformation (scaling in a specific direction) on the input vector.

        Args:
            x: Input vector to transform.
            condition: Not used, present for API compatibility.

        Returns:
            The transformed vector.
        """
        return self.scale_vals(x, self.scale)

    def transform_and_log_det(self, x: jnp.ndarray, condition: Array | None = None):
        """Apply the transformation and return the log-determinant of the Jacobian.

        The log-determinant is simply the logarithm of the scaling factor `params[0]`.

        Args:
            x: Input vector.
            condition: Not used, present for API compatibility.

        Returns:
            A tuple of (transformed vector, log-determinant).
        """
        return self.transform(x), jnp.array(0.0)

    def inverse(self, y: Array, condition: Array | None = None) -> Array:
        """Inverse the transformation by inverting the scaling factor.

        Args:
            y: Transformed vector.
            condition: Not used, present for API compatibility.

        Returns:
            The inverse transformed vector.
        """
        return self.scale_vals(y, self.scale)

    def inverse_and_log_det(self, y: Array, condition: Array | None = None):
        """Invert the transformation and return the log-determinant of the Jacobian.

        Args:
            y: Transformed vector.
            condition: Not used, present for API compatibility.

        Returns:
            A tuple of (inverse transformed vector, negative log-determinant).
        """
        return self.inverse(y), jnp.array(0.0)


class MvScale(AbstractBijection):
    """A bijection that scales the target space in one direction.

    The scaling factor is `exp(params[0])`, while the direction is determined
    by `params[1:]`.

    Attributes:
        shape: The shape of the bijection, defined by the size of params[1:].
        params: Array containing scaling and direction parameters.
    """

    shape: tuple[int, ...]
    scale: Array
    bias: Array
    cond_shape = None

    def __init__(self, scale: Array, bias: Array):
        """Initialize the MvScale bijection with `params`."""
        super().__init__()
        self.shape = (scale.shape[-1],)
        self.scale = scale
        self.bias = bias

    @staticmethod
    def householder_reflection(x: Array) -> Array:
        """Compute the Householder reflection that maps (1, 0, 0, ...) to (1, 1, 1, ...)/sqrt(n).

        This reflection is useful for changing the direction of the vector to be aligned
        with the uniform direction. It avoids storing the full reflection matrix, and
        instead directly computes the result using vector operations.

        Args:
            x: Input vector to apply the Householder reflection.

        Returns:
            Reflected vector based on the Householder transformation.
        """
        n = x.shape[0]

        # Define the target vector: (1/sqrt(n), 1/sqrt(n), ..., 1/sqrt(n))
        target = jnp.ones(n) / jnp.sqrt(n)

        # Define the initial vector a = e1 = (1, 0, 0, ...)
        e1 = jnp.zeros_like(x)
        e1 = e1.at[0].set(1.0)

        # Compute lambda = ±||a||, in our case lambda = ±1 since ||e1|| = 1
        lambda_ = jnp.sign(e1[0]) * jnp.linalg.norm(e1)

        # Compute the Householder vector v = (a - lambda * target) / ||a - lambda * target||
        v = e1 - lambda_ * target
        v = -v / jnp.linalg.norm(v)  # Normalizing the vector

        # Compute the Householder reflection: Hx = x - 2 * v * (v.T @ x)
        v_dot_x = v @ x
        return x - 2.0 * v * v_dot_x

    def scale_vals(self, x: Array, params: Array) -> Array:
        """Apply the scaling operation in the direction determined by the parameters.

        Args:
            x: Input vector to scale.
            params: Array where params[0] controls the scale, and params[1:] control the direction.

        Returns:
            The scaled vector.
        """
        eig = jnp.exp(params[0])  # Scaling factor controlled by params[0]
        vec_params = params.at[0].set(0.0)  # Set the first param to 0 for direction

        # Apply the Householder reflection to the direction parameters
        transformed_params = self.householder_reflection(vec_params)

        # Apply softmax to the reflected parameters to get the Householder vector
        vec = jnp.sqrt(jnn.softmax(transformed_params))

        # Ensure numerical stability by normalizing the vector
        vec = vec / jnp.linalg.norm(vec)

        # Apply the scaling in the direction of vec
        return vec * ((vec @ x) * (eig - 1)) + x

    def transform(self, x: jnp.ndarray, condition: Array | None = None) -> Array:
        """Apply the transformation (scaling in a specific direction) on the input vector.

        Args:
            x: Input vector to transform.
            condition: Not used, present for API compatibility.

        Returns:
            The transformed vector.
        """
        return self.scale_vals(x, self.scale) + self.bias

    def transform_and_log_det(self, x: jnp.ndarray, condition: Array | None = None):
        """Apply the transformation and return the log-determinant of the Jacobian.

        The log-determinant is simply the logarithm of the scaling factor `params[0]`.

        Args:
            x: Input vector.
            condition: Not used, present for API compatibility.

        Returns:
            A tuple of (transformed vector, log-determinant).
        """
        return self.transform(x), self.scale[0]

    def inverse(self, y: Array, condition: Array | None = None) -> Array:
        """Inverse the transformation by inverting the scaling factor.

        Args:
            y: Transformed vector.
            condition: Not used, present for API compatibility.

        Returns:
            The inverse transformed vector.
        """
        return self.scale_vals(y - self.bias, self.scale.at[0].set(-self.scale[0]))

    def inverse_and_log_det(self, y: Array, condition: Array | None = None):
        """Invert the transformation and return the log-determinant of the Jacobian.

        Args:
            y: Transformed vector.
            condition: Not used, present for API compatibility.

        Returns:
            A tuple of (inverse transformed vector, negative log-determinant).
        """
        return self.inverse(y), -self.scale[0]


class MvScale4(AbstractBijection):
    """A bijection that scales the target space in one direction.

    The scaling factor is `exp(params[0])`, while the direction is determined
    by `params[1:]`.

    Attributes:
        shape: The shape of the bijection, defined by the size of params[1:].
        params: Array containing scaling and direction parameters.
    """

    shape: tuple[int, ...]
    params: Array
    cond_shape = None

    def __init__(self, params: Array):
        """Initialize the MvScale bijection with `params`."""
        super().__init__()
        self.shape = (params.shape[-1],)
        self.params = params

    def scale_vals(self, x: Array, params: Array, *, inverse: bool = False) -> Array:
        """Apply the scaling operation in the direction determined by the parameters.

        Args:
            x: Input vector to scale.
            params: Array where params[0] controls the scale, and params[1:] control the direction.

        Returns:
            The scaled vector.
        """
        norm_sq = params @ params
        norm = jnp.sqrt(norm_sq)

        if inverse:
            log_lamda = -norm_sq
        else:
            log_lamda = norm_sq

        lamda = jnp.exp(log_lamda)

        vec = params / norm
        # return x + (x @ vec) * (lamda - 1) * vec, log_lamda
        return x - 2 * vec * (x @ vec), jnp.zeros(())

    def transform(self, x: jnp.ndarray, condition: Array | None = None) -> Array:
        """Apply the transformation (scaling in a specific direction) on the input vector.

        Args:
            x: Input vector to transform.
            condition: Not used, present for API compatibility.

        Returns:
            The transformed vector.
        """
        return self.scale_vals(x, self.params, inverse=False)[0]

    def transform_and_log_det(self, x: jnp.ndarray, condition: Array | None = None):
        """Apply the transformation and return the log-determinant of the Jacobian.

        The log-determinant is simply the logarithm of the scaling factor `params[0]`.

        Args:
            x: Input vector.
            condition: Not used, present for API compatibility.

        Returns:
            A tuple of (transformed vector, log-determinant).
        """
        return self.scale_vals(x, self.params, inverse=False)

    def inverse(self, y: Array, condition: Array | None = None) -> Array:
        """Inverse the transformation by inverting the scaling factor.

        Args:
            y: Transformed vector.
            condition: Not used, present for API compatibility.

        Returns:
            The inverse transformed vector.
        """
        return self.scale_vals(y, self.params, inverse=True)[0]

    def inverse_and_log_det(self, y: Array, condition: Array | None = None):
        """Invert the transformation and return the log-determinant of the Jacobian.

        Args:
            y: Transformed vector.
            condition: Not used, present for API compatibility.

        Returns:
            A tuple of (inverse transformed vector, negative log-determinant).
        """
        return self.scale_vals(y, self.params, inverse=True)


class MvScaleDCT(AbstractBijection):
    """A bijection that scales the target space in one direction.

    The scaling factor is `exp(params[0])`, while the direction is determined
    by `params[1:]`.

    Attributes:
        shape: The shape of the bijection, defined by the size of params[1:].
        params: Array containing scaling and direction parameters.
    """

    shape: tuple[int, ...]
    trafo: AbstractBijection
    cond_shape = None

    def __init__(self, affine):
        """Initialize the MvScale bijection with `params`."""
        super().__init__()
        self.shape = affine.shape
        self.trafo = affine

    def dct(self, x: Array, inverse: bool = False) -> Array:
        """Apply the scaling operation in the direction determined by the parameters.

        Args:
            x: Input vector to scale.
            params: Array where params[0] controls the scale, and params[1:] control the direction.

        Returns:
            The scaled vector.
        """
        if inverse:
            z = fft.idct(x, norm="ortho")
        else:
            z = fft.dct(x, norm="ortho")

        return z

    def transform(self, x: jnp.ndarray, condition: Array | None = None) -> Array:
        """Apply the transformation (scaling in a specific direction) on the input vector.

        Args:
            x: Input vector to transform.
            condition: Not used, present for API compatibility.

        Returns:
            The transformed vector.
        """

        z = self.dct(x)
        z, logdet = self.trafo.transform_and_log_det(z)
        y = self.dct(z, inverse=True)

        return y

    def transform_and_log_det(self, x: jnp.ndarray, condition: Array | None = None):
        """Apply the transformation and return the log-determinant of the Jacobian.

        The log-determinant is simply the logarithm of the scaling factor `params[0]`.

        Args:
            x: Input vector.
            condition: Not used, present for API compatibility.

        Returns:
            A tuple of (transformed vector, log-determinant).
        """
        z = self.dct(x)
        z, logdet = self.trafo.transform_and_log_det(z)
        y = self.dct(z, inverse=True)

        return y, logdet

    def inverse(self, y: Array, condition: Array | None = None) -> Array:
        """Inverse the transformation by inverting the scaling factor.

        Args:
            y: Transformed vector.
            condition: Not used, present for API compatibility.

        Returns:
            The inverse transformed vector.
        """
        z = self.dct(y)
        z, logdet = self.trafo.inverse_and_log_det(z)
        x = self.dct(z, inverse=True)

        return x

    def inverse_and_log_det(self, y: Array, condition: Array | None = None):
        """Invert the transformation and return the log-determinant of the Jacobian.

        Args:
            y: Transformed vector.
            condition: Not used, present for API compatibility.

        Returns:
            A tuple of (inverse transformed vector, negative log-determinant).
        """
        z = self.dct(y)
        z, logdet = self.trafo.inverse_and_log_det(z)
        x = self.dct(z, inverse=True)

        return x, logdet
