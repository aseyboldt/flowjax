"""Distributions, including the abstract and concrete classes."""

import inspect
from abc import abstractmethod
from functools import wraps
from math import prod
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from equinox import AbstractVar
from jax import Array
from jax.lax import stop_gradient
from jax.numpy import linalg
from jax.scipy import stats as jstats

from flowjax._custom_types import ArrayLike
from flowjax.bijections import (
    AbstractBijection,
    Affine,
    Chain,
    Exp,
    Scale,
    TriangularAffine,
)
from flowjax.utils import _get_ufunc_signature, arraylike_to_array, merge_cond_shapes
from flowjax.wrappers import unwrap


class AbstractDistribution(eqx.Module):
    """Abstract distribution class.

    Distributions are registered as jax PyTrees (as they are equinox modules), and as
    such they are compatible with normal jax operations.

    Concrete subclasses can be implemented as follows:
        (1) Inherit from :class:`AbstractDistribution`.
        (2) Define the abstract attributes ``shape`` and ``cond_shape``.
            ``cond_shape`` should be ``None`` for unconditional distributions.
        (3) Define the abstract methods `_sample` and `_log_prob`.

    See the source code for :class:`StandardNormal` for a simple concrete example.

    Attributes:
        shape: Tuple denoting the shape of a single sample from the distribution.
        cond_shape: Tuple denoting the shape of an instance of the conditioning
            variable. This should be None for unconditional distributions.

    """

    shape: AbstractVar[tuple[int, ...]]
    cond_shape: AbstractVar[tuple[int, ...] | None]

    @abstractmethod
    def _log_prob(self, x: Array, condition: Array | None = None) -> Array:
        """Evaluate the log probability of point x.

        This method should be be valid for inputs with shapes matching
        ``distribution.shape`` and ``distribution.cond_shape`` for conditional
        distributions (i.e. the method defined for unbatched inputs).
        """

    @abstractmethod
    def _sample(self, key: Array, condition: Array | None = None) -> Array:
        """Sample a point from the distribution.

        This method should return a single sample with shape matching
        ``distribution.shape``.
        """

    def _sample_and_log_prob(self, key, condition=None):
        """Sample a point from the distribution, and return its log probability."""
        x = self._sample(key, condition)
        return x, self._log_prob(x, condition)

    def log_prob(self, x: ArrayLike, condition: ArrayLike | None = None) -> Array:
        """Evaluate the log probability.

        Uses numpy-like broadcasting if additional leading dimensions are passed.

        Args:
            x: Points at which to evaluate density.
            condition: Conditioning variables. Defaults to None.

        Returns:
            Array: Jax array of log probabilities.
        """
        x = arraylike_to_array(x, err_name="x")
        if self.cond_shape is not None:
            condition = arraylike_to_array(condition, err_name="condition")
        lps = self._vectorize(self._log_prob)(x, condition)
        return jnp.where(jnp.isnan(lps), -jnp.inf, lps)

    def sample(
        self,
        key: Array,
        sample_shape: tuple[int, ...] = (),
        condition: ArrayLike | None = None,
    ) -> Array:
        """Sample from the distribution.

        For unconditional distributions, the output will be of shape
        ``sample_shape + dist.shape``. For conditional distributions, a batch dimension
        in the condition is supported, and the output shape will be
        ``sample_shape + condition_batch_shape + dist.shape``.
        See the example for more information.

        Args:
            key: Jax random key.
            condition: Conditioning variables. Defaults to None.
            sample_shape: Sample shape. Defaults to ().

        Example:
            The below example shows the behaviour of sampling, for an unconditional
            and a conditional distribution.

            .. testsetup::

                from flowjax.distributions import StandardNormal
                import jax.random as jr
                import jax.numpy as jnp
                from flowjax.flows import coupling_flow
                from flowjax.bijections import Affine
                # For a unconditional distribution:
                key = jr.PRNGKey(0)
                dist = StandardNormal((2,))
                # For a conditional distribution
                cond_dist = coupling_flow(
                    key, base_dist=StandardNormal((2,)), cond_dim=3
                    )

            For an unconditional distribution:

            .. doctest::

                >>> dist.shape
                (2,)
                >>> samples = dist.sample(key, (10, ))
                >>> samples.shape
                (10, 2)

            For a conditional distribution:

            .. doctest::

                >>> cond_dist.shape
                (2,)
                >>> cond_dist.cond_shape
                (3,)
                >>> # Sample 10 times for a particular condition
                >>> samples = cond_dist.sample(key, (10,), condition=jnp.ones(3))
                >>> samples.shape
                (10, 2)
                >>> # Sampling, batching over a condition
                >>> samples = cond_dist.sample(key, condition=jnp.ones((5, 3)))
                >>> samples.shape
                (5, 2)
                >>> # Sample 10 times for each of 5 conditioning variables
                >>> samples = cond_dist.sample(key, (10,), condition=jnp.ones((5, 3)))
                >>> samples.shape
                (10, 5, 2)


        """
        if self.cond_shape is not None:
            condition = arraylike_to_array(condition, err_name="condition")
        keys = self._get_sample_keys(key, sample_shape, condition)
        return self._vectorize(self._sample)(keys, condition)

    def sample_and_log_prob(
        self,
        key: Array,
        sample_shape: tuple[int, ...] = (),
        condition: ArrayLike | None = None,
    ):
        """Sample the distribution and return the samples with their log probabilities.

        For transformed distributions (especially flows), this will generally be more
        efficient than calling the methods seperately. Refer to the
        :py:meth:`~flowjax.distributions.AbstractDistribution.sample` documentation for
        more information.

        Args:
            key: Jax random key.
            condition: Conditioning variables. Defaults to None.
            sample_shape: Sample shape. Defaults to ().
        """
        if self.cond_shape is not None:
            condition = arraylike_to_array(condition, err_name="condition")
        keys = self._get_sample_keys(key, sample_shape, condition)
        return self._vectorize(self._sample_and_log_prob)(keys, condition)

    @property
    def ndim(self):
        """Number of dimensions in the distribution (the length of the shape)."""
        return len(self.shape)

    @property
    def cond_ndim(self):
        """Number of dimensions of the conditioning variable (length of cond_shape)."""
        return None if self.cond_shape is None else len(self.cond_shape)

    def _vectorize(self, method: callable) -> callable:
        """Returns a vectorized version of the distribution method."""
        # Get shapes without broadcasting - note the (2, ) corresponds to key arrays.
        maybe_cond = [] if self.cond_shape is None else [self.cond_shape]
        in_shapes = {
            "_sample_and_log_prob": [(2,)] + maybe_cond,
            "_sample": [(2,)] + maybe_cond,
            "_log_prob": [self.shape] + maybe_cond,
        }
        out_shapes = {
            "_sample_and_log_prob": [self.shape, ()],
            "_sample": [self.shape],
            "_log_prob": [()],
        }
        in_shapes, out_shapes = in_shapes[method.__name__], out_shapes[method.__name__]

        def _check_shapes(method):
            # Wraps unvectorised method with shape checking
            @wraps(method)
            def _wrapper(*args, **kwargs):
                bound = inspect.signature(method).bind(*args, **kwargs)
                for in_shape, (name, arg) in zip(
                    in_shapes,
                    bound.arguments.items(),
                    strict=False,
                ):
                    if arg.shape != in_shape:
                        raise ValueError(
                            f"Expected trailing dimensions matching {in_shape} for "
                            f"{name}; got {arg.shape}.",
                        )
                return method(*args, **kwargs)

            return _wrapper

        signature = _get_ufunc_signature(in_shapes, out_shapes)
        ex = frozenset([1]) if self.cond_shape is None else frozenset()
        return jnp.vectorize(_check_shapes(method), signature=signature, excluded=ex)

    def _get_sample_keys(self, key, sample_shape, condition):
        if self.cond_shape is not None:
            leading_cond_shape = condition.shape[: -self.cond_ndim or None]
        else:
            leading_cond_shape = ()
        key_shape = sample_shape + leading_cond_shape
        key_size = max(1, prod(key_shape))  # Still need 1 key for scalar sample
        return jnp.reshape(jr.split(key, key_size), (*key_shape, 2))


class AbstractTransformed(AbstractDistribution):
    """Abstract class respresenting transformed distributions.

    We take the forward bijection for use in sampling, and the inverse for use in
    density evaluation. See also :class:`Transformed`.

    Concete implementations should subclass :class:`AbstractTransformed`, and
    define the abstract attributes ``base_dist`` and ``bijection``. See the source code
    for :class:`Normal` as a simple example.

    Attributes:
        base_dist: The base distribution.
        bijection: The transformation to apply.
    """

    base_dist: AbstractVar[AbstractDistribution]
    bijection: AbstractVar[AbstractBijection]

    def _log_prob(self, x, condition=None):
        z, log_abs_det = self.bijection.inverse_and_log_det(x, condition)
        p_z = self.base_dist._log_prob(z, condition)
        return p_z + log_abs_det

    def _sample(self, key, condition=None):
        base_sample = self.base_dist._sample(key, condition)
        return self.bijection.transform(base_sample, condition)

    def _sample_and_log_prob(
        self,
        key: Array,
        condition: Array | None = None,
    ):  # TODO add overide decorator when python>=3.12 is common
        # We override to avoid computing the inverse transformation.
        base_sample, log_prob_base = self.base_dist._sample_and_log_prob(key, condition)
        sample, forward_log_dets = self.bijection.transform_and_log_det(
            base_sample,
            condition,
        )
        return sample, log_prob_base - forward_log_dets

    def __check_init__(self):  # TODO test errors and test conditional base distribution
        """Checks cond_shape is compatible in both bijection and distribution."""
        if (
            self.base_dist.cond_shape is not None
            and self.bijection.cond_shape is not None
            and self.base_dist.cond_shape != self.bijection.cond_shape
        ):
            raise ValueError(
                "The base distribution and bijection are both conditional "
                "but have mismatched cond_shape attributes. Base distribution has"
                f"{self.base_dist.cond_shape}, and the bijection has"
                f"{self.bijection.cond_shape}.",
            )

    def merge_transforms(self):
        """Unnests nested transformed distributions.

        Returns an equivilent distribution, but ravelling nested
        :class:`AbstractTransformed` distributions such that the returned distribution
        has a base distribution that is not an :class:`AbstractTransformed` instance.
        """
        if not isinstance(self.base_dist, AbstractTransformed):
            return self
        base_dist = self.base_dist
        bijections = [self.bijection]
        while isinstance(base_dist, AbstractTransformed):
            bijections.append(base_dist.bijection)
            base_dist = base_dist.base_dist
        bijection = Chain(list(reversed(bijections))).merge_chains()
        return Transformed(base_dist, bijection)

    @property
    def shape(self):
        return self.base_dist.shape

    @property
    def cond_shape(self):
        return merge_cond_shapes((self.bijection.cond_shape, self.base_dist.cond_shape))


class Transformed(AbstractTransformed):
    """Form a distribution like object using a base distribution and a bijection.

    We take the forward bijection for use in sampling, and the inverse
    bijection for use in density evaluation.

    .. warning::
            It is the currently the users responsibility to ensure the bijection is
            valid across the entire support of the distribution. Failure to do so may
            lead to to unexpected results.

    Args:
        base_dist: Base distribution.
        bijection: Bijection to transform distribution.

    Example:
        .. doctest::

            >>> from flowjax.distributions import StandardNormal, Transformed
            >>> from flowjax.bijections import Affine
            >>> normal = StandardNormal()
            >>> bijection = Affine(1)
            >>> transformed = Transformed(normal, bijection)
    """

    base_dist: AbstractDistribution
    bijection: AbstractBijection


class StandardNormal(AbstractDistribution):
    """Standard normal distribution.

    Note unlike :class:`Normal`, this has no trainable parameters.

    Args:
        shape: The shape of the distribution. Defaults to ().
    """

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def _log_prob(self, x, condition=None):
        return jstats.norm.logpdf(x).sum()

    def _sample(self, key, condition=None):
        return jr.normal(key, self.shape)


class Normal(AbstractTransformed):
    """An independent Normal distribution with mean and std for each dimension.

    ``loc`` and ``scale`` should broadcast to the desired shape of the distribution.

    Args:
        loc: Means. Defaults to 0.
        scale: Standard deviations. Defaults to 1.
    """

    base_dist: StandardNormal
    bijection: Affine
    cond_shape: ClassVar[None] = None

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        self.base_dist = StandardNormal(
            jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale)),
        )
        self.bijection = Affine(loc=loc, scale=scale)

    @property
    def loc(self):
        """Location of the distribution."""
        return self.bijection.loc

    @property
    def scale(self):
        """Scale of the distribution."""
        return unwrap(self.bijection.scale)


class LogNormal(AbstractTransformed):
    """Log normal distribution.

    ``loc`` and ``scale`` here refers to the underlying normal distribution.

    Args:
        loc: Location paramter. Defaults to 0.
        scale: Scale parameter. Defaults to 1.
    """

    base_dist: StandardNormal
    bijection: Chain

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        shape = jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        self.base_dist = StandardNormal(shape)
        self.bijection = Chain([Affine(loc, scale), Exp(shape)])

    @property
    def loc(self):
        """Location of the distribution."""
        return self.bijection[0].loc

    @property
    def scale(self):
        """Scale of the distribution."""
        return unwrap(self.bijection[0].scale)


class MultivariateNormal(AbstractTransformed):
    """Multivariate normal distribution.

    Internally this is parameterised using the Cholesky decomposition of the covariance
    matrix.

    Args:
        loc: The location/mean parameter vector. If this is scalar it is broadcast to
            the dimension implied by the covariance matrix.
        covariance: Covariance matrix.
    """

    base_dist: StandardNormal
    bijection: TriangularAffine

    def __init__(self, loc: ArrayLike, covariance: ArrayLike):
        self.bijection = TriangularAffine(loc, linalg.cholesky(covariance))
        self.base_dist = StandardNormal(self.bijection.shape)

    @property
    def loc(self):
        """Location (mean) of the distribution."""
        return self.bijection.loc

    @property
    def covariance(self):
        """The covariance matrix."""
        arr = unwrap(self.bijection.arr)
        return arr @ arr.T


class _StandardUniform(AbstractDistribution):
    r"""Standard Uniform distribution."""

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def _log_prob(self, x, condition=None):
        return jstats.uniform.logpdf(x).sum()

    def _sample(self, key, condition=None):
        return jr.uniform(key, shape=self.shape)


class Uniform(AbstractTransformed):
    """Uniform distribution.

    ``minval`` and ``maxval`` should broadcast to the desired distribution shape.

    Args:
        minval: Minimum values.
        maxval: Maximum values.
    """

    base_dist: _StandardUniform
    bijection: Affine

    def __init__(self, minval: ArrayLike, maxval: ArrayLike):
        minval, maxval = arraylike_to_array(minval), arraylike_to_array(maxval)
        minval, maxval = eqx.error_if(
            (minval, maxval), maxval <= minval, "minval must be less than the maxval."
        )
        self.base_dist = _StandardUniform(
            jnp.broadcast_shapes(minval.shape, maxval.shape),
        )
        self.bijection = Affine(loc=minval, scale=maxval - minval)

    @property
    def minval(self):
        """Minimum value of the uniform distribution."""
        return self.bijection.loc

    @property
    def maxval(self):
        """Maximum value of the uniform distribution."""
        return self.bijection.loc + unwrap(self.bijection.scale)


class _StandardGumbel(AbstractDistribution):
    """Standard gumbel distribution (https://en.wikipedia.org/wiki/Gumbel_distribution)."""

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def _log_prob(self, x, condition=None):
        return -(x + jnp.exp(-x)).sum()

    def _sample(self, key, condition=None):
        return jr.gumbel(key, shape=self.shape)


class Gumbel(AbstractTransformed):
    """Gumbel distribution (https://en.wikipedia.org/wiki/Gumbel_distribution).

    ``loc`` and ``scale`` should broadcast to the dimension of the distribution.

    Args:
        loc: Location paramter.
        scale: Scale parameter. Defaults to 1.
    """

    base_dist: _StandardGumbel
    bijection: Affine

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        self.base_dist = _StandardGumbel(
            jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale)),
        )
        self.bijection = Affine(loc, scale)

    @property
    def loc(self):
        """Location of the distribution."""
        return self.bijection.loc

    @property
    def scale(self):
        """Scale of the distribution."""
        return unwrap(self.bijection.scale)


class _StandardCauchy(AbstractDistribution):
    """Implements standard cauchy distribution (loc=0, scale=1).

    Ref: https://en.wikipedia.org/wiki/Cauchy_distribution.
    """

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def _log_prob(self, x, condition=None):
        return jstats.cauchy.logpdf(x).sum()

    def _sample(self, key, condition=None):
        return jr.cauchy(key, shape=self.shape)


class Cauchy(AbstractTransformed):
    """Cauchy distribution (https://en.wikipedia.org/wiki/Cauchy_distribution).

    ``loc`` and ``scale`` should broadcast to the dimension of the distribution.

    Args:
        loc: Location paramter.
        scale: Scale parameter. Defaults to 1.
    """

    base_dist: _StandardCauchy
    bijection: Affine

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        self.base_dist = _StandardCauchy(
            jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale)),
        )
        self.bijection = Affine(loc, scale)

    @property
    def loc(self):
        """Location of the distribution."""
        return self.bijection.loc

    @property
    def scale(self):
        """Scale of the distribution."""
        return unwrap(self.bijection.scale)


class _StandardStudentT(AbstractDistribution):
    """Implements student T distribution with specified degrees of freedom."""

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    log_df: Array

    def __init__(self, df: ArrayLike):
        if jnp.any(df <= 0):
            raise ValueError("degrees of freedom values must be positive.")
        self.shape = jnp.shape(df)
        self.log_df = jnp.log(df)

    def _log_prob(self, x, condition=None):
        return jstats.t.logpdf(x, df=self.df).sum()

    def _sample(self, key, condition=None):
        return jr.t(key, df=self.df, shape=self.shape)

    @property
    def df(self):
        """The degrees of freedom of the distibution."""
        return jnp.exp(self.log_df)


class StudentT(AbstractTransformed):
    """Student T distribution (https://en.wikipedia.org/wiki/Student%27s_t-distribution).

    ``df``, ``loc`` and ``scale`` broadcast to the dimension of the distribution.

    Args:
        df: The degrees of freedom.
        loc: Location parameter. Defaults to 0.
        scale: Scale parameter. Defaults to 1.
    """

    base_dist: _StandardStudentT
    bijection: Affine

    def __init__(self, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1):
        df, loc, scale = jnp.broadcast_arrays(df, loc, scale)
        self.base_dist = _StandardStudentT(df)
        self.bijection = Affine(loc, scale)

    @property
    def loc(self):
        """Location of the distribution."""
        return self.bijection.loc

    @property
    def scale(self):
        """Scale of the distribution."""
        return unwrap(self.bijection.scale)

    @property
    def df(self):
        """The degrees of freedom of the distribution."""
        return self.base_dist.df


class _StandardLaplace(AbstractDistribution):
    """Implements standard laplace distribution (loc=0, scale=1)."""

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def _log_prob(self, x, condition=None):
        return jstats.laplace.logpdf(x).sum()

    def _sample(self, key, condition=None):
        return jr.laplace(key, shape=self.shape)


class Laplace(AbstractTransformed):
    """Laplace distribution.

    ``loc`` and ``scale`` should broadcast to the dimension of the distribution..

    Args:
        loc: Location paramter. Defaults to 0.
        scale: Scale parameter. Defaults to 1.
    """

    base_dist: _StandardLaplace
    bijection: Affine

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        shape = jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        self.base_dist = _StandardLaplace(shape)
        self.bijection = Affine(loc, scale)

    @property
    def loc(self):
        """Location of the distribution."""
        return self.bijection.loc

    @property
    def scale(self):
        """Scale of the distribution."""
        return unwrap(self.bijection.scale)


class _StandardExponential(AbstractDistribution):
    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def _log_prob(self, x, condition=None):
        return jstats.expon.logpdf(x).sum()

    def _sample(self, key, condition=None):
        return jr.exponential(key, shape=self.shape)


class Exponential(AbstractTransformed):
    """Exponential distribution.

    Args:
        rate: The rate parameter (1 / scale).
    """

    base_dist: _StandardExponential
    bijection: Scale

    def __init__(self, rate: Array):
        self.base_dist = _StandardExponential(rate.shape)
        self.bijection = Scale(1 / rate)

    @property
    def rate(self):
        return 1 / unwrap(self.bijection.scale)


class SpecializeCondition(AbstractDistribution):  # TODO check tested
    """Specialise a distribution to a particular conditioning variable instance.

    This makes the distribution act like an unconditional distribution, i.e. the
    distribution methods implicitly will use the condition passed on instantiation
    of the class.

    Args:
        dist: Conditional distribution to specialize.
        condition: Instance of conditioning variable with shape matching
            ``dist.cond_shape``. Defaults to None.
        stop_gradient: Whether to use ``jax.lax.stop_gradient`` to prevent training of
            the condition array. Defaults to True.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    def __init__(
        self,
        dist: AbstractDistribution,
        condition: ArrayLike,
        *,
        stop_gradient: bool = True,
    ):
        condition = arraylike_to_array(condition)
        if self.dist.cond_shape != condition.shape:
            raise ValueError(
                f"Expected condition shape {self.dist.cond_shape}, got "
                f"{condition.shape}",
            )
        self.dist = dist
        self._condition = condition
        self.shape = dist.shape
        self.stop_gradient = stop_gradient

    def _log_prob(self, x, condition=None):
        return self.dist._log_prob(x, self.condition)

    def _sample(self, key, condition=None):
        return self.dist._sample(key, self.condition)

    def _sample_and_log_prob(self, key, condition=None):
        return self.dist._sample_and_log_prob(key, self.condition)

    @property
    def condition(self):
        """The conditioning variable, possibly with stop_gradient applied."""
        return stop_gradient(self._condition) if self.stop_gradient else self._condition
