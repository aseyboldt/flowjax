import pytest
from flowjax.utils import tile_until_length, broadcast_arrays_1d
import jax.numpy as jnp


def test_tile_until_length():
    x = jnp.array([1, 2])

    y = tile_until_length(x, 4)
    assert jnp.all(y == jnp.array([1, 2, 1, 2]))

    y = tile_until_length(x, 3)
    assert jnp.all(y == jnp.array([1, 2, 1]))

    y = tile_until_length(x, 1)
    assert jnp.all(y == jnp.array([1]))



test_cases = [
    # arrays, expected_shape
    ((jnp.ones(3), jnp.ones(3)), (3, )),
    ((jnp.ones(3), 1.), (3, )),
    ((1., jnp.ones(3)), (3, )),
    ((1., 1.), (1, )),
    (((1.), ), (1, ))
]

@pytest.mark.parametrize("arrays,expected", test_cases)
def test_broadcast_arrays_1d(arrays, expected):
    out = broadcast_arrays_1d(*arrays)
    assert len(arrays) == len(out)
    match_expected = [a.shape == expected for a in out]
    assert jnp.all(jnp.array(match_expected))
