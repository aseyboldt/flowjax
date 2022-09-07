## flowjax
-------

Normalising flows in JAX. Training a flow can be done in a few lines of code:

```
from flowjax.flows import BlockNeuralAutoregressiveFlow
from flowjax.train_utils import train_flow
from flowjax.distributions import Normal
from jax import random

data_key, flow_key, train_key = random.split(random.PRNGKey(0), 3)

x = random.uniform(data_key, (10000, 3))  # Toy data
base_dist = Normal(3)
flow = BlockNeuralAutoregressiveFlow(flow_key, base_dist)
flow, losses = train_flow(train_key, flow, x, learning_rate=0.05)

# We can now evaluate the log-probability of arbitrary points
flow.log_prob(x)
```

The package currently has the following features:

- Easy composition of transformers with **coupling** or **masked autoregressive** conditioner architectures, e.g. allowing construction of
    - [Affine coupling flows](https://arxiv.org/abs/1906.04032/) (i.e. RealNVP)
    - [Rational quadratic spline coupling flows](https://arxiv.org/abs/1906.04032/) (i.e. neural spline flows)
    - [Affine masked autoregressive flows](https://arxiv.org/abs/1705.07057v4)
    - Rational quadratic spline masked autoregressive flows
- [Block neural autoregressive flows](https://arxiv.org/abs/1904.04676)

For more detailed examples, see [examples](https://github.com/danielward27/flowjax/blob/main/examples/).

## Installation
```
pip install flowjax
```

## Warning
This package is new and may have substantial breaking changes between major releases.

## TODO
A few limitations / things that could be worth including in the future:

- Support embedding networks (for dimensionality reduction of conditioning variables)
- Add batch/layer normalisation to neural networks
- Training script for variational inference
- Add documentation

## Related
We make use of the [Equinox](https://arxiv.org/abs/2111.00254) package, which facilitates object-oriented programming with Jax. 

## Authors
`flowjax` was written by `Daniel Ward <danielward27@outlook.com>`.
