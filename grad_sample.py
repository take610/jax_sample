import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random

# grad sample


def f(x1, x2):
    # x1 derive -> 2x1 + x2
    # x2 derive -> x1 + 2x2
    return x1 ** 2 + x1 * x2 + x2 ** 2


x1 = 1.0
x2 = 1.0
x1_grad = grad(f, argnums=0)(x1, x2)
print(f"x1 grad: {x1_grad}")
x2_grad = grad(f, argnums=1)(x1, x2)
print(f"x2 grad: {x2_grad}")

