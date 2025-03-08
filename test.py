import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (10,))
print(x)

