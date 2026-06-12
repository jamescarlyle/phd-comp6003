# Hopper using tensorneat.

On macos, tensorneat needs jax, and jax needs specific versions to run on Mac M4 Metal GPU:

> uv pip install jax==0.4.35 jaxlib==0.4.35 jax-metal==0.1.0

Then use the following script to check that jax is working properly:

   ```python
    import os
    os.environ["JAX_METAL_COLLECTIVE_OPS"] = "0"
    import jax
    import jax.numpy as jnp
    print(f"Active devices: {jax.devices()}")
    try:
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (5000, 5000))
        
        @jax.jit
        def multiply(mat):
            return jnp.matmul(mat, mat)

        result = multiply(x)
        print("Multiplication successful on M4 GPU!")
    except Exception as e:
        print(f"Still hitting an error: {e}")
   ```

