from rejax import SAC

# Get train function and initialize config for training
algo = SAC.create(env="CartPole-v1", learning_rate=0.001)

# Jit the training function
train_fn = jax.jit(algo.train)

# Vmap training function over 300 initial seeds
vmapped_train_fn = jax.vmap(train_fn)

# Train 300 agents!
keys = jax.random.split(jax.random.PRNGKey(0), 300)
train_state, evaluation = vmapped_train_fn(keys)