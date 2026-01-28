import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import wandb
from doal import DOAL

# 初始化WandB
wandb.init(
    project="doal-training",
    config={"env": "Pendulum-v1"},
)

# 保存原始的train方法
original_train = DOAL.train

def train_with_wandb(self, rng=None, train_state=None):
    """添加WandB日志的train方法"""
    if train_state is None and rng is None:
        raise ValueError("Either train_state or rng must be provided")

    ts = train_state or self.init_state(rng)

    if not self.skip_initial_evaluation:
        initial_evaluation = self.eval_callback(self, ts, ts.rng)
        # 记录初始评估
        if initial_evaluation is not None and len(initial_evaluation) >= 2:
            returns = initial_evaluation[0]
            if returns.size > 0:
                wandb.log({
                    "eval/return": float(jnp.mean(returns)),
                    "eval/step": 0
                }, step=0)

    def eval_iteration(ts, eval_idx):
        # Run a few training iterations
        steps_per_train_it = self.num_envs * self.policy_delay
        num_train_its = np.ceil(self.eval_freq / steps_per_train_it).astype(int)
        ts = jax.lax.fori_loop(
            0,
            num_train_its,
            lambda _, ts: self.train_iteration(ts),
            ts,
        )

        # Run evaluation
        eval_result = self.eval_callback(self, ts, ts.rng)
        
        # 记录到WandB
        if eval_result is not None and len(eval_result) >= 2:
            returns, lengths = eval_result[0], eval_result[1]
            if returns.size > 0:
                step = int(ts.global_step)
                wandb.log({
                    "eval/return": float(jnp.mean(returns)),
                    "eval/length": float(jnp.mean(lengths)),
                    "train/step": step
                }, step=step)
                print(f"[Step {step}] Return: {float(jnp.mean(returns)):.2f}")
        
        return ts, eval_result

    import numpy as np
    ts, evaluation = jax.lax.scan(
        eval_iteration,
        ts,
        jnp.arange(np.ceil(self.total_timesteps / self.eval_freq).astype(int)),
        np.ceil(self.total_timesteps / self.eval_freq).astype(int),
    )

    if not self.skip_initial_evaluation:
        evaluation = jax.tree.map(
            lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
            initial_evaluation,
            evaluation,
        )

    return ts, evaluation

# 替换train方法
DOAL.train = train_with_wandb

# 创建并训练算法
algo = DOAL.create(
    env="Pendulum-v1",
    total_timesteps=1000,
    eval_freq=100,
    num_envs=1,
)

rng = jax.random.PRNGKey(42)
train_state, evaluation = algo.train(rng=rng)

wandb.finish()
print("训练完成!")