
import jax
import jax.numpy as jnp
import wandb
import numpy as np
from doal import DOAL

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
        
        return ts, (eval_result, ts.global_step)  # 返回global_step

    ts, results = jax.lax.scan(
        eval_iteration,
        ts,
        jnp.arange(np.ceil(self.total_timesteps / self.eval_freq).astype(int)),
        np.ceil(self.total_timesteps / self.eval_freq).astype(int),
    )
    
    # 分离评估结果和步骤
    evaluation, steps = results
    
    # ========== 在scan外部处理WandB日志 ==========
    # 这是一个关键修复：在JAX扫描外部处理日志
    def process_logs(eval_batch, step_batch):
        """处理一批评估结果并记录到WandB"""
        eval_batch = jax.device_get(eval_batch)  # 从设备获取数据
        step_batch = jax.device_get(step_batch)
        
        for i in range(len(step_batch)):
            eval_result = jax.tree.map(lambda x: x[i], eval_batch)
            step = step_batch[i]
            
            if eval_result is not None and len(eval_result) >= 2:
                returns, lengths = eval_result[0], eval_result[1]
                if returns.size > 0:
                    # 转换为Python原生类型
                    step_int = int(step)
                    mean_return = float(jnp.mean(returns))
                    mean_length = float(jnp.mean(lengths))
                    
                    wandb.log({
                        "eval/return": mean_return,
                        "eval/length": mean_length,
                        "train/step": step_int
                    }, step=step_int)
                    print(f"[Step {step_int}] Return: {mean_return:.2f}")

    # 处理所有评估结果
    process_logs(evaluation, steps)
    
    # 处理初始评估
    if not self.skip_initial_evaluation and initial_evaluation is not None:
        if len(initial_evaluation) >= 2:
            returns, lengths = initial_evaluation[0], initial_evaluation[1]
            if returns.size > 0:
                wandb.log({
                    "eval/return": float(jnp.mean(returns)),
                    "eval/length": float(jnp.mean(lengths)),
                    "train/step": 0
                }, step=0)

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