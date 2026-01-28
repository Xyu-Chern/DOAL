import jax
import wandb
import numpy as np

# 导入你修改后的DOAL类
from doal import DOAL  # 假设你的文件名为doal.py

# ========== 初始化WandB ==========
wandb.init(
    project="doal-integrated",
    config={
        "env": "Pendulum-v1",
        "total_timesteps": 50000,
        "eval_freq": 5000,
        "num_envs": 1,
    },
    name="doal-pendulum",
)

print("使用内置WandB的DOAL训练")

# ========== 创建并训练算法 ==========
algo = DOAL.create(
    env="Pendulum-v1",
    total_timesteps=50000,
    eval_freq=5000,
    num_envs=1,
    learning_rate=0.001,
    batch_size=256,
    gamma=0.99,
    fill_buffer=1000,
    flow_steps=10,
    max_q_samples=4,
    policy_delay=2,
    alpha=0.2,
    delta=2.0,
    exploration_noise=0.1,
    target_noise=0.2,
    target_noise_clip=0.5,
)

# 训练单个智能体
print("开始训练单个智能体...")
rng = jax.random.PRNGKey(42)
train_state, evaluation = algo.train(rng=rng)

print(f"训练完成！最终步数: {train_state.global_step}")

# ========== 批量训练版本 ==========
print("\n现在批量训练多个智能体...")
num_seeds = 3

# 准备随机种子
keys = jax.random.split(jax.random.PRNGKey(0), num_seeds)

# Vmap训练
train_fn = jax.jit(algo.train)
vmapped_train_fn = jax.vmap(train_fn)

print(f"训练 {num_seeds} 个智能体...")
train_states, evaluations = vmapped_train_fn(keys)

print("批量训练完成!")

# 分析批量结果
if evaluations is not None and len(evaluations) >= 2:
    returns_batch, _ = evaluations[0], evaluations[1]
    
    if len(returns_batch.shape) == 2:  # 批量结果
        for i in range(num_seeds):
            final_return = float(np.mean(returns_batch[i]))
            print(f"种子 {i} 最终回报: {final_return:.2f}")
        
        final_returns = [float(np.mean(returns_batch[i])) for i in range(num_seeds)]
        
        wandb.summary.update({
            "mean_final_return": np.mean(final_returns),
            "std_final_return": np.std(final_returns),
            "best_final_return": np.max(final_returns),
            "worst_final_return": np.min(final_returns),
        })
    else:
        final_return = float(np.mean(returns_batch))
        wandb.summary["final_return"] = final_return
        print(f"最终回报: {final_return:.2f}")

# ========== 完成 ==========
wandb.finish()
print("\n所有训练完成!")