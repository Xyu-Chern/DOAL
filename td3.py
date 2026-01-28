#!/usr/bin/env python
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import wandb
import numpy as np
from rejax import TD3

# ========== 配置 ==========
config = {
    "env": "Pendulum-v1",
    "total_timesteps": 50000,
    "eval_freq": 5000,
    "num_envs": 1,
    "learning_rate": 0.001,
    "batch_size": 256,
    "gamma": 0.99,
    "exploration_noise": 0.1,
    "target_noise": 0.2,
    "target_noise_clip": 0.5,
    "policy_delay": 2,
}

num_seeds = 5

# ========== 初始化WandB ==========
wandb.init(
    project="td3-direct",
    config={**config, "num_seeds": num_seeds},
    name=f"td3-{num_seeds}-seeds",
)

print(f"训练 {num_seeds} 个TD3智能体...")

# ========== 自定义评估回调 ==========
def create_wandb_eval_callback():
    """创建WandB评估回调"""
    def evaluate_callback(algo, ts, rng):
        # 运行默认评估
        result = algo.eval_callback(algo, ts, rng)
        
        if result is not None and len(result) >= 2:
            returns, lengths = result[0], result[1]
            
            if returns.size > 0:
                step = int(ts.global_step)
                
                # 记录到WandB
                wandb.log({
                    "eval/return": float(np.mean(returns)),
                    "eval/return_std": float(np.std(returns)),
                    "eval/length": float(np.mean(lengths)),
                    "train/step": step,
                }, step=step)
                
                print(f"[Step {step}] Return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
        
        return result
    
    return evaluate_callback

# ========== 创建TD3配置 ==========
td3_config = TD3.create(**config)

# ========== 替换评估回调 ==========
wandb_callback = create_wandb_eval_callback()
td3_config = td3_config.replace(eval_callback=wandb_callback)

# ========== 批量训练 ==========
print("编译训练函数...")

# 创建vmap训练函数
train_fn = jax.jit(td3_config.train)
vmapped_train_fn = jax.vmap(train_fn, in_axes=(0,))

# 准备随机种子
keys = jax.random.split(jax.random.PRNGKey(0), num_seeds)

print("开始训练...")
train_states, evaluations = vmapped_train_fn(keys)

print("训练完成!")

# ========== 分析结果 ==========
if evaluations is not None and len(evaluations) >= 2:
    returns_batch, lengths_batch = evaluations[0], evaluations[1]
    
    # 处理批量维度
    if len(returns_batch.shape) > 1:
        for i in range(num_seeds):
            returns = returns_batch[i]
            final_return = float(np.mean(returns))
            
            wandb.log({f"seed_{i}/final_return": final_return})
            print(f"种子 {i} 最终回报: {final_return:.2f}")
        
        final_returns = [float(np.mean(returns_batch[i])) for i in range(num_seeds)]
    else:
        final_return = float(np.mean(returns_batch))
        wandb.log({"final_return": final_return})
        print(f"最终回报: {final_return:.2f}")
        final_returns = [final_return]
    
    # 记录统计
    if len(final_returns) > 1:
        stats = {
            "mean_final_return": np.mean(final_returns),
            "std_final_return": np.std(final_returns),
            "best_final_return": np.max(final_returns),
            "worst_final_return": np.min(final_returns),
        }
        
        wandb.log(stats)
        wandb.summary.update(stats)
        
        print(f"\n最终统计:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")

# ========== 完成 ==========
wandb.finish()
print("\n实验完成!")