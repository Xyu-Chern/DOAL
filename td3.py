
import jax
import wandb
import numpy as np
from rejax import TD3

# ========== 初始化WandB ==========
wandb.init(
    project="td3-no-recursion",
    config={"env": "Pendulum-v1", "num_seeds": 3},
    name="td3-fixed",
)

print("训练TD3（修复递归错误）")

# ========== 创建TD3算法 ==========
algo = TD3.create(
    env="Pendulum-v1",
    total_timesteps=20000,  # 先用小步数测试
    eval_freq=2000,
    num_envs=1,
    learning_rate=0.001,
    batch_size=256,
    gamma=0.99,
    skip_initial_evaluation=True,  # 跳过初始评估避免递归
)

# ========== 保存原始评估回调 ==========
# 我们需要获取原始的eval_callback
original_eval_callback = None

# 方法1：尝试从算法实例获取
if hasattr(algo, '_eval_callback'):
    original_eval_callback = algo._eval_callback
else:
    # 方法2：创建一个简单的评估函数
    def simple_eval_callback(algo_instance, ts, rng):
        # 简化评估：运行几个episode
        env = algo_instance.env
        env_params = algo_instance.env_params
        
        # 评估参数
        num_eval_episodes = 5
        max_steps = 200
        
        returns = []
        lengths = []
        
        # 运行评估
        for ep in range(num_eval_episodes):
            rng, rng_ep = jax.random.split(rng)
            obs, env_state = env.reset(rng_ep, env_params)
            
            episode_return = 0
            episode_length = 0
            done = False
            
            for step in range(max_steps):
                # 使用策略选择动作
                rng_ep, rng_act = jax.random.split(rng_ep)
                action = algo_instance.make_act(ts)(obs, rng_act)
                
                # 执行一步
                next_obs, env_state, reward, done, _ = env.step(
                    rng_ep, env_state, action, env_params
                )
                
                episode_return += reward
                episode_length += 1
                obs = next_obs
                
                if done:
                    break
            
            returns.append(episode_return)
            lengths.append(episode_length)
        
        return (jnp.array(returns), jnp.array(lengths))
    
    original_eval_callback = simple_eval_callback

# ========== 创建WandB回调 ==========
def wandb_eval_callback(algo_instance, ts, rng):
    """包装评估回调以记录到WandB"""
    # 调用原始评估函数
    result = original_eval_callback(algo_instance, ts, rng)
    
    if result is not None and len(result) >= 2:
        returns, lengths = result[0], result[1]
        
        if returns.size > 0:
            step = int(ts.global_step)
            mean_return = float(np.mean(returns))
            mean_length = float(np.mean(lengths))
            
            wandb.log({
                "eval/return": mean_return,
                "eval/length": mean_length,
                "train/step": step,
            }, step=step)
            
            print(f"[Step {step}] Return: {mean_return:.2f}, Length: {mean_length:.1f}")
    
    return result

# ========== 替换回调函数 ==========
algo = algo.replace(eval_callback=wandb_eval_callback)

# ========== 训练单个智能体（先测试） ==========
print("\n先训练单个智能体测试...")

rng = jax.random.PRNGKey(42)
train_fn = jax.jit(algo.train)

train_state, evaluation = train_fn(rng=rng)

print(f"单个智能体训练完成！最终步数: {train_state.global_step}")

# ========== 批量训练 ==========
print("\n现在批量训练多个智能体...")
num_seeds = 3

# 准备随机种子
keys = jax.random.split(jax.random.PRNGKey(0), num_seeds)

# 创建新的算法实例用于批量训练
algo_batch = TD3.create(
    env="Pendulum-v1",
    total_timesteps=20000,
    eval_freq=2000,
    num_envs=1,
    learning_rate=0.001,
    batch_size=256,
    gamma=0.99,
    skip_initial_evaluation=True,
)

# 设置相同的回调
algo_batch = algo_batch.replace(eval_callback=wandb_eval_callback)

# Vmap训练
train_fn_batch = jax.jit(algo_batch.train)
vmapped_train_fn = jax.vmap(train_fn_batch)

print(f"训练 {num_seeds} 个智能体...")
train_states, evaluations = vmapped_train_fn(keys)

print("批量训练完成!")

# ========== 记录最终结果 ==========
if evaluations is not None and len(evaluations) >= 2:
    returns_batch, _ = evaluations[0], evaluations[1]
    
    if len(returns_batch.shape) == 2:  # 批量结果
        for i in range(num_seeds):
            final_return = float(np.mean(returns_batch[i]))
            wandb.log({f"final/seed_{i}": final_return})
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
        wandb.log({"final_return": final_return})
        wandb.summary["final_return"] = final_return
        print(f"最终回报: {final_return:.2f}")

# ========== 完成 ==========
wandb.finish()
print("\n实验完成!")