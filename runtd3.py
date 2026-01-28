import jax
import jax.numpy as jnp
import numpy as np
import wandb
from rejax import TD3

class WandBTD3(TD3):
    def train(self, rng=None, train_state=None):
        ts = train_state or self.init_state(rng)

        # 1. 初始评估
        if not self.skip_initial_evaluation:
            initial_evaluation = self.eval_callback(self, ts, ts.rng)
            if initial_evaluation is not None:
                returns, lengths = initial_evaluation
                jax.debug.callback(
                    lambda r, l: self._log_to_wandb(0, r, l, prefix="eval"),
                    returns, lengths
                )

        # 2. 定义每一轮的训练 + 评估
        def eval_iteration(ts, unused):
            # 训练直到下一个评估点
            steps_per_train_it = self.num_envs * self.policy_delay
            num_train_its = np.ceil(self.eval_freq / steps_per_train_it).astype(int)
            
            ts = jax.lax.fori_loop(
                0, num_train_its,
                lambda _, ts: self.train_iteration(ts),
                ts
            )

            # 评估
            eval_result = self.eval_callback(self, ts, ts.rng)
            returns, lengths = eval_result
            
            # 使用 debug.callback 发送到 Python 侧记录 WandB
            jax.debug.callback(
                lambda step, r, l: self._log_to_wandb(step, r, l, prefix="eval"),
                ts.global_step, returns, lengths
            )
            
            return ts, eval_result

        # 3. 运行主循环
        num_evals = np.ceil(self.total_timesteps / self.eval_freq).astype(int)
        ts, evaluation = jax.lax.scan(eval_iteration, ts, None, length=num_evals)

        return ts, evaluation

    def _log_to_wandb(self, step, returns, lengths, prefix="eval"):
        """此函数在 CPU/Python 侧运行"""
        mean_return = float(np.mean(returns))
        # 只有在单种子训练时记录 wandb，或者只记录第一个种子，避免多进程冲突
        # 这里我们假设您是逐个种子训练或监控总平均值
        wandb.log({
            f"{prefix}/return": mean_return,
            f"{prefix}/length": float(np.mean(lengths)),
            "train/step": int(step)
        }, step=int(step))
        print(f"[{prefix.upper()}] Step: {int(step)} | Return: {mean_return:.2f}")

# 复用你之前的 JAX 友好评估函数
def custom_eval_callback(algo, train_state, rng):
    env, env_params = algo.env, algo.env_params
    max_steps, num_episodes = 200, 10
    act_fn = algo.make_act(train_state)

    def single_step(carry, _):
        obs, state, done, cum_rew, rng = carry
        rng, act_rng, env_rng = jax.random.split(rng, 3)
        action = act_fn(obs, act_rng)
        next_obs, next_state, reward, next_done, _ = env.step(env_rng, state, action, env_params)
        return (next_obs, next_state, jnp.logical_or(done, next_done), cum_rew + reward * (1.0 - done), rng), None

    def evaluate_episode(ep_rng):
        r_rng, run_rng = jax.random.split(ep_rng)
        obs, state = env.reset(r_rng, env_params)
        final_carry, _ = jax.lax.scan(single_step, (obs, state, jnp.array(False), 0.0, run_rng), None, length=max_steps)
        return final_carry[3], jnp.array(max_steps)

    returns, lengths = jax.vmap(evaluate_episode)(jax.random.split(rng, num_episodes))
    return returns, lengths

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from rejax import TD3

class WandBTD3(TD3):
    def train(self, rng=None, train_state=None):
        ts = train_state or self.init_state(rng)

        # 1. 初始评估
        if not self.skip_initial_evaluation:
            initial_evaluation = self.eval_callback(self, ts, ts.rng)
            if initial_evaluation is not None:
                returns, lengths = initial_evaluation
                jax.debug.callback(
                    lambda r, l: self._log_to_wandb(0, r, l, prefix="eval"),
                    returns, lengths
                )

        # 2. 定义每一轮的训练 + 评估
        def eval_iteration(ts, unused):
            # 训练直到下一个评估点
            steps_per_train_it = self.num_envs * self.policy_delay
            num_train_its = np.ceil(self.eval_freq / steps_per_train_it).astype(int)
            
            ts = jax.lax.fori_loop(
                0, num_train_its,
                lambda _, ts: self.train_iteration(ts),
                ts
            )

            # 评估
            eval_result = self.eval_callback(self, ts, ts.rng)
            returns, lengths = eval_result
            
            # 使用 debug.callback 发送到 Python 侧记录 WandB
            jax.debug.callback(
                lambda step, r, l: self._log_to_wandb(step, r, l, prefix="eval"),
                ts.global_step, returns, lengths
            )
            
            return ts, eval_result

        # 3. 运行主循环
        num_evals = np.ceil(self.total_timesteps / self.eval_freq).astype(int)
        ts, evaluation = jax.lax.scan(eval_iteration, ts, None, length=num_evals)

        return ts, evaluation

    def _log_to_wandb(self, step, returns, lengths, prefix="eval"):
        """This function runs on the CPU/Python side"""
        # Safety check: If wandb isn't initialized in this process, skip or init
        if wandb.run is None:
            return 

        mean_return = float(np.mean(returns))
        wandb.log({
            f"{prefix}/return": mean_return,
            f"{prefix}/length": float(np.mean(lengths)),
            "train/step": int(step)
        }, step=int(step))
        print(f"[{prefix.upper()}] Step: {int(step)} | Return: {mean_return:.2f}")

# 复用你之前的 JAX 友好评估函数
def custom_eval_callback(algo, train_state, rng):
    env, env_params = algo.env, algo.env_params
    max_steps, num_episodes = 200, 10
    act_fn = algo.make_act(train_state)

    def single_step(carry, _):
        obs, state, done, cum_rew, rng = carry
        rng, act_rng, env_rng = jax.random.split(rng, 3)
        action = act_fn(obs, act_rng)
        next_obs, next_state, reward, next_done, _ = env.step(env_rng, state, action, env_params)
        return (next_obs, next_state, jnp.logical_or(done, next_done), cum_rew + reward * (1.0 - done), rng), None

    def evaluate_episode(ep_rng):
        r_rng, run_rng = jax.random.split(ep_rng)
        obs, state = env.reset(r_rng, env_params)
        final_carry, _ = jax.lax.scan(single_step, (obs, state, jnp.array(False), 0.0, run_rng), None, length=max_steps)
        return final_carry[3], jnp.array(max_steps)

    returns, lengths = jax.vmap(evaluate_episode)(jax.random.split(rng, num_episodes))
    return returns, lengths


# 初始化 WandB
wandb.init(
    project="doal-integrated", # 保持和 DOAL 一样
    name="td3-pendulum-baseline",
    config={"algo": "TD3", "env": "brax/halfcheetah"}
)

# 使用我们重写后的 WandBTD3
# algo = WandBTD3.create(
#     env="Pendulum-v1",
#     total_timesteps=50000,
#     eval_freq=5000,
#     learning_rate=0.001,
# )

from flax import linen as nn

# algo = WandBTD3.create(
#     env="brax/hopper",
#     # 1. 激活函数必须是字符串，因为源码内部使用了 getattr(nn, activation)
#     actor_kwargs={"activation": "tanh"}, 
#     critic_kwargs={"activation": "tanh"},
    
#     # 2. 目标网络更新系数在 rejax 中叫 polyak (对应通常的 1-tau 或 tau)
#     # 如果你的 tuned 参数 tau=0.95 是指保留旧参数的比例，那么 polyak=0.95 是正确的
#     polyak=0.95, 
    
#     # 3. 算法特定参数 (对应源码中的字段)
#     exploration_noise=0.5,
#     target_noise=0.8,
#     target_noise_clip=0.5,
#     policy_delay=3,
    
#     # 4. 训练规模参数
#     total_timesteps=1_000_000,
#     eval_freq=50000,
#     num_envs=128,
#     learning_rate=0.00018789,
#     batch_size=256,
#     buffer_size=1_000_000,
#     fill_buffer=8192,
#     max_grad_norm=2.0,
#     normalize_observations=True,
# )


algo = WandBTD3.create(
    # 环境设置（选择其中一个）
    env="brax/halfcheetah",  # 或 "brax/hopper"
    
    # 网络结构参数
    actor_kwargs={"activation": "tanh"},
    critic_kwargs={"activation": "tanh"},
    
    # TD3特定参数
    exploration_noise=0.3,
    target_noise=0.8,
    target_noise_clip=0.6,
    policy_delay=1,  # halfcheetah使用1，hopper使用3
    
    # 训练参数
    num_envs=128,
    learning_rate=0.0001199272452295516,  # halfcheetah的学习率
    # learning_rate=0.00018789,  # hopper的学习率
    
    # 缓冲区参数
    buffer_size=1048576,  # 1M
    fill_buffer=8192,
    batch_size=512,  # halfcheetah的batch_size
    # batch_size=256,  # hopper的batch_size
    
    # 优化参数
    num_epochs=128,
    total_timesteps=5242880,  # 5.24M steps
    eval_freq=262144,
    
    # 算法核心参数
    gamma=0.99,
    polyak=0.99,  # halfcheetah的polyak
    # polyak=0.95,  # hopper的polyak
    
    # 训练稳定性
    max_grad_norm=5.0,
    normalize_observations=True,
    
    # 可选：wandb日志设置
    # project="td3-brax",
    # group="halfcheetah"  # 或 "hopper"
)

algo = algo.replace(eval_callback=custom_eval_callback)

# Jit 并训练
print("开始训练 TD3...")
rng = jax.random.PRNGKey(0)
# Outside the class/JIT
train_state, evaluation = jax.jit(algo.train)(rng=rng)

# Log the results collected during the scan
returns, lengths = evaluation
for i, (r, l) in enumerate(zip(returns, lengths)):
    step = (i + 1) * algo.eval_freq
    wandb.log({
        "eval/return": np.mean(r),
        "eval/length": np.mean(l),
    }, step=step)

wandb.finish()
