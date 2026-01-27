import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
from doal import DOAL

# 简单训练一个智能体（先不要用vmap训练300个）
print("初始化算法...")
algo = DOAL.create(
    env="Pendulum-v1", 
    learning_rate=0.001,
    total_timesteps=200,  # 先小规模测试
    batch_size=256,
    gamma=0.99,
    eval_freq=10_000,
    max_grad_norm=1.0,
    fill_buffer=1_000,
    num_envs=1,
    flow_steps=10,
    max_q_samples=4,
    policy_delay=2,
    alpha=0.2,
    delta=2.0,
    exploration_noise=0.1,
    target_noise=0.2,
    target_noise_clip=0.5,
)

print("开始训练...")
# 训练一个智能体
rng = jax.random.PRNGKey(42)
train_state, evaluation = algo.train(rng=rng)

print("训练完成!")
print(f"评估结果: {evaluation}")