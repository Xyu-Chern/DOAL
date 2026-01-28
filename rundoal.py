# import os
# os.environ['JAX_PLATFORMS'] = 'cpu'

# import jax
# from doal import DOAL

# # 简单训练一个智能体（先不要用vmap训练300个）
# print("初始化算法...")
# algo = DOAL.create(
#     env="Pendulum-v1", 
#     learning_rate=0.001,
#     total_timesteps=200,  # 先小规模测试
#     batch_size=256,
#     gamma=0.99,
#     eval_freq=10_000,
#     max_grad_norm=1.0,
#     fill_buffer=1_000,
#     num_envs=1,
#     flow_steps=10,
#     max_q_samples=4,
#     policy_delay=2,
#     alpha=0.2,
#     delta=2.0,
#     exploration_noise=0.1,
#     target_noise=0.2,
#     target_noise_clip=0.5,
# )

# print("开始训练...")
# # 训练一个智能体
# rng = jax.random.PRNGKey(42)
# train_state, evaluation = algo.train(rng=rng)

# print("训练完成!")
# print(f"评估结果: {evaluation}")


import jax
import wandb

from rejax import PPO


CONFIG = {
    "env": "brax/ant",
    "env_params": {"backend": "positional"},
    "agent_kwargs": {"activation": "relu"},
    "total_timesteps": 10_000_000,
    "eval_freq": 100_000,
    "num_envs": 2_000,
    "num_steps": 5,
    "num_epochs": 4,
    "num_minibatches": 4,
    "learning_rate": 0.0003,
    "max_grad_norm": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
}

wandb.init(project="my-awesome-project", config=CONFIG)

ppo = PPO.create(**CONFIG)
eval_callback = ppo.eval_callback


def wandb_callback(ppo, train_state, rng):
    lengths, returns = eval_callback(ppo, train_state, rng)

    def log(step, data):
        # io_callback returns np.array, which wandb does not like.
        # In jax 0.4.27, this becomes a jax array, should check when upgrading...
        step = step.item()
        wandb.log(data, step=step)

    jax.experimental.io_callback(
        log,
        (),  # result_shape_dtypes (wandb.log returns None)
        train_state.global_step,
        {"episode_length": lengths.mean(), "return": returns.mean()},
    )

    # Since we log to wandb, we don't want to return anything that is collected
    # throughout training
    return ()


ppo = ppo.replace(eval_callback=wandb_callback)

rng = jax.random.PRNGKey(0)
print("Compiling...")
compiled_train = jax.jit(ppo.train).lower(rng).compile()
print("Training...")
compiled_train(rng)