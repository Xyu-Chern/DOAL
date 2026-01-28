import gymnasium as gym

# 测试环境
env = gym.make("Pendulum-v1")
# 获取原始环境（去掉TimeLimit包装）
unwrapped_env = env.unwrapped
print(f"环境: {env}")
print(f"原始环境类型: {type(unwrapped_env)}")

# 对于Pendulum，我们可以直接获取奖励信息
# 或者通过测试来了解奖励范围

# 测试随机动作的奖励
episode_rewards = []
for episode in range(10):  # 测试10个episodes
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
    
    episode_rewards.append(total_reward)
    print(f"Episode {episode}: 总奖励={total_reward:.2f}, 步数={steps}")

env.close()

print(f"\n平均总奖励: {np.mean(episode_rewards):.2f}")
print(f"最小总奖励: {np.min(episode_rewards):.2f}")
print(f"最大总奖励: {np.max(episode_rewards):.2f}")