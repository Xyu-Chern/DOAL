# import ogbench
# import gym

# # 列出所有环境名称
# env_names = [
#     'antmaze-large-navigate-singletask-task1-v0',
#     'antmaze-giant-navigate-singletask-task1-v0', 
#     'humanoidmaze-medium-navigate-singletask-task1-v0',
#     'humanoidmaze-large-navigate-singletask-task1-v0',
#     'antsoccer-arena-navigate-singletask-task4-v0',
#     'cube-single-play-singletask-task2-v0',
#     'cube-double-play-singletask-task2-v0',
#     'scene-play-singletask-task2-v0',
#     'puzzle-3x3-play-singletask-task4-v0',
#     'puzzle-4x4-play-singletask-task4-v0'
# ]

# print("环境名称及其action_dim:")
# print("=" * 50)

# for env_name in env_names:
#     try:
#         # 创建环境和数据集
#         env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name, dataset_dir = "/home/bml/storage/.ogbench/data")
        
#         # 获取action_dim
#         action_dim = env.action_space.shape[0]
        
#         # 打印结果
#         print(f"{env_name}: {action_dim}")
        
#         # 关闭环境（如果支持）
#         env.close()
        
#     except Exception as e:
#         print(f"处理环境 {env_name} 时出错: {str(e)}")
#         continue

# print("=" * 50)
# print("所有环境处理完成！")


import gym
import d4rl  # 导入d4rl以注册环境

# D4RL环境列表
d4rl_env_names = [
    'antmaze-umaze-v2',
    'antmaze-umaze-diverse-v2',
    'antmaze-medium-play-v2',
    'antmaze-medium-diverse-v2',
    'antmaze-large-play-v2',
    'antmaze-large-diverse-v2',
    'pen-human-v1',
    'pen-cloned-v1',
    'pen-expert-v1',
    'door-human-v1',
    'door-cloned-v1',
    'door-expert-v1',
    'hammer-human-v1',
    'hammer-cloned-v1',
    'hammer-expert-v1',
    'relocate-human-v1',
    'relocate-cloned-v1',
    'relocate-expert-v1'
]

print("D4RL环境名称及其action_dim:")
print("=" * 60)

for env_name in d4rl_env_names:
    try:
        # 创建环境
        env = gym.make(env_name)
        
        # 获取action_dim
        if hasattr(env.action_space, 'shape'):
            action_dim = env.action_space.shape[0]
        else:
            # 对于离散动作空间
            action_dim = env.action_space.n
        
        # 获取动作空间类型
        action_space_type = "连续" if hasattr(env.action_space, 'shape') else "离散"
        
        # 打印结果
        print(f"{env_name:<35}: {action_dim:>3} ({action_space_type}动作空间)")
        
        # 关闭环境
        env.close()
        
    except Exception as e:
        print(f"处理环境 {env_name} 时出错: {str(e)}")
        continue

print("=" * 60)
print("所有D4RL环境处理完成！")

# 如果需要同时处理之前的ogbench环境和D4RL环境，可以使用以下代码
print("\n\n所有环境汇总:")
print("=" * 60)