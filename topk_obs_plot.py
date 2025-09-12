import os
import platform
import json
import random
import time
import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.hps import hyperparameters
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import evaluate, evaluate_parallel, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb
import re
import matplotlib.pyplot as plt
from jax import vmap
import jax.numpy as jnp

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-single-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('agent_name', "diql", 'Agent name.')
flags.DEFINE_string('exp_name', "", 'extra experiment name.')
flags.DEFINE_string('save_dir', '/home/bml/storage/exp/', 'Save directory.')
flags.DEFINE_string('restore_path', "/home/bml/storage/exp/fql/Debug/for_draw", 'Restore path')
flags.DEFINE_integer('restore_epoch', 1000000, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_float('alpha',-1, 'coffeient for conservative')
flags.DEFINE_float('gn',-1, 'coffeient for conservative')
flags.DEFINE_float('expectile',-1, 'coffeient for conservative')
flags.DEFINE_float('alpha_actor',-1, 'coffeient for conservative') 
flags.DEFINE_float('distill_factor',-1, 'coffeient for conservative') 
flags.DEFINE_string('solver',None, 'coffeient for conservative') 
flags.DEFINE_boolean('time_weight', None , 'coffeient for conservative')
flags.DEFINE_boolean('normalize_q_loss', False, 'coffeient for conservative')
flags.DEFINE_string('decode_type', None, 'coffeient for conservative')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')
flags.DEFINE_integer('num_neighbors', 8,  'Number of nearest neighbors to consider.')
flags.DEFINE_integer('candidate_size', 10000, 'Number of candidate observations to sample.')

def find_obs_with_most_variable_actions(all_obs, all_actions, num_candidates=1000, num_neighbors=100):
    """
    找到这样一个obs：它的最近num_neighbors个邻居的action方差最大
    """
    # 随机选择一些候选obs
    candidate_indices = np.random.choice(len(all_obs), size=min(num_candidates, len(all_obs)), replace=False)
    candidate_obs = all_obs[candidate_indices]
    
    print(f"从 {len(all_obs)} 个obs中随机选择 {len(candidate_indices)} 个候选obs")
    
    best_variance = -1
    best_obs_idx = -1
    best_neighbor_indices = None
    best_distances = None
    
    # 对每个候选obs，计算其邻居的action方差
    for i, candidate_ob in enumerate(tqdm.tqdm(candidate_obs, desc="Scanning candidates")):
        # 计算所有obs与当前候选obs的距离
        distances = np.linalg.norm(all_obs - candidate_ob, axis=1)
        
        # 找到最近的num_neighbors个邻居
        neighbor_indices = np.argpartition(-distances, -num_neighbors)[-num_neighbors:]  
        neighbor_actions = all_actions[neighbor_indices]
        
        distance = distances[neighbor_indices]
        # 计算这些邻居actions的方差
        action_variance = np.mean(np.var(neighbor_actions, axis=0))
        
        # 更新最佳候选
        if action_variance > best_variance and np.mean(distance)< 0.8:
            best_variance = action_variance
            best_obs_idx = candidate_indices[i]
            best_neighbor_indices = neighbor_indices
            best_distances = distances[neighbor_indices]
    
    print(f"找到最佳obs (索引 {best_obs_idx})，其邻居action平均方差: {best_variance:.6f}")
    print(f"最近邻居距离范围: [{best_distances.min():.6f}, {best_distances.max():.6f}]")
    
    return best_obs_idx, best_neighbor_indices, best_distances

def compute_actions_for_neighbors_vmap(neighbor_obs, neighbor_data_actions, agent, seed):
    """使用vmap批量计算adjusted和sampled actions以及Q值"""
    
    @vmap
    def compute_adjusted_action(obs, data_action):
        obs_batch = obs.reshape(1, -1)
        action_batch = data_action.reshape(1, -1)
        
        adjusted_actions, adjustment, hd, g, q = agent.get_guided_action(
            action_batch, action_batch, obs_batch,
            alpha=agent.config["alpha"],
            delta=agent.config["delta"],
            params=agent.network.params
        )
        return adjusted_actions[0], q[0]
    
    @vmap
    def compute_sampled_action(obs, seed_offset):
        obs_batch = obs.reshape(1, -1)
        key = jax.random.PRNGKey(seed + seed_offset)
        action = agent.sample_actions(observations=obs_batch, temperature=0, seed=key)
        return jnp.array(action)[0]
    
    @vmap
    def compute_q_value(obs, action):
        obs_batch = obs.reshape(1, -1)
        action_batch = action.reshape(1, -1)
        # 使用双Q网络结构
        q1, q2 = agent.network.select('critic')(obs, actions=action, params=agent.network.params)
        # 取最小值作为保守Q值估计
        q = jnp.minimum(q1, q2)
        print("q : ", q)
        return q
    
    # 批量计算所有邻居的adjusted和sampled actions
    neighbor_adjusted_actions, neighbor_adjusted_qs = compute_adjusted_action(neighbor_obs, neighbor_data_actions)
    neighbor_sampled_actions = compute_sampled_action(neighbor_obs, jnp.arange(len(neighbor_obs)))
    
    # 计算数据动作和采样动作的Q值
    neighbor_data_qs = compute_q_value(neighbor_obs, neighbor_data_actions)
    neighbor_sampled_qs = compute_q_value(neighbor_obs, neighbor_sampled_actions)
    
    return (neighbor_adjusted_actions, neighbor_sampled_actions, 
            neighbor_adjusted_qs, neighbor_data_qs, neighbor_sampled_qs)

def main(_):
    # Set up logger.
    config_flags.DEFINE_config_file('agent', f'agents/{FLAGS.agent_name}.py', lock_config=False)
    config = FLAGS.agent

     
    pattern = r"-task\d-"

    # The replacement string
    replacement = "-"

    # Use re.sub() to perform the substitution
    env_class = re.sub(pattern, replacement, FLAGS.env_name )
    if env_class in hyperparameters:
        for k, v in hyperparameters[env_class].items():
            if not isinstance(v,dict):
                config[k] = v
                print ("update",k,v)
    if env_class in hyperparameters and config['agent_name'] in hyperparameters[env_class]:
        config.update(hyperparameters[env_class][config['agent_name']])
        print ("update",hyperparameters[env_class][config['agent_name']])

        
    exp_name = FLAGS.exp_name 
    if FLAGS.alpha != -1:
        config["alpha"] = FLAGS.alpha
        exp_name +=  "_alpha _" + str(config["alpha"])
    if FLAGS.expectile != -1:
        config["expectile"] = FLAGS.expectile
        exp_name +=  "_expectile _" + str(config["expectile"])
    if FLAGS.gn != -1:
        config["gn"] = FLAGS.gn
        exp_name +=  "_gn _" + str(config["gn"])
    elif env_class in hyperparameters and "alpha" in hyperparameters[env_class] :
        alpha = hyperparameters[env_class]["alpha"]
        config.update({"alpha":alpha})
        print("env alpha is ", alpha)
    if FLAGS.alpha_actor != -1:
        config["alpha_actor"] = FLAGS.alpha_actor
        exp_name +=  "_alpha_actor_" + str(config["alpha_actor"])
    if FLAGS.solver is not None:
        config["solver"] = FLAGS.solver
        exp_name +=  "_solver_" + str(config["solver"])
    if FLAGS.distill_factor != -1:
        config["distill_factor"] = FLAGS.distill_factor
        exp_name +=  "_distill_factor_" + str(config["distill_factor"])  
    if FLAGS.normalize_q_loss:
        config['normalize_q_loss'] = FLAGS.normalize_q_loss
    if FLAGS.time_weight is not None:
        config['time_weight'] = FLAGS.time_weight
        print ("time_weight", FLAGS.time_weight)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, "fql", FLAGS.run_group, exp_name)

    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Make environment and datasets.
    env, envs, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=FLAGS.frame_stack,eval_episodes=FLAGS.eval_episodes + FLAGS.video_episodes)
    if FLAGS.video_episodes > 0:
        assert 'singletask' in FLAGS.env_name, 'Rendering is currently only supported for OGBench environments.'
    if FLAGS.online_steps > 0:
        assert 'visual' not in FLAGS.env_name, 'Online fine-tuning is currently not supported for visual environments.'

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Set up datasets.
    train_dataset = Dataset.create(**train_dataset)

    if FLAGS.balanced_sampling:
        # Create a separate replay buffer so that we can sample from both the training dataset and the replay buffer.
        example_transition = {k: v[0] for k, v in train_dataset.items()}
        replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)
    else:
        # Use the training dataset as the replay buffer.
        train_dataset = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
        )
        replay_buffer = train_dataset
    # Set p_aug and frame_stack.
    for dataset in [train_dataset, val_dataset, replay_buffer]:
        if dataset is not None:
            dataset.p_aug = FLAGS.p_aug
            dataset.frame_stack = FLAGS.frame_stack
            dataset.batch_size = config['batch_size']
            if 'rebrac' in config['agent_name'] or ('return_next_actions' in config and config['return_next_actions'] ):
                dataset.return_next_actions = True

    # Create agent.
    example_batch = train_dataset.sample(1)

    agent_class = agents[config['agent_name']]
    print("agent_name : ", config['agent_name'])
    flag_dict["agent_config"] = config

    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # 采样一个大的batch
    large_batch = train_dataset.sample(10000)  # 采样大量数据
    all_obs = large_batch['observations']
    all_actions = large_batch['actions']
    
    print(f"所有观测值形状: {all_obs.shape}")
    print(f"所有动作形状: {all_actions.shape}")
    
    # 找到action方差最大的obs及其邻居
    best_obs_idx, neighbor_indices, neighbor_distances = find_obs_with_most_variable_actions(
        all_obs, all_actions, 
        num_candidates=FLAGS.candidate_size,
        num_neighbors=FLAGS.num_neighbors
    )
    
    # 获取最佳obs和其邻居
    best_obs = all_obs[best_obs_idx]
    neighbor_obs = all_obs[neighbor_indices]
    neighbor_data_actions = all_actions[neighbor_indices]
    
    print(f"\n最佳obs索引: {best_obs_idx}")
    print(f"最佳obs值: {best_obs[:10]}...")  # 显示前10个值
    print(f"邻居数量: {len(neighbor_indices)}")
    print(f"邻居距离统计 - 最小: {neighbor_distances.min():.6f}, 最大: {neighbor_distances.max():.6f}, 平均: {neighbor_distances.mean():.6f}")
    
    # 使用vmap批量计算adjusted和sampled actions以及Q值
    (neighbor_adjusted_actions, neighbor_sampled_actions, 
     neighbor_adjusted_qs, neighbor_data_qs, neighbor_sampled_qs) = compute_actions_for_neighbors_vmap(
        neighbor_obs, neighbor_data_actions, agent, FLAGS.seed
    )
    
    print ("obs var",np.var(neighbor_obs,axis=0))
    # 转换为numpy数组
    neighbor_data_actions_np = np.array(neighbor_data_actions)
    neighbor_adjusted_actions_np = np.array(neighbor_adjusted_actions)
    neighbor_sampled_actions_np = np.array(neighbor_sampled_actions)
    neighbor_adjusted_qs_np = np.array(neighbor_adjusted_qs)
    neighbor_data_qs_np = np.array(neighbor_data_qs)
    neighbor_sampled_qs_np = np.array(neighbor_sampled_qs)
    
    print(f"\n动作形状:")
    print(f"数据动作: {neighbor_data_actions_np.shape}")
    print(f"调整动作: {neighbor_adjusted_actions_np.shape}")
    print(f"采样动作: {neighbor_sampled_actions_np.shape}")
    
    # 分析方差并选择方差最大的维度
    def select_high_variance_dims(actions, num_dims=2):
        variances = np.var(actions, axis=0)
        top_indices = np.argsort(variances)[-num_dims:]
        return top_indices, variances
    
    dim_indices, variances = select_high_variance_dims(neighbor_data_actions_np)
    print(f"\n选择的维度: {dim_indices}")
    print(f"各维度方差: {variances}")
    print(f"选择的维度方差: {variances[dim_indices]}")
    
    # 分析观测值方差最大的两个维度
    obs_variances = np.var(neighbor_obs, axis=0)
    obs_top_indices = np.argsort(obs_variances)[-2:]
    print(f"\n观测值方差最大的两个维度: {obs_top_indices}")
    print(f"观测值维度方差: {obs_variances[obs_top_indices]}")
    
    # 绘制散点图
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    titles = ['Data Actions', 'Adjusted Actions', 'Sampled Actions']
    colors = ['blue', 'orange', 'green']
    action_sets = [neighbor_data_actions_np, neighbor_adjusted_actions_np, neighbor_sampled_actions_np]
    q_sets = [neighbor_data_qs_np, neighbor_adjusted_qs_np, neighbor_sampled_qs_np]

    # 为每个观测值分配唯一的颜色
    unique_obs_indices = {}
    obs_colors = []
    for j, obs in enumerate(neighbor_obs):
        obs_tuple = tuple(obs.round(2))  # 四舍五入到小数点后两位以便比较
        if obs_tuple not in unique_obs_indices:
            unique_obs_indices[obs_tuple] = len(unique_obs_indices)
        obs_colors.append(unique_obs_indices[obs_tuple])

    # 创建颜色映射
    cmap = plt.cm.get_cmap('tab10', len(unique_obs_indices))

    for i, (ax, title, actions, q_values) in enumerate(zip(axes, titles, action_sets, q_sets)):
        # 计算圆圈大小（基于Q值，Q值越大圆圈越大）
        min_q, max_q = np.min(q_values), np.max(q_values)
        if max_q > min_q:
            sizes = 50 + 200 * (q_values - min_q) / (max_q - min_q)
        else:
            sizes = np.full_like(q_values, 100)
        
        # 绘制散点图，颜色表示观测值，大小表示Q值
        scatter = ax.scatter(actions[:, dim_indices[0]], 
                        actions[:, dim_indices[1]], 
                        c=obs_colors, cmap=cmap, alpha=0.7, s=sizes, label=title)
        
        # 为每个点添加文本标签（观测值方差最大的两个维度和Q值）
        for j, (x, y, q_val, size) in enumerate(zip(actions[:, dim_indices[0]], 
                                                actions[:, dim_indices[1]], 
                                                q_values, sizes)):
            obs_dim1 = neighbor_obs[j, obs_top_indices[0]]
            obs_dim2 = neighbor_obs[j, obs_top_indices[1]]
            ax.annotate(f"O({obs_dim1:.1f},{obs_dim2:.1f})\nQ:{q_val:.2f}", 
                    (x, y), 
                    xytext=(5, 5), 
                    textcoords='offset points', 
                    fontsize=6, 
                    alpha=0.7)
        
        ax.set_title(f'{title} (Top {FLAGS.num_neighbors} neighbors)', fontsize=14)
        ax.set_xlabel(f'Action Dimension {dim_indices[0]}', fontsize=12)
        ax.set_ylabel(f'Action Dimension {dim_indices[1]}', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Observation ID', fontsize=12)
        cbar.set_ticks(range(len(unique_obs_indices)))
        
        # 添加图例说明圆圈大小表示Q值
        ax.text(0.02, 0.98, f'Circle size = Q value\nMin: {min_q:.2f}, Max: {max_q:.2f}', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 设置统一的坐标轴范围
    all_actions_combined = np.concatenate([neighbor_data_actions_np[:, dim_indices],
                                        neighbor_adjusted_actions_np[:, dim_indices],
                                        neighbor_sampled_actions_np[:, dim_indices]], axis=0)

    x_min, x_max = all_actions_combined[:, 0].min(), all_actions_combined[:, 0].max()
    y_min, y_max = all_actions_combined[:, 1].min(), all_actions_combined[:, 1].max()

    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1

    for ax in axes:
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

    plt.suptitle(f'Action Distribution for Obs with Highest Variance (Obs index: {best_obs_idx})\n'
                f'Top Observation Dimensions: {obs_top_indices[0]} and {obs_top_indices[1]}\n'
                f'Color: Same observation, Size: Q value', 
                fontsize=16)
    plt.tight_layout()
    plt.savefig('out/highest_variance_obs_action_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 详细统计分析
    print(f"\n=== 详细统计分析 ===")
    for i, (name, actions, q_values) in enumerate(zip(['Data', 'Adjusted', 'Sampled'], 
                                          [neighbor_data_actions_np, neighbor_adjusted_actions_np, neighbor_sampled_actions_np],
                                          [neighbor_data_qs_np, neighbor_adjusted_qs_np, neighbor_sampled_qs_np])):
        print(f"\n{name} Actions:")
        print(f"  均值: {actions.mean(axis=0)}")
        print(f"  标准差: {actions.std(axis=0)}")
        print(f"  范围: [{actions.min(axis=0)}, {actions.max(axis=0)}]")
        print(f"  Q值范围: [{q_values.min():.4f}, {q_values.max():.4f}], 平均: {q_values.mean():.4f}")
    
    # 计算动作的一致性
    data_consistency = np.mean(np.std(neighbor_data_actions_np, axis=0))
    adjusted_consistency = np.mean(np.std(neighbor_adjusted_actions_np, axis=0))
    sampled_consistency = np.mean(np.std(neighbor_sampled_actions_np, axis=0))
    
    print(f"\n动作一致性（平均标准差）:")
    print(f"  数据动作: {data_consistency:.4f}")
    print(f"  调整动作: {adjusted_consistency:.4f}")
    print(f"  采样动作: {sampled_consistency:.4f}")
    
    # 保存数据以供进一步分析
    np.savez('out/highest_variance_analysis.npz',
             best_obs_index=best_obs_idx,
             best_obs=best_obs,
             neighbor_obs=neighbor_obs,
             neighbor_distances=neighbor_distances,
             data_actions=neighbor_data_actions_np,
             adjusted_actions=neighbor_adjusted_actions_np,
             sampled_actions=neighbor_sampled_actions_np,
             data_qs=neighbor_data_qs_np,
             adjusted_qs=neighbor_adjusted_qs_np,
             sampled_qs=neighbor_sampled_qs_np,
             selected_dims=dim_indices,
             action_variances=variances,
             obs_top_indices=obs_top_indices,
             obs_variances=obs_variances)
    
    print("\n分析数据已保存到 highest_variance_analysis.npz")
    print("包含以下数据:")
    print("- best_obs_index: 最佳obs的索引")
    print("- best_obs: 最佳obs的值")
    print("- neighbor_obs: 邻居obs")
    print("- neighbor_distances: 邻居距离")
    print("- data_actions: 数据动作")
    print("- adjusted_actions: 调整动作")
    print("- sampled_actions: 采样动作")
    print("- data_qs: 数据动作的Q值")
    print("- adjusted_qs: 调整动作的Q值")
    print("- sampled_qs: 采样动作的Q值")
    print("- selected_dims: 选择的维度")
    print("- action_variances: 各维度方差")
    print("- obs_top_indices: 观测值方差最大的维度")
    print("- obs_variances: 观测值各维度方差")

if __name__ == '__main__':
    app.run(main)