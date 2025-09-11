
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

FLAGS = flags.FLAGS


flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-single-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('agent_name', "fql", 'Agent name.')
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

  #  if "normalize_action" in config and config["normalize_action"]:
  #      mean,sigma = train_dataset.get_action_stats()
  #      config["sigma"] = sigma
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
    setup_wandb(project='doal', group=FLAGS.env_name, name=exp_name,config=flag_dict)

    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

# if FLAGS.restore_path is not None:
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)


    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()


    batch = train_dataset.sample(config['batch_size'])
    print("batch :" , batch.keys())
    print("data_action.shape", batch["actions"].shape)
    print("Action space:", env.action_space)


    online_rng = jax.random.PRNGKey(FLAGS.seed)
    online_rng, key = jax.random.split(online_rng)

    action = agent.sample_actions(observations=batch["observations"], temperature=1, seed=key)
    sample_action = np.array(action)

    adjusted_actions , adjustment,hd, g, q = agent.get_guided_action(  batch['actions'], batch['actions'],batch['observations'],alpha=agent.config["alpha"],delta=agent.config["delta"],params=agent.network.params)
    print("sample_action : ", sample_action.shape)
    print( "adjusted_actions : ", adjusted_actions.shape)

    # 转换为numpy数组
    data_actions_np = np.array(batch['actions'])
    adjusted_actions_np = np.array(adjusted_actions)
    sampled_actions_np = np.array(sample_action)

    # 绘制散点图
    selected_dims = plot_action_comparison(
        data_actions_np, adjusted_actions_np, sampled_actions_np,
        save_path='action_comparison_scatter.png'
    )

    # 打印统计信息
    print(f"\n=== 统计信息 ===")
    print(f"选择的维度: {selected_dims}")
    print(f"数据动作范围: [{data_actions_np[:, selected_dims].min():.3f}, {data_actions_np[:, selected_dims].max():.3f}]")
    print(f"调整动作范围: [{adjusted_actions_np[:, selected_dims].min():.3f}, {adjusted_actions_np[:, selected_dims].max():.3f}]")
    print(f"采样动作范围: [{sampled_actions_np[:, selected_dims].min():.3f}, {sampled_actions_np[:, selected_dims].max():.3f}]")
    
    train_logger.close()
    eval_logger.close()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def select_high_variance_dims(actions, num_dims=2):
    """选择方差最大的维度"""
    variances = np.var(actions, axis=0)
    top_indices = np.argsort(variances)[-num_dims:]
    return top_indices, variances

def plot_action_comparison(data_actions, adjusted_actions, sampled_actions, save_path='action_comparison.png'):
    """绘制三种动作的散点图比较"""
    
    # 选择方差最大的两个维度（基于data_actions）
    dim_indices, variances = select_high_variance_dims(data_actions)
    print(f"选择的维度: {dim_indices}")
    print(f"各维度方差: {variances}")
    print(f"选择的维度方差: {variances[dim_indices]}")
    
    # 创建散点图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Data actions
    axes[0].scatter(data_actions[:, dim_indices[0]], 
                   data_actions[:, dim_indices[1]], 
                   alpha=0.7, s=20, color='blue', label='Data Actions')
    axes[0].set_title('Data Actions', fontsize=14)
    axes[0].set_xlabel(f'Dimension {dim_indices[0]}', fontsize=12)
    axes[0].set_ylabel(f'Dimension {dim_indices[1]}', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Adjusted actions
    axes[1].scatter(adjusted_actions[:, dim_indices[0]], 
                   adjusted_actions[:, dim_indices[1]], 
                   alpha=0.7, s=20, color='orange', label='Adjusted Actions')
    axes[1].set_title('Adjusted Actions', fontsize=14)
    axes[1].set_xlabel(f'Dimension {dim_indices[0]}', fontsize=12)
    axes[1].set_ylabel(f'Dimension {dim_indices[1]}', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Sampled actions
    axes[2].scatter(sampled_actions[:, dim_indices[0]], 
                   sampled_actions[:, dim_indices[1]], 
                   alpha=0.7, s=20, color='green', label='Sampled Actions')
    axes[2].set_title('Sampled Actions', fontsize=14)
    axes[2].set_xlabel(f'Dimension {dim_indices[0]}', fontsize=12)
    axes[2].set_ylabel(f'Dimension {dim_indices[1]}', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # 设置统一的坐标轴范围以便比较
    all_data = np.concatenate([data_actions[:, dim_indices], 
                              adjusted_actions[:, dim_indices], 
                              sampled_actions[:, dim_indices]], axis=0)
    
    x_min, x_max = all_data[:, 0].min(), all_data[:, 0].max()
    y_min, y_max = all_data[:, 1].min(), all_data[:, 1].max()
    
    # 添加一些边距
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    
    for ax in axes:
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return dim_indices


if __name__ == '__main__':
    app.run(main)
