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
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

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
    flag_dict["agent_config"] = config

    setup_wandb(project='doal', group=FLAGS.env_name, name=exp_name,config=flag_dict)

   # artifact = wandb.Artifact(name="agent", type="code")
   # artifact.add_file(f'agents/{FLAGS.agent_name}.py')
   # wandb.log_artifact(artifact)
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    step = 0
    done = True
    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(FLAGS.seed)
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + FLAGS.online_steps + 1), smoothing=0.1, dynamic_ncols=True):
        if i <= FLAGS.offline_steps:
            # Offline RL.
            batch = train_dataset.sample(config['batch_size'])

            if 'rebrac' in config['agent_name'] :
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info = agent.update(batch)
        else:
            # Online fine-tuning.
            online_rng, key = jax.random.split(online_rng)

            if done:
                step = 0
                ob, _ = env.reset()

            action = agent.sample_actions(observations=ob, temperature=1, seed=key)
            action = np.array(action)

            next_ob, reward, terminated, truncated, info = env.step(action.copy())
            done = terminated or truncated

            if 'antmaze' in FLAGS.env_name and (
                'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
            ):
                # Adjust reward for D4RL antmaze.
                reward = reward - 1.0

            replay_buffer.add_transition(
                dict(
                    observations=ob,
                    actions=action,
                    rewards=reward,
                    terminals=float(done),
                    masks=1.0 - terminated,
                    next_observations=next_ob,
                )
            )
            ob = next_ob

            if done:
                expl_metrics = {f'exploration/{k}': np.mean(v) for k, v in flatten(info).items()}

            step += 1

            # Update agent.
            if FLAGS.balanced_sampling:
                # Half-and-half sampling from the training dataset and the replay buffer.
                dataset_batch = train_dataset.sample(config['batch_size'] // 2)
                replay_batch = replay_buffer.sample(config['batch_size'] // 2)
                batch = {k: np.concatenate([dataset_batch[k], replay_batch[k]], axis=0) for k in dataset_batch}
            else:
                batch = replay_buffer.sample(config['batch_size'])

            if 'rebrac' in config['agent_name']:
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            renders = []
            eval_metrics = {}
            eval_info, trajs, cur_renders = evaluate_parallel(
                agent=agent,
                envs = envs,
                config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            renders.extend(cur_renders)
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)
        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
