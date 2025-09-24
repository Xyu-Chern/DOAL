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
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-single-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('exp_name', "", 'extra experiment name.')
flags.DEFINE_string('save_dir', '../exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_boolean('restore', False, 'Restore path.')
flags.DEFINE_integer('restore_epoch', 0, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 500, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 1000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 100, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')


config_flags.DEFINE_config_file('agent', f'agents/dtrigflow.py', lock_config=False)

flags.DEFINE_float('alpha_actor',None, 'coffeient for conservative') 
flags.DEFINE_float('alpha',None, 'coffeient for conservative') 
flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')
import jax.numpy as jnp
def main(_):   #num_samples
    # Set up logger.

    if FLAGS.restore:
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
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
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, "doal", FLAGS.run_group, exp_name,FLAGS.env_name,config['agent_name'],str(FLAGS.seed))

    flag_dict = get_flag_dict()
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    for key, value in flag_dict.items():
        if key in config and value is not None :
            config[key] = value
            print (key+" is updated to be "+ str(config[key]))

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

   # artifact = wandb.Artifact(name="agent", type="code")
   # artifact.add_file(f'agents/{FLAGS.agent_name}.py')
   # wandb.log_artifact(artifact)
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )
   # print ("config",config)
    # Restore agent.

    if FLAGS.restore :
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        print ("FLAGS.save_dir",FLAGS.save_dir)
        eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 're_eval.csv'))
      #  jax.disable_jit()
        restored_agent = restore_agent(agent, FLAGS.save_dir, FLAGS.restore_epoch)
        agent = restored_agent.replace(config=agent.config)

        for num_samples in [1,2,4,8,16,32]:
            config = agent.config.copy({"num_samples":num_samples})
            agent = agent.replace(config=config)
            eval_metrics = {}
            renders = []
            eval_info, trajs, cur_renders = evaluate_parallel(
                agent=agent,
                envs = envs,
                config=config,
                num_eval_episodes=100,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            renders.extend(cur_renders)
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v

            print (num_samples, eval_metrics["evaluation/success"])
            eval_logger.log(eval_metrics, step=num_samples)
        assert False 
    setup_wandb(project='doal', group=FLAGS.run_group, name=exp_name,config=flag_dict)
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    # Train agent.
    first_time = time.time()
    last_time = time.time()

    step = 0
    done = True
    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(FLAGS.seed)

    dataset = train_dataset._dict
    data_size = dataset["masks"].shape[0] 
    log_interval = 5000
    n_complete_batches = 100000  // log_interval #data_size // config['batch_size']
    truncated_size = n_complete_batches * config['batch_size']

    print (f"Dataset size: {data_size}")
    print (f"Truncated dataset size: {truncated_size}")
    print (f"Num complete batches per epoch: {n_complete_batches}")
    num_epochs = FLAGS.offline_steps // n_complete_batches
    print (f"Num epochs: {num_epochs}")
    print (f"Num effective training steps: {num_epochs * n_complete_batches}")

    @jax.jit
    def scan_update(agent, batch):
   #     jax.debug.print("before {bar}", bar=str(jax.tree_util.tree_map(lambda x: x.dtype, batch)))
        batch = jax.tree_util.tree_map(lambda x: jnp.array(x), batch)
    #    jax.debug.print("after {bar}", bar=str(jax.tree_util.tree_map(lambda x: x.dtype, batch)))
        agent, info = agent.update(batch)
        return agent, info

    pbar = tqdm.tqdm(range(1, num_epochs + 1), smoothing=0.1, dynamic_ncols=True)
    #for i in tqdm.tqdm(range(1, FLAGS.offline_steps + FLAGS.online_steps + 1), smoothing=0.1, dynamic_ncols=True):
    rng = jax.random.PRNGKey(FLAGS.seed)
    for i in pbar:
        # Generate new random key for shuffling
        rng, subkey = jax.random.split(rng)

        before_shuffle = time.time() 
        
        batches = train_dataset.sample(truncated_size)
        
        # Shuffle and reshape dataset into batches
        batches = jax.tree_util.tree_map(
            lambda x: x.reshape(-1, config['batch_size'], *x.shape[1:]),
            batches
        )
        after_shuffle = time.time() 
        # Perform updates using scan over all batches
        agent, update_info = jax.jit(jax.lax.scan,static_argnums=(0,))(
            scan_update,
            agent,
            batches
        )
        # Log metrics.

        if i % log_interval == 0 or i == num_epochs:
            eval_metrics = {}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                eval_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            renders = []
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

            wandb.log(eval_metrics, step=i*n_complete_batches)
            eval_logger.log(eval_metrics, step=i*n_complete_batches)
            save_agent(agent, FLAGS.save_dir, 0)
            pbar.set_postfix({k.split('/')[-1]: f"{v:.1f}" for k, v in eval_metrics.items()})
        if i % 100 == 0:
            update_info = jax.tree_util.tree_map(
                lambda xs: jnp.mean(xs), 
                update_info
            )
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics['time/data_time'] = after_shuffle- before_shuffle
            train_metrics['time/compute_time'] = time.time() - after_shuffle
            train_metrics['time/total_time'] = (time.time() - last_time) / log_interval
            last_time = time.time()
            wandb.log(train_metrics, step=i*n_complete_batches)
            train_logger.log(train_metrics, step=i*n_complete_batches)
            pbar.set_postfix({k.split('/')[-1]: f"{v:.1f}" for k, v in train_metrics.items()})
        # Save agent.

    train_logger.close()
    eval_logger.close()

    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 're_eval.csv'))
    #  jax.disable_jit()
    restored_agent = restore_agent(agent, FLAGS.save_dir, FLAGS.restore_epoch)
    agent = restored_agent.replace(config=agent.config)

    for num_samples in [2,4,8,16]:
        config = agent.config.copy({"num_samples":num_samples})
        agent = agent.replace(config=config)
        eval_metrics = {}
        renders = []
        eval_info, trajs, cur_renders = evaluate_parallel(
            agent=agent,
            envs = envs,
            config=config,
            num_eval_episodes=100,
            num_video_episodes=FLAGS.video_episodes,
            video_frame_skip=FLAGS.video_frame_skip,
        )
        renders.extend(cur_renders)
        for k, v in eval_info.items():
            eval_metrics[f'evaluation/{k}'] = v

        print (num_samples, eval_metrics["evaluation/success"])
        eval_logger.log(eval_metrics, step=num_samples)
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
