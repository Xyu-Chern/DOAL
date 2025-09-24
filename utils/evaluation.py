from collections import defaultdict

import jax
import numpy as np
from tqdm import trange
from jax import random

def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()

        done = False
        step = 0
        render = []
        while not done:
            action = actor_fn(observations=observation, temperature=eval_temperature)
            action = np.array(action)
            action = np.clip(action, -1, 1)
            # print(action)

            next_observation, reward, terminated, truncated, info = env.step(action)
            # print("reward :" , reward)
            # assert False
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders


import numpy as np
import jax
from collections import defaultdict
from tqdm import trange
import copy

def  norm_obs(obs, obs_statistics = (0, 1, np.inf),):
    obs_mean, obs_std, obs_clip = obs_statistics
    return np.clip((obs - obs_mean) / (obs_std + 1e-6), -obs_clip, obs_clip)

def unnorm_act(act, act_statistics):
    act_mean, act_std, act_clip = act_statistics
    return np.clip( (act + act_mean) * act_std, -act_clip,act_clip)

def evaluate_parallel(
    agent,
    envs, # this is a list now
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    obs_statistics=(0, 1, np.inf),
    act_statistics=None,
    fix_seed =False,
):
    actor_fn = jax.vmap(agent.sample_actions,in_axes =(0,0,None))
    total_episodes = num_eval_episodes + num_video_episodes

    trajs = []

    renders = []
    if fix_seed:
        outs = [env.reset(seed = i % 50) for i, env in enumerate(envs)]
    else:
        outs = [env.reset() for env in envs]
    next_observations = [obs for obs, _ in outs]
    rewards = [ 0.0 for i in range(total_episodes)]
    step = 0

    render_list = [ [] for i in range(total_episodes)]
    should_render_lists = [i >= num_eval_episodes for i in range(total_episodes)]
    traj_list = [ defaultdict(list) for i in range(total_episodes) ]

    stats = defaultdict(list) 
    key = jax.random.PRNGKey(np.random.randint(0, 2**32))

    while len(envs)>0:

        observations = np.stack(next_observations, axis=0)
        # print("obs shape :", observations.shape)
        # observations = norm_obs(observations , obs_statistics)
        # print("observations :", observations )

        key,random_key = random.split(key)
        random_keys = random.split(random_key,len(envs))
        actions = actor_fn(observations,random_keys, eval_temperature)
        # print("act shape :", actions.shape)
        actions = np.array(actions)
        actions = np.clip(actions, -1, 1)
        # print("act shape :", actions.shape)
        # print("act :", actions)
        # assert False
        

        # if act_statistics is not None:
        #     actions = unnorm_act(actions, act_statistics)

        out = [ envs[i].step(actions[i]) for i in range(len(envs))]

        next_observations = []
        next_envs = []
        next_traj_list = []

        for i, (next_observation, reward, terminated, truncated, info) in enumerate(out):
            rewards[i] += reward
            # print("reward :" , reward)
            done = terminated or truncated
            step += 1

            if should_render_lists[i] and (step % video_frame_skip == 0 or done):
                frame = envs[i].render().copy()
                render_list[i].append(frame)

            transition = dict(
                observation = observations[i],
                next_observation = next_observation,
                action = actions[i],
                reward = reward,
                done = done,
                info = info,
            )
            add_to(traj_list[i], transition)

            if done:
                if len(renders) < num_video_episodes:
                    renders.append(np.array(render_list[i]))
                else:
                    add_to(stats, flatten(info))
                    trajs.append(traj_list[i])
            if not done:
                next_observations.append(next_observation)
                next_envs.append(envs[i])
                next_traj_list.append(traj_list[i])

        envs = next_envs
        traj_list = next_traj_list

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders




