import chex
import jax
import numpy as np
import optax
import dataclasses
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from jax import numpy as jnp
import wandb

from typing import Sequence, Callable

from rejax.algos.algorithm import Algorithm, register_init
from rejax.algos.mixins import (
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    VectorizedEnvMixin,
    ReplayBufferMixin,
    TargetNetworkMixin,
)

from rejax.buffers import Minibatch
from functools import partial

import chex
import jax
import numpy as np
from flax import struct
from jax import numpy as jnp
from optax import linear_schedule

from rejax.algos.algorithm import register_init
from rejax.buffers import ReplayBuffer

from typing import NamedTuple, Union

from gymnax.environments import spaces
class SarsaMinibatch(NamedTuple):
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    next_action: chex.Array


class SarsaReplayBuffer(ReplayBuffer):
    """
    Circular buffer for storing transitions. Implements appending and sampling
    while being `jit`-able.
    """

    data: SarsaMinibatch

    @classmethod
    def empty(
        cls,
        size: int,
        obs_space: Union[spaces.Discrete, spaces.Box],
        action_space: Union[spaces.Discrete, spaces.Box],
    ) -> "SarsaReplayBuffer":
        """Returns an empty replay buffer with the given size and shapes.

        Args:
            size (int): Maximum number of transitions to store.
            obs_shape (chex.Shape): Shape of the observations.
            action_shape (chex.Shape): Shape of the actions.

        Returns:
            ReplayBuffer: The initialized replay buffer.
        """
        # Skip checking sizes as we know they are correct here
        data = SarsaMinibatch(
            obs=jnp.empty((size, *obs_space.shape)).astype(obs_space.dtype),
            action=jnp.empty((size, *action_space.shape)).astype(action_space.dtype),
            reward=jnp.empty(size),
            done=jnp.empty(size).astype(bool),
            next_obs=jnp.empty((size, *obs_space.shape)).astype(obs_space.dtype),
            next_action=jnp.empty((size, *action_space.shape)).astype(action_space.dtype),
        )
        return cls(size=size, data=data, index=0, full=False)



class SarsaReplayBufferMixin(ReplayBufferMixin):

    @register_init
    def initialize_replay_buffer(self, rng):
        buf = SarsaReplayBuffer.empty(self.buffer_size, self.obs_space, self.action_space)
        return {"replay_buffer": buf}


class FlowMLP(nn.Module):
    action_dim: int
    hidden_layer_sizes: Sequence[int]
    activation: Callable
    action_range: tuple = None  # 添加这个字段

    @nn.compact
    def __call__(self, obs, x, t):
        if t.ndim == x.ndim - 1:
            t = t[..., None]
            
        inputs = jnp.concatenate([obs, x, t], axis=-1)
        
        y = inputs
        for size in self.hidden_layer_sizes:
            y = nn.Dense(size)(y)
            y = self.activation(y)
        
        return nn.Dense(self.action_dim)(y)
    

from rejax.networks import QNetwork


class DOALSARSA(
    SarsaReplayBufferMixin,
    TargetNetworkMixin,
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    Algorithm,
):
    actor: nn.Module = struct.field(pytree_node=False, default=None)
    critic: nn.Module = struct.field(pytree_node=False, default=None)
    num_critics: int = struct.field(pytree_node=False, default=2)
    num_epochs: int = struct.field(pytree_node=False, default=1)
    exploration_noise: chex.Scalar = struct.field(pytree_node=True, default=0.3)
    target_noise: chex.Scalar = struct.field(pytree_node=True, default=0.2)
    target_noise_clip: chex.Scalar = struct.field(pytree_node=True, default=0.5)
    alpha: chex.Scalar = struct.field(pytree_node=True, default=0.2)
    delta: chex.Scalar = struct.field(pytree_node=True, default=2.0)
    flow_steps: int = struct.field(pytree_node=False, default=10)
    max_q_samples: int = struct.field(pytree_node=False, default=4)
    policy_delay: int = struct.field(pytree_node=False, default=2)

    @jax.jit
    def auto(self, ts, minibatch):
        def bc_loss_wrt_q_action(q_action):
            qs = self.vmap_critic(ts.critic_ts.params, minibatch.obs, q_action)
            q = jnp.mean(qs, axis=0)
            return jnp.sum(q)

        v_grad_q = jax.value_and_grad(bc_loss_wrt_q_action) 
        q, g = v_grad_q(minibatch.action)

        norm = jnp.linalg.norm(g, axis=-1, keepdims=True) + 1e-5
        norm_mean = jnp.mean(norm)
        norm_std = jnp.std(norm)
        norm_up = norm_mean + self.delta * norm_std

        clipped_g = jnp.where(norm > norm_up, g * norm_up / norm, g)
        dx = (self.alpha / norm_mean) * clipped_g 
        adjusted_actions = minibatch.action + dx
        
        # 使用 self.action_space.low/high 而不是 self.actor.action_range
        low, high = self.action_space.low, self.action_space.high
        adjusted_actions = jnp.clip(adjusted_actions, low, high)
        
        adjusted_actions = jax.lax.stop_gradient(adjusted_actions)
        return adjusted_actions

    def _sample_actions(self, params, obs, rng, num_samples=1):
        batch_size = obs.shape[0]
        action_dim = self.actor.action_dim
        
        # Initial noise
        rng, key_x0 = jax.random.split(rng)
        x = jax.random.normal(key_x0, (batch_size, num_samples, action_dim))
        
        # Expand obs: (batch, obs_dim) -> (batch, num_samples, obs_dim)
        obs_expanded = jnp.repeat(obs[:, None, :], num_samples, axis=1)
        
        dt = 1.0 / self.flow_steps
        
        def body_fn(i, x):
            t_val = i / self.flow_steps
            t = jnp.full((batch_size, num_samples, 1), t_val)
            vel = self.actor.apply(params, obs_expanded, x, t)
            return x + vel * dt

        x = jax.lax.fori_loop(0, self.flow_steps, body_fn, x)
        
        # Clip to action range
        low, high = self.action_space.low, self.action_space.high
        x = jnp.clip(x, low, high)
        return x

    def make_act(self, ts):
        def act(obs, rng):
            if self.normalize_observations:
                obs = self.normalize_obs(ts.obs_rms_state, obs)

            # obs is (obs_dim,), expand to (1, obs_dim)
            obs = jnp.expand_dims(obs, 0)
            
            # Sample multiple actions using flow
            # (1, num_samples, action_dim)
            actions = self._sample_actions(ts.actor_ts.params, obs, rng, num_samples=self.max_q_samples)
            
            # Score actions with critic
            # Critic expects (batch, obs_dim) and (batch, action_dim)
            # We have (1, num_samples, action_dim).
            # Expand obs to (1, num_samples, obs_dim)
            obs_expanded = jnp.repeat(obs[:, None, :], self.max_q_samples, axis=1)
            
            # Flatten for critic: (num_samples, obs_dim)
            obs_flat = obs_expanded.reshape(-1, obs.shape[-1])
            actions_flat = actions.reshape(-1, actions.shape[-1])
            
            # Evaluate Q-values
            qs = self.vmap_critic(ts.critic_ts.params, obs_flat, actions_flat)
            # qs is (num_critics, num_samples)
            
            # Min over critics
            q_min = jnp.min(qs, axis=0) # (num_samples,)
            
            # Pick best action
            best_idx = jnp.argmax(q_min)
            best_action = actions_flat[best_idx]
            
            return best_action

        return act

    @classmethod
    def create_agent(cls, config, env, env_params):
        actor_kwargs = config.pop("actor_kwargs", {})
        activation = actor_kwargs.pop("activation", "swish")
        actor_kwargs["activation"] = getattr(nn, activation)
        
        action_space = env.action_space(env_params)
        action_dim = np.prod(action_space.shape)
        action_range = (action_space.low, action_space.high)
        
        # Use FlowMLP instead of DeterministicPolicy
        actor = FlowMLP(
            action_dim=action_dim, 
            hidden_layer_sizes=(64, 64), 
            action_range=action_range,  # 传递动作范围
            **actor_kwargs
        )

        critic_kwargs = config.pop("critic_kwargs", {})
        activation = critic_kwargs.pop("activation", "swish")
        critic_kwargs["activation"] = getattr(nn, activation)
        critic = QNetwork(hidden_layer_sizes=(64, 64), **critic_kwargs)

        return {"actor": actor, "critic": critic}
    

    @register_init
    def initialize_env_state(self, rng):
        rng, rng_action = jax.random.split(rng)
        state = super().initialize_env_state(rng)
        
        # Initialize last_action
        sample_fn = self.env.action_space(self.env_params).sample
        last_action = jax.vmap(sample_fn)(jax.random.split(rng_action, self.num_envs))
        
        state["last_action"] = last_action
        return state

    @register_init
    def initialize_network_params(self, rng):
        rng, rng_actor, rng_critic = jax.random.split(rng, 3)
        rng_critic = jax.random.split(rng_critic, self.num_critics)
        obs_ph = jnp.empty((1, *self.env.observation_space(self.env_params).shape))
        action_ph = jnp.empty((1, *self.env.action_space(self.env_params).shape))
        t_ph = jnp.empty((1, 1))

        tx = optax.chain(
            optax.clip(self.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate),
        )

        # Initialize FlowMLP with (obs, x, t)
        actor_params = self.actor.init(rng_actor, obs_ph, action_ph, t_ph)
        actor_ts = TrainState.create(apply_fn=(), params=actor_params, tx=tx)

        vmap_init = jax.vmap(self.critic.init, in_axes=(0, None, None))
        critic_params = vmap_init(rng_critic, obs_ph, action_ph)
        critic_ts = TrainState.create(apply_fn=(), params=critic_params, tx=tx)
        return {
            "actor_ts": actor_ts,
            "actor_target_params": actor_params,
            "critic_ts": critic_ts,
            "critic_target_params": critic_params,
        }

    @property
    def vmap_critic(self):
        return jax.vmap(self.critic.apply, in_axes=(0, None, None))

    def train(self, rng=None, train_state=None):
        if train_state is None and rng is None:
            raise ValueError("Either train_state or rng must be provided")

        ts = train_state or self.init_state(rng)

        # 使用JAX的调试工具来处理日志
        if not self.skip_initial_evaluation:
            initial_evaluation = self.eval_callback(self, ts, ts.rng)
            
            # 使用jax.debug.callback替代host_callback
            if initial_evaluation is not None and len(initial_evaluation) >= 2:
                returns, lengths = initial_evaluation[0], initial_evaluation[1]
                
                # 记录初始评估
                jax.debug.callback(
                    lambda r, l, a: self._log_initial_eval(r, l, 0, a),
                    returns, lengths, self.alpha
                )

        def eval_iteration(ts, unused):
            # Run a few training iterations
            steps_per_train_it = self.num_envs * self.policy_delay
            num_train_its = np.ceil(self.eval_freq / steps_per_train_it).astype(int)
            ts = jax.lax.fori_loop(
                0,
                num_train_its,
                lambda _, ts: self.train_iteration(ts),
                ts,
            )

            # Run evaluation
            eval_result = self.eval_callback(self, ts, ts.rng)
            
            # 记录评估结果
            if eval_result is not None and len(eval_result) >= 2:
                returns, lengths = eval_result[0], eval_result[1]
                current_step = ts.global_step
                
                jax.debug.callback(
                    lambda step, r, l, a: self._log_eval_to_wandb(step, r, l, a),
                    current_step, returns, lengths, self.alpha
                )
            
            return ts, eval_result

        ts, evaluation = jax.lax.scan(
            eval_iteration,
            ts,
            None,
            np.ceil(self.total_timesteps / self.eval_freq).astype(int),
        )

        if not self.skip_initial_evaluation:
            evaluation = jax.tree.map(
                lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
                initial_evaluation,
                evaluation,
            )
        
        # 记录最终结果
        if evaluation is not None and len(evaluation) >= 2:
            # 取最后一次评估的结果
            if evaluation[0].shape[0] > 0:
                final_returns = evaluation[0][-1] if len(evaluation[0].shape) > 1 else evaluation[0]
                final_lengths = evaluation[1][-1] if len(evaluation[1].shape) > 1 else evaluation[1]
                
                jax.debug.callback(
                    lambda r, l, a: self._log_final_results(r, l, a),
                    final_returns, final_lengths, self.alpha
                )

        return ts, evaluation




    def _log_final_results(self, returns, lengths, alpha=None):
        """记录最终结果到WandB"""
        if returns.size > 0:
            if returns.ndim > 1: # Batched / Tuning case
                # returns: (num_alphas, num_seeds, num_episodes) or (num_seeds, num_episodes)
                # alpha: (num_alphas, num_seeds) or similar
                
                # Check if alpha is batched
                alpha = np.array(alpha)
                if alpha.ndim > 0:
                    # Assume we are tuning multiple alphas
                    # If double vmap: alpha shape (num_alphas, num_seeds)
                    # returns shape (num_alphas, num_seeds, num_episodes)
                    
                    # We want to iterate over the FIRST dimension (alphas)
                    # But if we have (num_alphas, num_seeds), alpha[i,0] is the alpha value
                    
                    # Handle case where returns is (num_alphas, num_seeds, num_episodes)
                    if returns.ndim == 3:
                        num_alphas = returns.shape[0]
                        for i in range(num_alphas):
                            curr_alpha = float(alpha[i].flat[0]) # Take first seed's alpha
                            alpha_returns = returns[i] # (num_seeds, num_episodes)
                            
                            # Mean over episodes first
                            episode_means = np.mean(alpha_returns, axis=-1) # (num_seeds,)
                            
                            final_return = float(np.mean(episode_means))
                            
                            print(f"Final evaluation (alpha={curr_alpha}): return={final_return:.2f}")
                            # Final logs are handled by main script analysis usually, but we can log here too
                    else:
                        # Maybe just seeds vmap?
                        pass
                
            else:
                final_return = float(np.mean(returns))
                final_length = float(np.mean(lengths))
                wandb.log({
                    "final/return": final_return,
                    "final/length": final_length
                })
                wandb.summary["final_return"] = final_return
                print(f"Final evaluation: return={final_return:.2f}")

    def _log_initial_eval(self, returns, lengths, step, alpha=None):
        """记录初始评估到WandB"""
        self._log_eval_to_wandb(step, returns, lengths, alpha, prefix="initial")

    def _log_eval_to_wandb(self, step, returns, lengths, alpha=None, prefix="eval"):
        """记录评估结果到WandB"""
        if returns.size > 0:
            # Handle step
            current_step = int(np.mean(step)) if np.ndim(step) > 0 else int(step)
            
            # Check if batched (Tuning case)
            if returns.ndim >= 3: # (num_alphas, num_seeds, num_episodes)
                # alpha should be (num_alphas, num_seeds)
                alpha = np.array(alpha)
                
                num_alphas = returns.shape[0]
                for i in range(num_alphas):
                    # Get alpha value for this batch
                    curr_alpha_val = float(alpha[i].flat[0])
                    
                    # Get returns for this alpha: (num_seeds, num_episodes)
                    alpha_returns = returns[i]
                    alpha_lengths = lengths[i]
                    
                    # Calculate stats across seeds
                    # First mean over episodes for each seed
                    seed_returns = np.mean(alpha_returns, axis=-1) # (num_seeds,)
                    seed_lengths = np.mean(alpha_lengths, axis=-1)
                    
                    mean_return = float(np.mean(seed_returns))
                    std_return = float(np.std(seed_returns))
                    mean_length = float(np.mean(seed_lengths))
                    
                    wandb.log({
                        f"tuning/alpha_{curr_alpha_val}_mean": mean_return,
                        f"tuning/alpha_{curr_alpha_val}_std": std_return,
                        f"tuning/alpha_{curr_alpha_val}_length": mean_length,
                        "train/step": current_step
                    }, step=current_step)
                    
                if current_step % 20000 == 0: # Print occasionally to avoid flooding
                    print(f"[Step {current_step}] Logged tuning stats for {num_alphas} alphas")
                    
            elif returns.ndim == 2: # Maybe just seeds vmap? (num_seeds, num_episodes)
                 # Treat as single alpha if alpha is scalar or uniform
                 pass # Logic for single alpha batch
                 
            else: # Single run
                mean_return = float(np.mean(returns))
                mean_length = float(np.mean(lengths))
                wandb.log({
                    f"{prefix}/return": mean_return,
                    f"{prefix}/length": mean_length,
                    "train/step": current_step
                }, step=current_step)
                print(f"[Step {current_step}] {prefix}: return={mean_return:.2f}, length={mean_length:.2f}")

    # ... 后面的方法保持不变 ...
    def train_iteration(self, ts):
        old_global_step = ts.global_step
        placeholder_minibatch = jax.tree.map(
            lambda sdstr: jnp.empty((self.num_epochs, *sdstr.shape), sdstr.dtype),
            ts.replay_buffer.sample(self.batch_size, jax.random.PRNGKey(0)),
        )
        ts, minibatch = jax.lax.fori_loop(
            0,
            self.policy_delay,
            lambda _, ts_mb: self.train_critic(ts_mb[0]),
            (ts, placeholder_minibatch),
        )
        ts = self.train_policy(ts, minibatch, old_global_step)
        return ts

    def train_critic(self, ts):
        start_training = ts.global_step > self.fill_buffer

        # Collect transition
        uniform = jnp.logical_not(start_training)
        ts, transitions = self.collect_transitions(ts, uniform=uniform)
        ts = ts.replace(replay_buffer=ts.replay_buffer.extend(transitions))

        def update_iteration(ts, unused):
            # Sample minibatch
            rng, rng_sample = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            minibatch = ts.replay_buffer.sample(self.batch_size, rng_sample)
            if self.normalize_observations:
                minibatch = minibatch._replace(
                    obs=self.normalize_obs(ts.obs_rms_state, minibatch.obs),
                    next_obs=self.normalize_obs(ts.obs_rms_state, minibatch.next_obs),
                )
            if self.normalize_rewards:
                minibatch = minibatch._replace(
                    reward=self.normalize_rew(ts.rew_rms_state, minibatch.reward)
                )

            # Update network
            ts = self.update_critic(ts, minibatch)
            return ts, minibatch

        def do_updates(ts):
            return jax.lax.scan(update_iteration, ts, None, self.num_epochs)

        placeholder_minibatch = jax.tree.map(
            lambda sdstr: jnp.empty((self.num_epochs, *sdstr.shape), sdstr.dtype),
            ts.replay_buffer.sample(self.batch_size, jax.random.PRNGKey(0)),
        )
        ts, minibatches = jax.lax.cond(
            start_training,
            do_updates,
            lambda ts: (ts, placeholder_minibatch),
            ts,
        )
        return ts, minibatches

    def train_policy(self, ts, minibatches, old_global_step):
        def do_updates(ts):
            ts, _ = jax.lax.scan(
                lambda ts, minibatch: (self.update_actor(ts, minibatch), None),
                ts,
                minibatches,
            )
            return ts

        start_training = ts.global_step > self.fill_buffer
        ts = jax.lax.cond(start_training, do_updates, lambda ts: ts, ts)

        # Update target networks
        if self.target_update_freq == 1:
            critic_tp = self.polyak_update(ts.critic_ts.params, ts.critic_target_params)
            actor_tp = self.polyak_update(ts.actor_ts.params, ts.actor_target_params)
        else:
            update_target_params = (
                ts.global_step % self.target_update_freq
                <= old_global_step % self.target_update_freq
            )
            critic_tp = jax.tree.map(
                lambda q, qt: jax.lax.select(update_target_params, q, qt),
                self.polyak_update(ts.critic_ts.params, ts.critic_target_params),
                ts.critic_target_params,
            )
            actor_tp = jax.tree.map(
                lambda pi, pit: jax.lax.select(update_target_params, pi, pit),
                self.polyak_update(ts.actor_ts.params, ts.actor_target_params),
                ts.actor_target_params,
            )

        ts = ts.replace(critic_target_params=critic_tp, actor_target_params=actor_tp)
        return ts

    def collect_transitions(self, ts, uniform=False):
        # Use stored last_action
        action = ts.last_action

        # Step environment
        rng, rng_steps = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        rng_steps = jax.random.split(rng_steps, self.num_envs)
        next_obs, env_state, rewards, dones, _ = self.vmap_step(
            rng_steps, ts.env_state, action, self.env_params
        )

        if self.normalize_observations:
            ts = ts.replace(
                obs_rms_state=self.update_obs_rms(ts.obs_rms_state, next_obs)
            )
        if self.normalize_rewards:
            ts = ts.replace(
                rew_rms_state=self.update_rew_rms(ts.rew_rms_state, rewards, dones)
            )

        # Sample next action
        rng, rng_action = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)

        def sample_uniform(rng):
            sample_fn = self.env.action_space(self.env_params).sample
            return jax.vmap(sample_fn)(jax.random.split(rng, self.num_envs))

        def sample_policy(rng):
            if self.normalize_observations:
                curr_obs = self.normalize_obs(ts.obs_rms_state, next_obs)
            else:
                curr_obs = next_obs

            # Use flow sampling
            # (batch, 1, action_dim)
            actions = self._sample_actions(ts.actor_ts.params, curr_obs, rng, num_samples=1)
            actions = actions.squeeze(1)
            
            # Add exploration noise
            noise = self.exploration_noise * jax.random.normal(rng, actions.shape)
            action_low, action_high = self.action_space.low, self.action_space.high
            return jnp.clip(actions + noise, action_low, action_high)

        next_action = jax.lax.cond(uniform, sample_uniform, sample_policy, rng_action)

        # Return minibatch and updated train state
        minibatch = SarsaMinibatch(
            obs=ts.last_obs,
            action=action,
            reward=rewards,
            done=dones,
            next_obs=next_obs,
            next_action=next_action,
        )

        ts = ts.replace(
            env_state=env_state,
            last_obs=next_obs,
            last_action=next_action,
            global_step=ts.global_step + self.num_envs,
        )

        return ts, minibatch

    def update_critic(self, ts, minibatch):
        rng, rng_sample = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        
        def critic_loss_fn(params):
            # Sample next action from target actor
            # (batch, 1, action_dim)
            action = minibatch.next_action
            
            noise = jnp.clip(
                self.target_noise * jax.random.normal(rng_sample, action.shape), 
                -self.target_noise_clip,
                self.target_noise_clip,
            )
            action_low, action_high = self.action_space.low, self.action_space.high
            action = jnp.clip(action + noise, action_low, action_high)

            qs_target = self.vmap_critic(
                ts.critic_target_params, minibatch.next_obs, action
            )
            q_target = jnp.min(qs_target, axis=0)
            target = minibatch.reward + (1 - minibatch.done) * self.gamma * q_target
            q1, q2 = self.vmap_critic(params, minibatch.obs, minibatch.action)

            loss_q1 = optax.l2_loss(q1, target).mean()
            loss_q2 = optax.l2_loss(q2, target).mean()
            return loss_q1 + loss_q2

        grads = jax.grad(critic_loss_fn)(ts.critic_ts.params)
        ts = ts.replace(critic_ts=ts.critic_ts.apply_gradients(grads=grads))
        return ts

    def update_actor(self, ts, minibatch):
        adjusted_actions = self.auto(ts, minibatch)
        
        rng, rng_loss = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        
        def actor_loss_fn(params, rng):
            # Flow matching loss
            batch_size, action_dim = minibatch.action.shape
            
            rng, x_rng, t_rng = jax.random.split(rng, 3)
            
            x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
            x_1 = adjusted_actions # Target
            
            t = jax.random.uniform(t_rng, (batch_size, 1))
            x_t = (1 - t) * x_0 + t * x_1
            vel_target = x_1 - x_0
            
            # Predict velocity
            vel_pred = self.actor.apply(params, minibatch.obs, x_t, t)
            
            raw_loss = (vel_pred - vel_target) ** 2
            return jnp.mean(raw_loss)

        grads = jax.grad(actor_loss_fn)(ts.actor_ts.params, rng_loss)
        ts = ts.replace(actor_ts=ts.actor_ts.apply_gradients(grads=grads))
        return ts


import jax
# 在你的主训练脚本中添加这个函数来替换默认的评估
def custom_eval_callback(algo, train_state, rng):
    """JAX 友好的评估函数，解决 Tracer 错误"""
    env = algo.env
    env_params = algo.env_params
    max_episode_steps = 200  # Pendulum-v1 的步数
    num_eval_episodes = 10   # 评估 10 个 episodes
    
    # 获取 act 函数 (闭包)
    act_fn = algo.make_act(train_state)

    def single_step(carry, _):
        # carry 存储在步骤之间传递的状态
        obs, state, done, cumulative_reward, step_rng = carry
        
        # 即使已经 done，为了保持 JAX 数组形状一致，我们依然执行计算，但会通过 mask 屏蔽结果
        step_rng, action_rng, env_rng = jax.random.split(step_rng, 3)
        
        # 选择动作
        action = act_fn(obs, action_rng)
        
        # 执行环境步
        next_obs, next_state, reward, next_done, _ = env.step(
            env_rng, state, action, env_params
        )
        
        # 如果当前已经 done，则不增加奖励
        new_done = jnp.logical_or(done, next_done)
        new_reward = cumulative_reward + reward * (1.0 - done.astype(jnp.float32))
        
        return (next_obs, next_state, new_done, new_reward, step_rng), None

    def evaluate_episode(episode_rng):
        # 初始化环境
        rng_reset, rng_run = jax.random.split(episode_rng)
        obs, state = env.reset(rng_reset, env_params)
        
        # 初始化 carry: (obs, state, done, reward, rng)
        init_carry = (obs, state, jnp.array(False), jnp.array(0.0), rng_run)
        
        # 使用 jax.lax.scan 替代 Python while 循环
        final_carry, _ = jax.lax.scan(
            single_step, init_carry, None, length=max_episode_steps
        )
        
        final_reward = final_carry[3]
        return final_reward, jnp.array(max_episode_steps)

    # 并行评估多个 Episode
    rngs = jax.random.split(rng, num_eval_episodes)
    returns, lengths = jax.vmap(evaluate_episode)(rngs)
    
    return returns, lengths




# ========== 创建并训练算法 ==========

algo = DOALSARSA.create(
    env="brax/hopper",
    total_timesteps=1000000,
    eval_freq=50000,
    num_envs=1,
    learning_rate=0.00018789,
    batch_size=256,
    gamma=0.99,
    fill_buffer=1000,
    flow_steps=10,
    max_q_samples=4,
    policy_delay=3,
    alpha=0.2,
    delta=2.0,
    exploration_noise=0.1,
    target_noise=0.2,
    target_noise_clip=0.5,
)

# ========== 初始化WandB ==========
# Extract configuration from algo for logging
config_dict = {
    "env": "brax/ant",
}

# Programmatically extract all scalar fields from the algorithm
if dataclasses.is_dataclass(algo):
    for field in dataclasses.fields(algo):
        val = getattr(algo, field.name)
        
        # Handle JAX/Numpy types
        if isinstance(val, (jnp.ndarray, np.ndarray)):
            if val.ndim == 0:
                val = val.item()
            else:
                continue # Skip non-scalar arrays (like params, buffers)
        
        # Only log scalar types
        if isinstance(val, (int, float, str, bool)):
            config_dict[field.name] = val

wandb.init(
    project="doal-integrated",
    config=config_dict,
    name="doal-tune",
)

print("使用修复后的DOAL训练")
# Instead of: algo.eval_callback = custom_eval_callback
algo = algo.replace(eval_callback=custom_eval_callback)

# ========== Hyperparameter Tuning for Alpha ==========
print("\nStarting Alpha Tuning with Vmap...")
alphas = jnp.array([0.01, 0.03, 0.1, 0.3])
num_seeds_per_alpha = 4  # Keeping the same number of seeds for statistical significance

# Total runs = 4 alphas * 16 seeds = 64 runs

# Prepare seeds: (num_alphas, num_seeds)
# We use the same set of seeds for each alpha for paired comparison (reduced variance).
seeds = jax.random.split(jax.random.PRNGKey(0), num_seeds_per_alpha)

def train_single_alpha_seed(alpha, rng):
    # Create a specialized algo for this alpha
    # Note: algo is captured from outer scope (the base configuration)
    tuning_algo = algo.replace(alpha=alpha)
    return tuning_algo.train(rng=rng)

# vmap over seeds (inner loop)
# in_axes: alpha is None (shared), rng is 0 (batched)
train_seeds_fn = jax.vmap(train_single_alpha_seed, in_axes=(None, 0))

# vmap over alphas (outer loop)
# in_axes: alpha is 0 (batched), rng is None (shared/broadcasted)
# We pass `seeds` as the second argument, and want it shared across alphas, so we use None.
# This means for every alpha, we run with the SAME set of 16 seeds.
train_tuning_fn = jax.vmap(train_seeds_fn, in_axes=(0, None))

# Compile
print(f"Compiling and training for alphas={alphas} with {num_seeds_per_alpha} seeds each...")
train_tuning_fn = jax.jit(train_tuning_fn)

# Run
# Output structure: (num_alphas, num_seeds, ...)
tuning_states, tuning_evaluations = train_tuning_fn(alphas, seeds)

print("Tuning complete!")

# Analysis
if tuning_evaluations is not None and len(tuning_evaluations) >= 2:
    # tuning_evaluations[0] is returns
    # Shape: (num_alphas, num_seeds, num_eval_steps, num_eval_episodes)
    # We want to look at the final performance (last eval step, mean over episodes)
    
    all_returns = tuning_evaluations[0] # (A, S, T, E)
    
    # Take mean over evaluation episodes (E)
    mean_returns_traj = jnp.mean(all_returns, axis=-1) # (A, S, T)
    
    # Log full history to WandB
    num_alphas, num_seeds, num_steps = mean_returns_traj.shape
    print(f"\nLogging full training history ({num_steps} steps) to WandB...")
    
    # Collect all data first
    table_data = []
    
    # Define columns: step, then (mean, std) for each alpha
    columns = ["step"]
    for alpha_val in alphas:
        columns.append(f"alpha_{alpha_val:.2f}_mean")
        columns.append(f"alpha_{alpha_val:.2f}_std")
    
    for t in range(num_steps):
        current_step = t * algo.eval_freq
        row = [current_step]
        
        for i, alpha_val in enumerate(alphas):
            step_returns = mean_returns_traj[i, :, t]
            mean_ret = float(np.mean(step_returns))
            std_ret  = float(np.std(step_returns))
            
            row.append(mean_ret)
            row.append(std_ret)
            
        table_data.append(row)

    # Create table once
    tuning_table = wandb.Table(columns=columns, data=table_data)

    # Log plots using custom chart (line_series) which works with this wide format
    # Or simply log the table and let user use custom charts
    
    # Construct arguments for line_series
    # We want to plot all means
    xs = [row[0] for row in table_data] # steps
    ys_means = []
    keys_means = []
    ys_stds = []
    keys_stds = []
    
    for i, alpha_val in enumerate(alphas):
        # mean is at index 1 + 2*i
        # std is at index 1 + 2*i + 1
        mean_idx = 1 + 2*i
        std_idx = 2 + 2*i
        
        ys_means.append([row[mean_idx] for row in table_data])
        keys_means.append(f"alpha_{alpha_val:.2f}")
        
        ys_stds.append([row[std_idx] for row in table_data])
        keys_stds.append(f"alpha_{alpha_val:.2f}")

    wandb.log({
        "tuning/mean_return_plot": wandb.plot.line_series(
            xs=xs,
            ys=ys_means,
            keys=keys_means,
            title="Mean Return vs. Step",
            xname="step"
        ),
        "tuning/std_return_plot": wandb.plot.line_series(
            xs=xs,
            ys=ys_stds,
            keys=keys_stds,
            title="Return Std vs. Step",
            xname="step"
        ),
        # Also log the raw table for inspection
        "tuning/full_results_table": tuning_table
    })
    # Take the last evaluation step (final performance)
    # Check if we have steps
    if mean_returns_traj.shape[-1] > 0:
        final_returns = mean_returns_traj[:, :, -1] # (A, S)
        
        print("\nResults Summary:")
        for i, alpha in enumerate(alphas):
            alpha_returns = final_returns[i]
            mean_perf = float(jnp.mean(alpha_returns))
            std_perf = float(jnp.std(alpha_returns))
            print(f"Alpha {alpha:.2f}: Mean Return = {mean_perf:.2f} ± {std_perf:.2f}")
            
            # (Optional) Log final summary metrics if not already covered by step logging
            # wandb.log(...)

        # Find best alpha
        mean_perfs = jnp.mean(final_returns, axis=1)
        best_idx = jnp.argmax(mean_perfs)
        best_alpha = alphas[best_idx]
        print(f"\nBest Alpha: {best_alpha:.2f} with Return: {float(mean_perfs[best_idx]):.2f}")
        
        wandb.summary["best_alpha"] = float(best_alpha)
        wandb.summary["best_alpha_return"] = float(mean_perfs[best_idx])
    else:
        print("No evaluation steps recorded.")

# ========== Finish ==========
wandb.finish()
print("\nAll training completed.")

