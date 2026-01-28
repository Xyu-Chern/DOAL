import chex
import jax
import numpy as np
import optax
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from jax import numpy as jnp
from jax.experimental import host_callback as hcb
import wandb  # 添加WandB导入

from typing import Sequence, Callable


from rejax.algos.algorithm import Algorithm, register_init
from rejax.algos.mixins import (
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    ReplayBufferMixin,
    TargetNetworkMixin,
)
from rejax.buffers import Minibatch


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
    

from rejax.networks import  QNetwork


class DOAL(
    ReplayBufferMixin,
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

        if not self.skip_initial_evaluation:
            initial_evaluation = self.eval_callback(self, ts, ts.rng)
            
            # ====== 添加：记录初始评估到WandB ======
            if initial_evaluation is not None and len(initial_evaluation) >= 2:
                def log_initial(returns, lengths):
                    if returns.size > 0:
                        wandb.log({
                            "eval/return": float(np.mean(returns)),
                            "eval/length": float(np.mean(lengths)),
                            "train/step": 0
                        }, step=0)
                        print(f"[Step 0] Initial eval: return={np.mean(returns):.2f}")
                
                returns, lengths = initial_evaluation[0], initial_evaluation[1]
                hcb.call(log_initial, (returns, lengths))
            # ====== 结束添加 ======

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
            
            # ====== 添加：记录评估结果到WandB ======
            if eval_result is not None and len(eval_result) >= 2:
                def log_to_wandb(args):
                    step, returns, lengths = args
                    if returns.size > 0:
                        wandb.log({
                            "eval/return": float(np.mean(returns)),
                            "eval/length": float(np.mean(lengths)),
                            "train/step": int(step)
                        }, step=int(step))
                        print(f"[Step {int(step)}] Eval: return={np.mean(returns):.2f}")
                
                returns, lengths = eval_result[0], eval_result[1]
                hcb.call(log_to_wandb, (ts.global_step, returns, lengths))
            # ====== 结束添加 ======
            
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
        
        # ====== 添加：记录最终结果到WandB ======
        if evaluation is not None and len(evaluation) >= 2:
            def log_final(returns, lengths):
                if returns.size > 0:
                    final_return = float(np.mean(returns))
                    wandb.log({
                        "final/return": final_return,
                        "final/length": float(np.mean(lengths))
                    })
                    wandb.summary["final_return"] = final_return
                    print(f"Final evaluation: return={final_return:.2f}")
            
            # 取最后一次评估的结果
            if evaluation[0].shape[0] > 0:
                final_returns = evaluation[0][-1] if len(evaluation[0].shape) > 1 else evaluation[0]
                final_lengths = evaluation[1][-1] if len(evaluation[1].shape) > 1 else evaluation[1]
                hcb.call(log_final, (final_returns, final_lengths))
        # ====== 结束添加 ======

        return ts, evaluation

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
        # Sample actions
        rng, rng_action = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)

        def sample_uniform(rng):
            sample_fn = self.env.action_space(self.env_params).sample
            return jax.vmap(sample_fn)(jax.random.split(rng, self.num_envs))

        def sample_policy(rng):
            if self.normalize_observations:
                last_obs = self.normalize_obs(ts.obs_rms_state, ts.last_obs)
            else:
                last_obs = ts.last_obs

            # Use flow sampling
            # (batch, 1, action_dim)
            actions = self._sample_actions(ts.actor_ts.params, last_obs, rng, num_samples=1)
            actions = actions.squeeze(1)
            
            # Should we add exploration noise? 
            # Flow policies are stochastic, but adding noise might still be useful or expected.
            # DOAL defaults: exploration_noise=0.3
            noise = self.exploration_noise * jax.random.normal(rng, actions.shape)
            action_low, action_high = self.action_space.low, self.action_space.high
            return jnp.clip(actions + noise, action_low, action_high)

        actions = jax.lax.cond(uniform, sample_uniform, sample_policy, rng_action)

        # Step environment
        rng, rng_steps = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        rng_steps = jax.random.split(rng_steps, self.num_envs)
        next_obs, env_state, rewards, dones, _ = self.vmap_step(
            rng_steps, ts.env_state, actions, self.env_params
        )

        if self.normalize_observations:
            ts = ts.replace(
                obs_rms_state=self.update_obs_rms(ts.obs_rms_state, next_obs)
            )
        if self.normalize_rewards:
            ts = ts.replace(
                rew_rms_state=self.update_rew_rms(ts.rew_rms_state, rewards, dones)
            )

        # Return minibatch and updated train state
        minibatch = Minibatch(
            obs=ts.last_obs,
            action=actions,
            reward=rewards,
            next_obs=next_obs,
            done=dones,
        )
        ts = ts.replace(
            last_obs=next_obs,
            env_state=env_state,
            global_step=ts.global_step + self.num_envs,
        )
        return ts, minibatch

    def update_critic(self, ts, minibatch):
        rng, rng_sample = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        
        def critic_loss_fn(params):
            # Sample next action from target actor
            # (batch, 1, action_dim)
            next_actions = self._sample_actions(ts.actor_target_params, minibatch.next_obs, rng_sample, num_samples=1)
            action = next_actions.squeeze(1)
            
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