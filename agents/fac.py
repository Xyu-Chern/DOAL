import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


class FACAgent(flax.struct.PyTreeNode):
    """Flow Actor-Critic (FAC) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def compute_behavior_density(self, observations, actions, grad_params=None):
        """Compute behavior proxy density log β̂(a|s) using instantaneous change of variables."""
        batch_size, action_dim = actions.shape
        
        # Solve backward ODE to find corresponding noise z
        z = self.solve_backward_ode(observations, actions, grad_params)
        
        # Compute base distribution probability log p₀(z)
        log_p0 = -0.5 * (jnp.sum(z**2, axis=-1) + action_dim * jnp.log(2 * jnp.pi))
        
        # Compute divergence integral ∫₀¹ ∇·v_ψ du using Euler method
        total_divergence = 0.0
        for i in range(self.config['flow_steps']):
            u = jnp.full((batch_size, 1), i / (self.config['flow_steps'] - 1))
            a_u = (1 - u) * z + u * actions  # Path point
            
            # Compute divergence using Hutchinson estimator for high-dimensional actions
            if action_dim > 8:  # Use Hutchinson for high-dim action spaces
                divergence = self.divergence_hutchinson(observations, a_u, u, grad_params)
            else:
                divergence = self.divergence_exact(observations, a_u, u, grad_params)
                
            total_divergence += divergence * (1.0 / (self.config['flow_steps'] - 1))
        
        log_density = log_p0 - total_divergence
        return log_density

    def divergence_hutchinson(self, observations, a_u, u, grad_params, num_probes=8):
        """Hutchinson estimator for divergence - O(d_A) complexity."""
        def compute_trace_estimate(probe):
            # Compute vector-Jacobian product e^T J
            def v_output(a_u_flat):
                a_u_reshaped = a_u_flat.reshape(a_u.shape)
                v = self.network.select('behavior_proxy')(observations, a_u_reshaped, u, 
                                                         params=grad_params, is_encoded=True)
                return jnp.sum(v * probe)
            
            e_jacobian = jax.grad(v_output)(a_u.reshape(-1))
            trace_estimate = jnp.sum(probe.reshape(-1) * e_jacobian)
            return trace_estimate
        
        probes = jax.random.normal(jax.random.PRNGKey(0), (num_probes, *a_u.shape))
        trace_estimates = jax.vmap(compute_trace_estimate)(probes)
        return jnp.mean(trace_estimates)

    def divergence_exact(self, observations, a_u, u, grad_params):
        """Exact divergence computation - O(d_A²) complexity."""
        def v_output(a_u_flat):
            a_u_reshaped = a_u_flat.reshape(a_u.shape)
            return self.network.select('behavior_proxy')(observations, a_u_reshaped, u, 
                                                        params=grad_params, is_encoded=True)
        
        # Compute full Jacobian matrix
        jacobian = jax.jacobian(v_output)(a_u.reshape(-1))
        
        # Reshape jacobian to proper dimensions and compute trace
        # jacobian shape: (batch_size * action_dim, batch_size * action_dim)
        # We need to reshape it to (batch_size, action_dim, batch_size, action_dim)
        # and then take trace over the action dimensions
        batch_size, action_dim = a_u.shape
        jacobian_reshaped = jacobian.reshape(batch_size, action_dim, batch_size, action_dim)
        
        # We only want the diagonal blocks (same batch element)
        # Extract diagonal blocks: (batch_size, action_dim, action_dim)
        diagonal_blocks = jacobian_reshaped[jnp.arange(batch_size), :, jnp.arange(batch_size), :]
        
        # Compute trace for each batch element: (batch_size,)
        divergence = jnp.trace(diagonal_blocks, axis1=1, axis2=2)
        
        return jnp.mean(divergence)  # Return mean over batch

    def solve_backward_ode(self, observations, actions, grad_params):
        """Solve backward ODE to find z such that φ₁(z) = a."""
        z = actions  # Initialize with final action
        # Backward Euler method
        for i in range(self.config['flow_steps'] - 1, -1, -1):
            u = jnp.full((observations.shape[0], 1), i / self.config['flow_steps'])
            if self.config['encoder'] is not None:
                encoded_obs = self.network.select('behavior_proxy_encoder')(observations)
            else:
                encoded_obs = observations
            vels = self.network.select('behavior_proxy')(encoded_obs, z, u, 
                                                        params=grad_params, is_encoded=True)
            z = z - vels / self.config['flow_steps']
        return z

    def compute_confidence_weights(self, observations, actions, grad_params):
        """Compute confidence weights w^β̂(s,a) = max(0, 1 - β̂(a|s)/ε)."""
        log_beta_hat = self.compute_behavior_density(observations, actions, grad_params)
        beta_hat_density = jnp.exp(log_beta_hat)
        

        epsilon = jnp.min(beta_hat_density)  

        weights = jnp.maximum(0.0, 1.0 - beta_hat_density / epsilon)
        
        return weights, epsilon



    def critic_loss(self, batch, grad_params, rng):
        """Compute the FAC critic loss with OOD penalization."""
        rng, actor_sample_rng = jax.random.split(rng)
        
        # ... (actor_actions 采样不变) ...
        batch_size = batch['observations'].shape[0]
        noises = jax.random.normal(actor_sample_rng, (batch_size, self.config['action_dim']))
        actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, 
                                                                params=grad_params)
        actor_actions = jnp.clip(actor_actions, -1, 1)
        
        # =================================================================
        # ✨ 停止行为代理（behavior_proxy）的梯度（逻辑保持正确）
        # =================================================================
        psi_params_key = 'modules_behavior_proxy'
        
        stopped_psi_params = jax.tree_util.tree_map(
            jax.lax.stop_gradient,
            grad_params[psi_params_key]
        )
        
        # 2. 构造用于计算密度的参数字典 (density_params)
        density_params = dict(grad_params) 
        density_params[psi_params_key] = stopped_psi_params
        
        # 3. 计算置信权重，使用停止梯度后的参数
        # policy_weights 将不依赖于 psi 的梯度
        policy_weights, epsilon = self.compute_confidence_weights(
            batch['observations'], actor_actions, density_params) 
        
        # ... (Standard TD loss 不变) ...
        rng, next_sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'], seed=next_sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)

        # ... (TD target Q calculation) ...
        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)
        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        # TD Loss on batch actions
        current_qs = self.network.select('critic')(batch['observations'], 
                                                actions=batch['actions'], 
                                                params=grad_params) 
        q = current_qs.mean(axis=0)
        td_loss = jnp.square(q - target_q).mean()
        
        # Critic penalization term (FAC innovation)
        # policy_qs 必须使用 grad_params，确保 Qϕ 接收到惩罚项的梯度
        policy_qs = self.network.select('critic')(batch['observations'], 
                                                actions=actor_actions, 
                                                params=grad_params)
        policy_q = policy_qs.mean(axis=0)
        
        # penalty_loss: w(s,a) * Q(s,a)
        penalty_loss = self.config['alpha_critic'] * jnp.mean(policy_weights * policy_q)
        
        total_critic_loss = td_loss + penalty_loss

        # ... [Return] ...
        return total_critic_loss, {
            'critic_loss': total_critic_loss,
            'td_loss': td_loss,
            'penalty_loss': penalty_loss,
            'q_mean': q.mean(),
            'policy_weights_mean': policy_weights.mean(),
            'epsilon': epsilon,
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FAC actor loss with Q-value normalization."""
        batch_size, action_dim = batch['actions'].shape
        rng, bc_rng, distill_rng = jax.random.split(rng, 3)

        # 1. 行为克隆流损失 (Flow Matching Loss)
        x_0 = jax.random.normal(bc_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(bc_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('behavior_proxy')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # 2. Q-Maximizing Term (核心缺失部分)
        rng, actor_sample_rng = jax.random.split(rng)
        # 使用 actor_onestep_flow 采样动作 a ~ πθ(·|s)
        actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], 
                                                                jax.random.normal(actor_sample_rng, (batch_size, action_dim)),
                                                                params=grad_params)
        actor_actions = jnp.clip(actor_actions, -1, 1)

        # 计算 Q 值（使用 twin critic 的 mean/min，这里使用 mean for simplicity）
        policy_qs = self.network.select('critic')(batch['observations'], 
                                                actions=actor_actions, 
                                                params=grad_params)
        
        if self.config['q_agg'] == 'min':
            policy_q = policy_qs.min(axis=0) # 使用 Q-minimization
        else:
            policy_q = policy_qs.mean(axis=0) # 使用 Q-mean
            
        # Q-Value 归一化 (论文 G.3.1 描述)
        if self.config['normalize_q_loss']:
            # |Qϕ| = 1/M * Σ |Qϕ(s, a)|
            q_norm = jnp.mean(jnp.abs(policy_q)) 
            normalized_policy_q = policy_q / (q_norm + 1e-6)
        else:
            normalized_policy_q = policy_q

        # Actor Loss: -E[Normalized Q] + lambda * E[BC Flow Loss]
        q_maximizing_loss = -jnp.mean(normalized_policy_q)
        
        # 结合两个损失项
        total_actor_loss = q_maximizing_loss * self.config['beta_actor'] + \
                        bc_flow_loss * self.config['alpha_actor']
        
        return total_actor_loss, {
            'actor_loss': total_actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'q_maximizing_loss': q_maximizing_loss,
            'q_mean_for_actor': policy_q.mean(),
        }


    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
        grad_params=None
    ):
        """Sample actions from the one-step policy."""
        action_seed, noise_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        if self.config["sample_with_flow"]:
            return self.compute_flow_actions(observations, noises, grad_params)
        actions = self.network.select('actor_onestep_flow')(observations, noises, params=grad_params)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
        grad_params=None
    ):
        """Compute actions from the behavior proxy flow model using Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('behavior_proxy_encoder')(observations)
        actions = noises
        # Euler method
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('behavior_proxy')(observations, actions, t, 
                                                        params=grad_params, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new FAC agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Define encoders
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['behavior_proxy'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # Define networks
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        behavior_proxy_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('behavior_proxy'),
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            behavior_proxy=(behavior_proxy_def, (ex_observations, ex_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions)),
        )
        if encoders.get('behavior_proxy') is not None:
            network_info['behavior_proxy_encoder'] = (encoders.get('behavior_proxy'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fac',  # Agent name
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=64,
            sample_with_flow=False,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            q_agg='mean',  # 'min' for D4RL, 'mean' for OGBench non-Antmaze
            alpha_actor=1.0,  # Distillation loss coefficient
            beta_actor=1.0,   # Q maximization coefficient
            alpha_critic=1.0, # Critic penalization coefficient
            flow_steps=10,
            normalize_q_loss=True,  # Q-value normalization for actor
            encoder=ml_collections.config_dict.placeholder(str),
            # FAC-specific parameters
            use_hutchinson_estimator=True,  # Use Hutchinson for high-dim actions
            threshold_method='batch_adaptive',  # 'batch_adaptive' or 'dataset_constant'
        )
    )
    return config