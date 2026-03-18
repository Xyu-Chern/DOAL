import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value, ActorVectorField
from utils.dit_jax import MFDiT, MFDiT_SIM
from agents.rebrac import ReBRACAgent
from agents.dmfql import DMFQLAgent


class DMFReBRAC_jitAgent(ReBRACAgent,DMFQLAgent):
    """Revisited behavior-regularized actor-critic (ReBRAC) agent.

    ReBRAC is a variant of TD3+BC with layer normalization and separate actor and critic penalization.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    current_alpha: float = 1.0  # Current alpha state
    loss_history: jnp.ndarray = None  # Initialize as fixed shape in __post_init__
    valid_count: int = 0  # Track valid history count

    def __post_init__(self):
        if self.loss_history is None:
            # Initialize as fixed shape zero array
            history_window_size = self.config.get('loss_history_window_size', 5)
            object.__setattr__(self, 'loss_history', jnp.zeros(history_window_size))


    def dmf_actor_loss(self, batch, grad_params, rng=None,aux={}):
        """Compute the behavioral flow-matching actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        
        alpha = self.config["alpha"] 
        adjusted_actions , adjustment,hd,g, q = self.get_guided_action(  x_1, x_1,batch['observations'],alpha,delta=self.config["delta"],params=self.network.params)
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * adjusted_actions
        vel = adjusted_actions - x_0

        pred = self.network.select('actor_flow')(batch['observations'], x_t, t, params=grad_params)

        # g = gn(batch['observations'], z, t, params=grad_params)
        # print(z.shape, batch['observations'].shape, t.shape, g.shape)  (256, 5) (256, 28) (256, 1) (256, 5)
        # assert False

        raw_actor_loss = (pred - vel) ** 2
        actor_loss =  jnp.mean(raw_actor_loss) * self.config['alpha_actor']

        return actor_loss, {
            "raw_actor_loss":jnp.mean(raw_actor_loss),
            'adj_norm': jnp.mean(jnp.linalg.vector_norm(adjustment,axis=-1)),
            'adj': jnp.mean(jnp.abs(adjustment)),
            "q":jnp.mean(q),
            "hd": jnp.mean(hd),
            "hd_abs": jnp.mean(jnp.abs(hd)),
            "hd_std": jnp.std(hd),
            "hd_max": jnp.max(hd),
            "hd_min": jnp.min(hd),
            "g": jnp.mean(g),
            "g_std": jnp.std(g),
            "g_abs": jnp.mean(jnp.abs(g)),
            "g_max": jnp.max(g),
            "g_min": jnp.min(g),
        }

    def critic_loss(self, batch, grad_params, rng):
        """Compute the ReBRAC critic loss."""
        rng, sample_rng = jax.random.split(rng)

        if self.config["flow_only"] :
            next_actions = self.sample_actions_simple(batch['next_observations'], seed=sample_rng)
            next_actions = jnp.clip(next_actions, -1, 1)
        else:
            next_dist = self.network.select('target_actor')(batch['next_observations'])
            next_actions = next_dist.mode()
            noise = jnp.clip(
                (jax.random.normal(sample_rng, next_actions.shape) * self.config['actor_noise']),
                -self.config['actor_noise_clip'],
                self.config['actor_noise_clip'],
            )
            next_actions = jnp.clip(next_actions + noise, -1, 1)

        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        next_q = next_qs.mean(axis=0)

        mse = jnp.square(next_actions - batch['next_actions']).sum(axis=-1)
        next_q = next_q - self.config['alpha_critic'] * mse

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        lam = 1 / jax.lax.stop_gradient(jnp.abs(q).mean())
        critic_loss = jnp.square(q - target_q).mean() 
 
        aux = {"lam":lam}
        if self.config['normalize_q_loss']:
            critic_loss = aux["lam"] * critic_loss
        
        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }, aux

    # @jax.jit
    # def sample_actions_simple(
    #     self,
    #     observations,
    #     seed=None,
    # ):
    #     orig_observations = observations

    #     print("encoder : ", self.config['encoder'])

    #     if self.config['encoder'] is not None:
    #         observations = self.network.select('actor_flow_encoder')(observations)
    #     action_seed, noise_seed = jax.random.split(seed)
    #     num_samples = self.config["target_num_samples"]
    #     # Sample `num_samples` noises and propagate them through the flow.
    #     n_observations = jnp.repeat(jnp.expand_dims(observations, 0), num_samples, axis=0)
    #     n_orig_observations = jnp.repeat(jnp.expand_dims(orig_observations, 0), num_samples, axis=0)
    #     actions = jax.random.normal(
    #         action_seed,
    #         (
    #             *n_observations.shape[:-1],
    #             self.config['action_dim'],
    #         ),
    #     )
    #     for i in range(self.config['flow_steps']):
    #         t = jnp.full((*n_observations.shape[:-1], 1), i / self.config['flow_steps'])
    #         print(n_observations.shape, actions.shape, t.shape)
    #         vels = self.network.select('actor_flow')(n_observations, actions, t, is_encoded=True)
    #         assert False
    #         actions = actions + vels / self.config['flow_steps']
    #     actions = jnp.clip(actions, -1, 1)

    #     # Pick the action with the highest Q-value.
    #     q = self.network.select('critic')(n_orig_observations, actions=actions).min(axis=0)
    #     if len(actions.shape) == 3:
    #         b = orig_observations.shape[0]
    #         actions = actions[jnp.argmax(q,axis=0),jnp.arange(b)]
    #     else:
    #         actions = actions[jnp.argmax(q)]
    #     return actions
    
    @jax.jit
    def sample_actions_simple(
        self,
        observations,
        seed=None,
    ):
        orig_observations = observations

        if self.config['encoder'] is not None:
            observations = self.network.select('actor_flow_encoder')(observations)
        
        action_seed, noise_seed = jax.random.split(seed)
        
        # 移除 target_num_samples 相关代码，直接使用原始形状
        # 不再重复扩展维度
        
        # 初始动作：从正态分布采样，形状直接匹配 observations 的 batch 维度
        actions = jax.random.normal(
            action_seed,
            (
                *observations.shape[:-1],  # 保持原始 batch 维度
                self.config['action_dim'],
            ),
        )
        
        # Flow matching 迭代
        for i in range(self.config['flow_steps']):
            # t 的形状匹配 actions 的 batch 维度
            t = jnp.full((*actions.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        
        actions = jnp.clip(actions, -1, 1)

        # 移除 Q-value 筛选部分，直接返回动作
        # q = self.network.select('critic')(n_orig_observations, actions=actions).min(axis=0)
        # if len(actions.shape) == 3:
        #     b = orig_observations.shape[0]
        #     actions = actions[jnp.argmax(q,axis=0),jnp.arange(b)]
        # else:
        #     actions = actions[jnp.argmax(q)]
        
        return actions

        
    @partial(jax.jit, static_argnames=('full_update',))
    def total_loss(self, batch, grad_params, full_update=True, rng=None, current_step=0):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_flow_rng,actor_rng, critic_rng = jax.random.split(rng, 4)

        critic_loss, critic_info,aux = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        dmf_actor_loss, actor_info = self.dmf_actor_loss(batch, grad_params, actor_flow_rng,aux)
        for k, v in actor_info.items():
            info[f'dmf_actor/{k}'] = v


        # Get alpha scheduling configuration
        use_dynamic_alpha = self.config.get('use_dynamic_alpha', False)
        alpha_update_interval = self.config.get('alpha_update_interval', 200)
        
        # Use JAX operations instead of Python boolean operators
        step_mod_interval = current_step % alpha_update_interval
        should_update_alpha = jnp.logical_and(
            step_mod_interval == 0,
            current_step > 0
        )
        
        # Calculate 20% of total training steps as warmup threshold for recording only without adjustment
        max_training_steps = self.config.get('pretrain_plus_offline_steps', 1000000)
        warmup_steps = int(max_training_steps * 0.1)  # First 20% steps for recording only without adjustment
        
        # Check if we are in the first 20% of training steps
        is_in_warmup = current_step < warmup_steps
        
        # Use current alpha value during warmup or when no update is needed
        should_calculate_dynamic_alpha = jnp.logical_and(
            should_update_alpha,
            jnp.logical_not(is_in_warmup)
        )
        
        # Calculate alpha weight based on schedule type - only compute what's needed
        if use_dynamic_alpha:
            # Only calculate dynamic alpha when needed (on update steps and not in warmup)
            alpha_weight = jnp.where(
                should_calculate_dynamic_alpha,
                self._calculate_dynamic_alpha(info['dmf_actor/raw_actor_loss'], current_step, info),
                self.current_alpha  # Use cached current_alpha when not updating or in warmup
            )
        else:
            # Only calculate cosine alpha for non-dynamic scheduling
            alpha_weight = self._calculate_cosine_alpha(current_step)

        loss = critic_loss + alpha_weight * dmf_actor_loss
        return loss, info

    def _calculate_cosine_alpha(self, current_step):
        """Calculate alpha weight using cosine decay schedule"""
        max_training_steps = self.config.get('pretrain_plus_offline_steps', 1000000)
        initial_alpha = self.config.get('dynamic_alpha', 1.0)
        min_alpha = initial_alpha * 0.1  # Dynamic calculation: alpha * 0.1
        
        # Cosine decay schedule: starts at initial_alpha, decays to min_alpha
        progress = jnp.minimum(current_step / max_training_steps, 1.0)
        alpha_weight = min_alpha + (initial_alpha - min_alpha) * 0.5 * (1 + jnp.cos(jnp.pi * progress))
        
        return alpha_weight
    
    def _calculate_dynamic_alpha(self, mean_flow_loss, current_step, info):
        """Calculate alpha weight based on mean_flow_loss dynamics with sliding window average"""
        # Get configuration parameters with dynamic calculation
        initial_alpha = self.config.get('dynamic_alpha', 1.0)
        min_alpha = initial_alpha * 0.1  # Dynamic calculation: alpha * 0.1
        max_alpha = initial_alpha * 10.0  # Dynamic calculation: alpha * 10
        
        # Dynamic adjustment parameters
        loss_multiplier_threshold = self.config.get('loss_multiplier_threshold', 5)
        alpha_increase_factor = self.config.get('alpha_increase_factor', 1.2)
        alpha_decrease_factor = self.config.get('alpha_decrease_factor', 0.8)
        history_window_size = self.config.get('loss_history_window_size', 20)
        
        base_alpha = self.current_alpha
        
        # Use valid_count to check if enough history data is available
        has_enough_history = self.valid_count >= history_window_size
        
        # Calculate average of valid history data using mask instead of dynamic slicing
        effective_count = jnp.minimum(self.valid_count, history_window_size)
        
        # Create a mask to sum only valid historical data
        mask = jnp.arange(history_window_size) < effective_count
        masked_history = self.loss_history * mask
        
        # Calculate historical average loss using mask
        historical_avg_loss = jnp.where(
            effective_count > 0,
            jnp.sum(masked_history) / effective_count,
            0.0
        )
        
        # Conditionally calculate alpha adjustment
        should_increase = jnp.logical_and(
            effective_count >= 2,  # Need at least 2 data points for comparison
            mean_flow_loss > (historical_avg_loss * loss_multiplier_threshold)
        )
        should_decrease = jnp.logical_and(
            effective_count >= 2,
            mean_flow_loss < (historical_avg_loss / loss_multiplier_threshold)
        )
        
        alpha_weight = jnp.where(
            should_increase,
            jnp.minimum(base_alpha * alpha_increase_factor, max_alpha),
            base_alpha
        )
        alpha_weight = jnp.where(
            should_decrease,
            jnp.maximum(alpha_weight * alpha_decrease_factor, min_alpha),
            alpha_weight
        )
    
        info['alpha/historical_avg_loss'] = historical_avg_loss
        info['alpha/valid_count'] = self.valid_count
        info['alpha/current_vs_historical_ratio'] = jnp.where(
            effective_count > 0,
            mean_flow_loss / jnp.maximum(historical_avg_loss, 1e-8),
            1.0
        )
        # Add dynamic bounds info for debugging
        info['alpha/min_alpha_bound'] = min_alpha
        info['alpha/max_alpha_bound'] = max_alpha
        
        return alpha_weight


    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @partial(jax.jit, static_argnames=('full_update',))
    def update(self, batch, full_update=True, current_step=0):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, full_update, rng=rng, current_step = current_step )

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        if full_update:
            # Update the target networks only when `full_update` is True.
            self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        orig_observations = observations
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_flow_encoder')(observations)
        action_seed, noise_seed = jax.random.split(seed)

        # Sample `num_samples` noises and propagate them through the flow.
        n_observations = jnp.repeat(jnp.expand_dims(observations, 0), self.config['num_samples'], axis=0)
        n_orig_observations = jnp.repeat(jnp.expand_dims(orig_observations, 0), self.config['num_samples'], axis=0)
        actions = jax.random.normal(
            action_seed,
            (
                *n_observations.shape[:-1],
                self.config['action_dim'],
            ),
        )
        for i in range(self.config['flow_steps']):
            t = jnp.full((*n_observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_flow')(n_observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)

        # Pick the action with the highest Q-value.
        q = self.network.select('critic')(n_orig_observations, actions=actions).min(axis=0)
        if len(actions.shape) == 3:
            b = orig_observations.shape[0]
            actions = actions[jnp.argmax(q,axis=0),jnp.arange(b)]
        else:
            actions = actions[jnp.argmax(q)]
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]

        config['action_dim'] = action_dim
        ex_times = ex_actions[..., :1]
        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_flow'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )

        actor_flow_def = MFDiT_SIM(
            # input_dim=action_dim, 
            hidden_dim=config['actor_hidden_dims'],
            depth=config['actor_depth'],
            num_heads=config['actor_num_heads'],
            output_dim=action_dim,  
            encoder=encoders.get('actor_flow'),
            tanh_squash = config['tanh_squash'],
            use_output_layernorm = config["use_output_layernorm"],
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_flow=(actor_flow_def, (ex_observations, ex_actions, ex_times)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.chain(
     #        optax.clip_by_global_norm(max_norm=config["gn"]),
            optax.adam(learning_rate=config['lr'])
        )
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='dmfrebrac_jit',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims= 256,  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            distill_from_target=False,  # BC coefficient (need to be tuned for each environment).
            target_num_samples = 1,
            actor_depth = 3, 
            actor_num_heads = 2,# Transformer num heads  
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            tanh_squash=False,  # Whether to squash actions with tanh.
            num_samples = 4,
            flow_steps= 10,  # Number of flow steps.
            flow_only=True,
            gn=100.0,
            use_output_layernorm=False,
            actor_fc_scale=0.01,  # Final layer initialization scale for actor.
            alpha=0.0,  # Actor BC coefficient.
            delta=2.0,  # Actor BC coefficient.
            solver="auto",  # Actor BC coefficient.
            alpha_actor=0.0,  # Actor BC coefficient.
            alpha_critic=0.0,  # Critic BC coefficient.
            clip=True,
            actor_freq=2,  # Actor update frequency.
            actor_noise=0.2,  # Actor noise scale.
            actor_noise_clip=0.5,  # Actor noise clipping threshold.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            
            dynamic_alpha=10.0,  # BC coefficient (need to be tuned for each environment). # hyper_paper 2: most important
            use_dynamic_alpha = True, # Boolean flag: True for dynamic, False for cosine
            alpha_update_interval=2000,  # Frequency of dynamic alpha adjustment
            loss_multiplier_threshold=5,  # High mean_flow_loss threshold for increasing alpha
            alpha_increase_factor=1.2,  # Factor to increase alpha when loss is high
            alpha_decrease_factor=0.8,  # Factor to decrease alpha when loss is low
            loss_history_window_size=20, # set the window size.
        )
    )
    return config