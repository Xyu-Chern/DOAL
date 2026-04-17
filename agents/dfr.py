
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


class DFRAgent(flax.struct.PyTreeNode):

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    
    @partial(jax.jit, static_argnames=("mode",))
    def critic_loss(self, batch, grad_params, rng, mode="offline"):
        rng, sample_rng = jax.random.split(rng)

        if self.config["flow_only"] :
            next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
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

        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        elif self.config['q_agg'] == 'max':
            next_q = next_qs.max(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        mse = jnp.square(next_actions - batch['next_actions']).sum(axis=-1)

        if mode == "offline":
            q_std = jnp.std(next_qs, axis=0) 
            mse_scale = jax.lax.stop_gradient(mse.mean() + 1e-6)
            # next_q = next_q - q_std / mse_scale * mse
            next_q = next_q - 0.01 * mse
        else:
            next_q = next_q 

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

    @jax.jit
    def get_guided_action(self, q_action, observation, delta, params):

        def bc_loss_wrt_q_action(q_action):
            qs = self.network.select('critic')(observation, q_action, params=params)
            q = jnp.mean(qs, axis=0)
            return  jnp.sum(q) 
    
        v_grad_q = jax.value_and_grad(bc_loss_wrt_q_action) 
        q_sum, g = v_grad_q(q_action) 

        norm = jnp.linalg.norm(g, axis=-1, keepdims=True) + 1e-6  # batch * 1
        norm_mean = jnp.mean(norm)
        norm_std = jnp.std(norm)

        if self.config["use_batch_nrom"]:
            # dx = delta / (norm_mean + norm_std) * g

            max_norm = 5.0
            scale = jnp.minimum( max_norm / norm, 1.0)
            clipped_g = g * scale
            dx =   (delta / norm_mean ) * clipped_g
        else:
            dx = delta * g

        target_action = jnp.clip(q_action + dx, -1.0, 1.0)  
        target_action = jax.lax.stop_gradient(target_action)
        dx = jax.lax.stop_gradient(dx)
        q_sum =  jax.lax.stop_gradient(q_sum)

        return target_action, dx, ( norm_mean / delta ) * jnp.ones(q_action.shape[0], dtype=q_action.dtype), g, q_sum
    
    @jax.jit
    def actor_loss(self, batch, grad_params, rng=None,aux={}):
        """Compute the behavioral flow-matching actor loss."""   
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        
        adjusted_actions , adjustment, hd, g, q = self.get_guided_action(x_1, batch['observations'], delta=self.config["delta"],params=self.network.params)

        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * adjusted_actions
        vel = adjusted_actions - x_0

        pred = self.network.select('actor_flow')(batch['observations'], x_t, t, params=grad_params)

        raw_actor_loss = (pred - vel) ** 2
        actor_loss =  jnp.mean(raw_actor_loss) *30

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

    @jax.jit
    def actor_loss(self, batch, grad_params, rng=None, aux={}):
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions'] 

        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0 
        v1 = self.network.select('actor_flow')(batch['observations'], x_t, t, params=grad_params)

        flow_loss = jnp.mean((v1 - vel) ** 2)

        target_action = jax.lax.stop_gradient(x_t + (1 - t) * v1) 
        target_action = jnp.clip(target_action, -1.0, 1.0)

        qs = self.network.select('critic')(batch['observations'], actions=target_action)
        q = jnp.min(qs, axis=0)

        if self.config['normalize_q_loss']:
            q_loss = -(aux["lam"] * q).mean()
        else:
            q_loss = - q.mean()

        actor_loss = flow_loss + q_loss

        return actor_loss, {
            "actor_loss":jnp.mean(actor_loss),
        }

      
    @partial(jax.jit, static_argnames=('full_update', "mode"))
    def total_loss(self, batch, grad_params, full_update=True, rng=None, mode="offline"):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_flow_rng,actor_rng, critic_rng = jax.random.split(rng, 4)

        critic_loss, critic_info,aux = self.critic_loss(batch, grad_params, critic_rng, mode)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_flow_rng,aux)
        for k, v in actor_info.items():
            info[f'dmf_actor/{k}'] = v

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

    @partial(jax.jit, static_argnames=('full_update', 'mode',))
    def update(self, batch, full_update=True, mode="offline"):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, full_update, rng=rng, mode=mode)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        if full_update:
            # Update the target networks only when `full_update` is True.
            self.target_update(new_network, 'critic')
            self.target_update(new_network, 'actor')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        orig_observations = observations
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_flow_encoder')(observations)
        action_seed, noise_seed = jax.random.split(seed)

        num_candidates = self.config['num_candidates']

        n_observations = jnp.repeat(jnp.expand_dims(observations, 0), num_candidates, axis=0)
        n_orig_observations = jnp.repeat(jnp.expand_dims(orig_observations, 0), num_candidates, axis=0)
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

        q = self.network.select('critic')(n_orig_observations, actions=actions).min(axis=0)
        if len(actions.shape) == 3:
            b = orig_observations.shape[0]
            actions = actions[jnp.argmax(q, axis=0), jnp.arange(b)]
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
            encoders['actor'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            tanh_squash=config['tanh_squash'],
            state_dependent_std=False,
            const_std=True,
            final_fc_init_scale=config['actor_fc_scale'],
            encoder=encoders.get('actor'),
        )

        actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_flow'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations,)),
            target_actor=(copy.deepcopy(actor_def), (ex_observations,)),
            actor_flow=(actor_flow_def, (ex_observations, ex_actions, ex_times)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.chain(
            optax.adam(learning_rate=config['lr'])
        )
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_actor'] = params['modules_actor']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='dfr',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            normalize_q_loss= False,  # Whether to normalize the Q loss.
            num_candidates = 4,
            delta= 0.1,  
            q_agg="mean",
            use_batch_nrom = True,
            max_norm = 5.0,
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            tanh_squash = True,  # Whether to squash actions with tanh.
            flow_steps = 10,  # Number of flow steps.
            flow_only=True,
            actor_fc_scale = 0.01,  # Final layer initialization scale for actor.
            actor_freq=2,  # Actor update frequency.
            actor_noise = 0.5,  # Actor noise scale.
            actor_noise_clip = 0.001,  # Actor noise clipping threshold.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config