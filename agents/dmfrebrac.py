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

from agents.rebrac import ReBRACAgent
from agents.dmfql import DMFQLAgent


class DMFReBRACAgent(ReBRACAgent,DMFQLAgent):
    """Revisited behavior-regularized actor-critic (ReBRAC) agent.

    ReBRAC is a variant of TD3+BC with layer normalization and separate actor and critic penalization.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()


    def dmf_actor_loss(self, batch, grad_params, rng=None,aux={}):
        """Compute the behavioral flow-matching actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        
        alpha = self.config["alpha"] 
        adjusted_actions , adjustment, hd, g, q = self.get_guided_action(x_1, x_1, batch['observations'],alpha,delta=self.config["delta"],params=self.network.params)
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * adjusted_actions
        vel = adjusted_actions - x_0

        pred = self.network.select('actor_flow')(batch['observations'], x_t, t, params=grad_params)

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

    def critic_loss(self, batch, grad_params, rng, mode="offline"):
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
        if mode=="offline":
            next_q = next_q - self.config['alpha_critic'] * mse
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
    def sample_actions_simple(
        self,
        observations,
        seed=None,
    ):
        orig_observations = observations
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_flow_encoder')(observations)
        action_seed, noise_seed = jax.random.split(seed)
        num_samples = self.config["target_num_samples"]
        # Sample `num_samples` noises and propagate them through the flow.
        n_observations = jnp.repeat(jnp.expand_dims(observations, 0), num_samples, axis=0)
        n_orig_observations = jnp.repeat(jnp.expand_dims(orig_observations, 0), num_samples, axis=0)
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
        
    @partial(jax.jit, static_argnames=('full_update', "mode"))
    def total_loss(self, batch, grad_params, full_update=True, rng=None, mode="offline"):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_flow_rng,actor_rng, critic_rng = jax.random.split(rng, 4)

        critic_loss, critic_info,aux = self.critic_loss(batch, grad_params, critic_rng, mode)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        dmf_actor_loss, actor_info = self.dmf_actor_loss(batch, grad_params, actor_flow_rng,aux)
        for k, v in actor_info.items():
            info[f'dmf_actor/{k}'] = v

        loss = critic_loss + dmf_actor_loss
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
     #        optax.clip_by_global_norm(max_norm=config["gn"]),
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
            agent_name='dmfrebrac',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            distill_from_target=False,  # BC coefficient (need to be tuned for each environment).
            target_num_samples = 4,
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            tanh_squash=True,  # Whether to squash actions with tanh.
            num_samples = 4,
            flow_steps=10,  # Number of flow steps.
            flow_only=True,
            gn=100.0,
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
        )
    )
    return config