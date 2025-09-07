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

from agents.iql import IQLAgent
class IFQLAgent(IQLAgent):
    """Implicit flow Q-learning (IFQL) agent.

    IFQL is the flow variant of implicit diffusion Q-learning (IDQL).
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()


    def actor_loss(self, batch, grad_params, rng=None,aux={}):
        """Compute the behavioral flow-matching actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_flow')(batch['observations'], x_t, t, params=grad_params)
        actor_loss = jnp.mean((pred - vel) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
        }



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
        actions = jax.random.normal(
            action_seed,
            (
                *observations.shape[:-1],
                self.config['num_samples'],
                self.config['action_dim'],
            ),
        )
        n_observations = jnp.repeat(jnp.expand_dims(observations, 0), self.config['num_samples'], axis=0)
        n_orig_observations = jnp.repeat(jnp.expand_dims(orig_observations, 0), self.config['num_samples'], axis=0)
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], self.config['num_samples'], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_flow')(n_observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)

        # Pick the action with the highest Q-value.
        q = self.network.select('critic')(n_orig_observations, actions=actions).min(axis=0)
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

        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = encoder_module()
            encoders['critic'] = encoder_module()
            encoders['actor_flow'] = encoder_module()

        # Define networks.
        value_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('value'),
        )
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_flow'),
        )

        network_info = dict(
            value=(value_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_flow=(actor_flow_def, (ex_observations, ex_actions, ex_times)),
        )
        if encoders.get('actor_flow') is not None:
            # Add actor_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_flow_encoder'] = (encoders.get('actor_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.chain(
            optax.clip_by_global_norm(max_norm=config["gn"]),
            optax.adam(learning_rate=config['lr'])
        )
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']

        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='ifql',  # Agent name.
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            alpha_actor=10,
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            gn=100.0,
            num_samples=32,  # Number of action samples for rejection sampling.
            flow_steps=10,  # Number of flow steps.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
