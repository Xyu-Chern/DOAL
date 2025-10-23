import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field,DOALAgent
from utils.networks import Actor, Value
from agents.rebrac import ReBRACAgent


class DReBRACAgent(ReBRACAgent,DOALAgent):
    """Revisited behavior-regularized actor-critic (ReBRAC) agent.

    ReBRAC is a variant of TD3+BC with layer normalization and separate actor and critic penalization.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()


    def actor_loss(self, batch, grad_params, rng,aux={}):
        """Compute the ReBRAC actor loss."""
        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        actions = dist.mode()


        adjusted_actions , adjustment,hd,g, q = self.get_guided_action(  batch['actions'], batch['actions'],batch['observations'],alpha=self.config["alpha"],delta=self.config["delta"],params=self.network.params)
        # BC loss.
        mse = jnp.square(actions - adjusted_actions).sum(axis=-1)

        bc_loss = ( mse).mean()

        total_loss = self.config['alpha_actor'] * bc_loss

        if self.config['tanh_squash']:
            action_std = dist._distribution.stddev()
        else:
            action_std = dist.stddev().mean()

        actor_info = {
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'q': jnp.mean(q),
        'adj_norm': jnp.mean(jnp.linalg.vector_norm(adjustment,axis=-1)),
        'adj_std': jnp.std(jnp.linalg.vector_norm(adjustment,axis=-1)),
        "g_norm": jnp.mean(jnp.linalg.vector_norm(g,axis=-1)),
        "eig_abs": jnp.mean(jnp.abs(hd)),
        "eig_abs_std": jnp.std(jnp.abs(hd)),
        "eig_std": jnp.std(hd),
        "g_std": jnp.std(jnp.linalg.vector_norm(g,axis=-1)),
            'total_loss': total_loss,
            'bc_loss': bc_loss,
            'std': action_std.mean(),
            'mse': mse.mean(),
            "hd": jnp.mean(hd),
        }
        return total_loss, actor_info



def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='drebrac',  # Agent name.
            solver="auto",
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            tanh_squash=True,  # Whether to squash actions with tanh.
            clip=True,
            gn=200.0,
            delta=2.0,
            actor_fc_scale=0.01,  # Final layer initialization scale for actor.
            alpha=0.0,  # Actor BC coefficient.
            alpha_actor=0.0,  # Actor BC coefficient.
            alpha_critic=0.0,  # Critic BC coefficient.
            actor_freq=2,  # Actor update frequency.
            actor_noise=0.2,  # Actor noise scale.
            actor_noise_clip=0.5,  # Actor noise clipping threshold.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config