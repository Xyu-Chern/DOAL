import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
import math
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field,DOALAgent
from utils.networks import Actor, Value

from jaxopt import linear_solve

from functools import partial
from agents.iql import IQLAgent
class DIQLAgent(DOALAgent,IQLAgent):
    """Implicit Q-learning (IQL) agent."""

    def actor_loss(self, batch, grad_params, rng=None,aux={}):
        """Compute the actor loss (AWR or DDPG+BC)."""
        if self.config['actor_loss'] == 'awr':

            alpha = self.config["alpha"] 
            adjusted_actions , adjustment, q = self.get_guided_action(  batch['actions'], batch['actions'],batch['observations'],alpha=alpha,delta=self.config["delta"],params=self.network.params)
            # AWR loss.
            v = jax.lax.stop_gradient(aux["v"])
            q1, q2 = self.network.select('critic')(batch['observations'], actions=adjusted_actions)
            aq = jnp.minimum(q1, q2)
            adv = aq - v

            exp_a = jnp.exp(adv * self.config['alpha_actor'])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = self.network.select('actor')(batch['observations'], params=grad_params)
            log_prob = dist.log_prob(adjusted_actions)

            actor_loss = -(exp_a * log_prob).mean()

            actor_info = {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                "alpha":alpha,
                'adj': jnp.mean(jnp.abs(adjustment)),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'q': jnp.mean(q),
                'aq': jnp.mean(aq),
                'std': jnp.mean(dist.scale_diag),
            }

            return actor_loss, actor_info
        elif self.config['actor_loss'] == 'ddpgbc':
            adjusted_actions , adjustment, q = self.get_guided_action(  batch['actions'], batch['actions'],batch['observations'],alpha=self.config["alpha"],delta=self.config["delta"],params=self.network.params)
            # DDPG+BC loss.
            dist = self.network.select('actor')(batch['observations'], params=grad_params)

            log_prob = dist.log_prob(batch['actions'])

            bc_loss = -(self.config['alpha_actor'] * log_prob).mean()

            actor_loss = bc_loss

            return actor_loss, {
                'actor_loss': actor_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')



def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='diql',  # Agent name.
            solver="diag_hess",
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.  , 512, 512
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.  , 512, 512
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            gn=100.0,
            actor_loss='awr',  # Actor loss type ('awr' or 'ddpgbc').
            actor_update_start=0.,
            alpha=10.0,  # Temperature in AWR or BC coefficient in DDPG+BC.
            alpha_actor = 10.0,
            delta=0.1,
            const_std=True,  # Whether to use constant standard deviation for the actor.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
