import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field,DOALAgent
from utils.networks import ActorVectorField, Value
from jax.lax import stop_gradient
from functools import partial
import math

from agents.trigflow import TrigFQLAgent

class DTrigFQLAgent(DOALAgent,TrigFQLAgent):
    """Flow Q-learning (FQL) agent."""

    def actor_loss(self, batch, grad_params, rng=None,aux={}):
        """Compute the FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        alpha = self.config["alpha"] 
        adjusted_actions , adjustment, q = self.get_guided_action(  batch['actions'], batch['actions'],batch['observations'],alpha=alpha,delta=self.config["delta"],params=self.network.params)

        # BC flow loss.
        z = jax.random.normal(x_rng, (batch_size, action_dim))
        t = jax.random.uniform(t_rng, (batch_size, 1))  *math.pi / 2
        x_t = jnp.cos(t)* adjusted_actions + jnp.sin(t) * z

        vel =  jnp.cos(t)* z  - jnp.sin(t) * adjusted_actions


        F_theta = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)


        bc_flow_loss = (( F_theta - vel ) ** 2).mean()  #/ jnp.sin(t).clip(min=0.1)

        # Total loss.
        total_loss = self.config['alpha_actor'] * bc_flow_loss 
        return total_loss, {
            'total_loss': total_loss,
            "bc_flow_loss":bc_flow_loss,
            'adj': jnp.mean(jnp.abs(adjustment)),
            'q': q.mean(),
        }



def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='dtrigflow',  # Agent name.
            solver="linear",
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='min',  # Aggregation method for target Q values.
            expectile=0.9,  # IQL expectile.
            gn=100.0,
            return_next_actions=True,
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            alpha_actor=10.0,  # BC coefficient (need to be tuned for each environment).
            delta=0.1,
            num_samples=32,  # Number of action samples for rejection sampling.
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
