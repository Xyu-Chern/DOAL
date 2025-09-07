import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, DOALAgent
from utils.networks import ActorVectorField, Value
from agents.fql import FQLAgent
from functools import partial
import math
class DFQLAgent(DOALAgent,FQLAgent):
    """Flow Q-learning (FQL) agent."""


    def actor_loss(self, batch, grad_params, rng,aux):
        """Compute the FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # Distillation loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)

        alpha = self.config["alpha"] 
        adjusted_actions , adjustment, q = self.get_guided_action(  target_flow_actions, target_flow_actions,batch['observations'],alpha=alpha,delta=self.config["delta"],params=self.network.params)
        actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
        distill_loss = jnp.mean((actor_actions - adjusted_actions) ** 2)


        # Total loss.
        actor_loss = bc_flow_loss + self.config['alpha_actor'] *  distill_loss 

        # Additional metrics for logging.
        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q': q.mean(),
            'mse': mse,
        }






def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='dfql',  # Agent name.
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
            adjusted_target=True,
            gn=10.0,
            q_agg='min',  # Aggregation method for target Q values.
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            alpha_actor=10.0,  # this is the alpha in fql, we need to use hps to configure
            delta=0.1,
            flow_steps=10,  # Number of flow steps.
            return_next_actions=False,
            normalize_q_loss=True,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
