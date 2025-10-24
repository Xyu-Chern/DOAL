
import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
import math
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, DOALAgent
from utils.networks import ActorVectorField, Value
from agents.trigql import TrigQLAgent

class DTrigQLAgent(TrigQLAgent,DOALAgent):
    """Flow Q-learning (FQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def actor_loss(self, batch, grad_params, rng=None,aux={}):
        """Compute the FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)


        # we should call this delta in paper 
        alpha = self.config["alpha"] 
        adjusted_actions , adjustment,hd,g, q = self.get_guided_action(  batch['actions'], batch['actions'],batch['observations'],alpha=alpha,delta=self.config["delta"],params=self.network.params)


        z = jax.random.normal(x_rng, (batch_size, action_dim))
        t = jax.random.uniform(t_rng, (batch_size, 1))  *math.pi  / 2

        x_t = jnp.cos(t)*  adjusted_actions + jnp.sin(t) * z

        F_theta = self.network.select('actor_flow')(batch['observations'], x_t, t, params=grad_params)
        pred_actions = x_t * jnp.cos(t) - F_theta * jnp.sin(t) 


        weight = jnp.ones_like(t) 
        time_weight_logits = jnp.zeros_like(t) 


        actor_loss = -q.mean()
        # Total loss.
        total_loss = actor_loss

        out = {'q': q.mean(),
                "weight":weight.mean(),
                'actor_loss': actor_loss,
            'adj_norm': jnp.mean(jnp.linalg.vector_norm(adjustment,axis=-1)),
            'adj_std': jnp.std(jnp.linalg.vector_norm(adjustment,axis=-1)),
            "g_norm": jnp.mean(jnp.linalg.vector_norm(g,axis=-1)),
            "eig_abs": jnp.mean(jnp.abs(hd)),
            "eig_abs_std": jnp.std(jnp.abs(hd)),
            "eig_std": jnp.std(hd),
            "g_std": jnp.std(jnp.linalg.vector_norm(g,axis=-1)),
            }

        raw_zero_shot_loss = ( ( pred_actions- adjusted_actions ) ** 2)
        bc_flow_loss = ( weight*  raw_zero_shot_loss -time_weight_logits).mean()   
        total_loss = total_loss  + self.config["alpha_actor"] *  bc_flow_loss 
        out["bc_flow_loss"]  = raw_zero_shot_loss.mean()   
        out['total_loss'] = total_loss
        return total_loss, out 


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='dtrigql',  # Agent name.
            solver="auto",  
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
            return_next_actions=True,
            q_agg='mean',  # Aggregation method for target Q values.
            target_num_samples = 4,
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            flow_steps=10,  # Number of flow steps.
            delta=2.0,
            clip=True,
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config