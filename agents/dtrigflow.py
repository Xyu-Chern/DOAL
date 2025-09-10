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
        adjusted_actions , adjustment,hd,g, q = self.get_guided_action(  batch['actions'], batch['actions'],batch['observations'],alpha=alpha,delta=self.config["delta"],params=self.network.params)


        # BC flow loss.
        z = jax.random.normal(x_rng, (batch_size, action_dim))
        t = jax.random.uniform(t_rng, (batch_size, 1))  *math.pi / 2
        x_t = jnp.cos(t)* adjusted_actions + jnp.sin(t) * z

        vel =  jnp.cos(t)* z  - jnp.sin(t) * adjusted_actions


        F_theta = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        pred_actions = x_t * jnp.cos(t) - F_theta * jnp.sin(t) 


      #  v = jax.lax.stop_gradient(aux["v"])
      #  q = jax.lax.stop_gradient(aux["q"])
      #  adv = q - v

     #   exp_a = jnp.exp(adv * self.config['vel_actor'])
      #  exp_a = jnp.minimum(exp_a, 100.0)

        if self.config["time_weight"]:
            time_weight_logits = self.network.select("time_weight")(t, params=grad_params)
            weight = jnp.exp(time_weight_logits) / action_dim
            time_weight_logits = time_weight_logits - jax.lax.stop_gradient(time_weight_logits)
        else:
            weight = jnp.ones_like(t) 
            time_weight_logits = jnp.zeros_like(t) 


        actor_loss = -q.mean()
        # Total loss.
        total_loss = actor_loss

        out = {
                'q': q.mean(),
                "weight":weight.mean(),
                'actor_loss': actor_loss,
                'adj': jnp.mean(jnp.abs(adjustment)),
            "hd": jnp.mean(hd),
            "hd_abs": jnp.mean(jnp.abs(hd)),
            "hd_std": jnp.std(hd),
            "hd_max": jnp.max(hd),
            "hd_min": jnp.min(hd),
            "g": jnp.mean(g),
            "g_std": jnp.std(g),
            "g_abs": jnp.mean(jnp.abs(g)),
            "g_std": jnp.std(g),
            "g_max": jnp.max(g),
            "g_min": jnp.min(g),
            }
        if self.config["vel_actor"] > 0:

            raw_bc_flow_loss = (( F_theta  - vel ) ** 2 ) .mean() #/ jnp.sin(t).clip(min=0.1)
            bc_flow_loss =  (weight* ( F_theta  - vel ) ** 2 -time_weight_logits) .mean()  #/ jnp.sin(t).clip(min=0.1)
            total_loss = total_loss + self.config['vel_actor'] * bc_flow_loss
            out["bc_flow_loss"]  = raw_bc_flow_loss
        if self.config["alpha_actor"] > 0:
            raw_zero_shot_loss = ( ( pred_actions-adjusted_actions ) ** 2).mean()   
            zero_shot_loss = ( weight*  ( pred_actions-adjusted_actions ) ** 2 -time_weight_logits).mean()   
            total_loss = total_loss  +  self.config["alpha_actor"]  *    zero_shot_loss 
            out["zero_shot_loss"]  = raw_zero_shot_loss
        
        out['total_loss'] = total_loss
        return total_loss, out 


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
            value_hidden_dims=(512, 512,512, 512),  # Value network hidden dimensions.
            time_hidden_dims=(32,),
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            distill_factor=0,
            q_agg='min',  # Aggregation method for target Q values.
            expectile=0.9,  # IQL expectile.
            gn=0.0,
            return_next_actions=True,
            time_weight=True,
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            alpha_actor=10.0,  # BC coefficient (need to be tuned for each environment).
            vel_actor=0.0,  # BC coefficient (need to be tuned for each environment).
            delta=0.2,
            num_samples=32,  # Number of action samples for rejection sampling.
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
