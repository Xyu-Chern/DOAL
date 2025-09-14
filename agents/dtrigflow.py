import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value
from jax.lax import stop_gradient
from functools import partial
import math

from agents.trigflow import TrigFQLAgent

class DTrigFQLAgent(TrigFQLAgent):
    """Flow Q-learning (FQL) agent."""


    def actor_loss(self, batch, grad_params, rng=None,aux={}):
        """Compute the FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        alpha = self.config["alpha_actor"] 
        adjusted_actions , adjustment,hd,g, q = self.get_guided_action(  batch['actions'], batch['actions'],batch['observations'],alpha=alpha,delta=self.config["delta"],params=self.network.params)


        # BC flow loss.
        z = jax.random.normal(x_rng, (batch_size, action_dim))
        t = jax.random.uniform(t_rng, (batch_size, 1))  *math.pi / 2

    #    vel =  jnp.cos(t)* z  - jnp.sin(t) * adjusted_actions

        x_t = jnp.cos(t)*  adjusted_actions + jnp.sin(t) * z

        F_theta = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        pred_actions = x_t * jnp.cos(t) - F_theta * jnp.sin(t) 


        #、 v = jax.lax.stop_gradient(aux["v"])
       #  q = jax.lax.stop_gradient(aux["q"])
       #  adv = q - v

       #  exp_a = jnp.exp(adv * self.config["alpha_actor"])
       # exp_a = jnp.expand_dims( jnp.minimum(exp_a, 100.0),1)

        if self.config["time_weight"]:
            time_weight_logits = self.network.select("time_weight")(t, params=grad_params)
            weight =   jnp.exp(time_weight_logits) / action_dim
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
            'adj_norm': jnp.mean(jnp.linalg.vector_norm(adjustment,axis=-1)),
            'adj': jnp.mean(jnp.abs(adjustment)),
            "hd": jnp.mean(hd),
            "hd_abs": jnp.mean(jnp.abs(hd)),
            "hd_std": jnp.std(hd),
            "hd_max": jnp.max(hd),
            "hd_min": jnp.min(hd),
            "g": jnp.mean(g),
            "g_abs": jnp.mean(jnp.abs(g)),
            "g_std": jnp.std(g),
            "g_max": jnp.max(g),
            "g_min": jnp.min(g),
            }

        raw_zero_shot_loss = ( ( pred_actions- batch['actions'] ) ** 2)
        zero_shot_loss = ( weight*  raw_zero_shot_loss -time_weight_logits).mean()   
        total_loss = total_loss  + self.config["alpha_actor"] *  zero_shot_loss 
        out["zero_shot_loss"]  = raw_zero_shot_loss.mean()   

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
            step_size=1.0,  # IQL expectile.
            num_steps=1,  # IQL expectile.
            gn=0.0,
            return_next_actions=True,
            time_weight=False,
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            test_alpha=0.0,
            alpha_actor=10.0,  # BC coefficient (need to be tuned for each environment).
            use_vel_loss=False,  # BC coefficient (need to be tuned for each environment).
            delta=1.0,
            num_samples=32,  # Number of action samples for rejection sampling.
            flow_steps=10,  # Number of flow steps.
            use_q_loss=False,  # Whether to normalize the Q loss.
            test_guidance=False,
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
