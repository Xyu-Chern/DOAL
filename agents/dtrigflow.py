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

        alpha = self.config["alpha"] 

        adjusted_actions , adjustment,hd,g, q = self.get_guided_action(  batch['actions'], batch['actions'],batch['observations'],alpha=alpha,delta=self.config["delta"],params=self.network.params)


        # BC flow loss.
        z = jax.random.normal(x_rng, (batch_size, action_dim))
        t = jax.random.uniform(t_rng, (batch_size, 1))  *math.pi  / 2

    #    vel =  jnp.cos(t)* z  - jnp.sin(t) * adjusted_actions
    # need ablation study 
        if self.config["use_acton_for_sample"]:
            x_t = jnp.cos(t)*   batch['actions'] + jnp.sin(t) * z
        else:
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
            'adj_std': jnp.std(jnp.linalg.vector_norm(adjustment,axis=-1)),
            "g_norm": jnp.mean(jnp.linalg.vector_norm(g,axis=-1)),
            "g_std": jnp.std(jnp.linalg.vector_norm(g,axis=-1)),
            }
        if  self.config["loss_type"] == "action":
            raw_zero_shot_loss = ( ( pred_actions- adjusted_actions ) ** 2)
            bc_flow_loss = ( weight*  raw_zero_shot_loss -time_weight_logits).mean()   
            total_loss = total_loss  + self.config["alpha_actor"] *  bc_flow_loss 
            out["bc_flow_loss"]  = raw_zero_shot_loss.mean()   
        elif  self.config["loss_type"] ==  "vel":
            vel =  jnp.cos(t)* z  - jnp.sin(t) * adjusted_actions
            raw_vel_loss = ( ( F_theta- vel ) ** 2)
            bc_flow_loss = ( weight*  raw_vel_loss -time_weight_logits).mean()   
            total_loss = total_loss  + self.config["alpha_actor"] *  bc_flow_loss 
            out["bc_flow_loss"]  = raw_vel_loss.mean()   
        else:
            assert False, self.config["loss_type"]+" does not exist"
        out['total_loss'] = total_loss
        return total_loss, out 


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='dtrigflow',  # Agent name.
            solver="auto",
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
            gn=10.0,
            return_next_actions=True,
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            time_weight=False,
            alpha=50.0,  # BC coefficient (need to be tuned for each environment).
            test_alpha=0.0,
            alpha_actor=10.0,  # BC coefficient (need to be tuned for each environment).
            use_vel_loss=False,  # BC coefficient (need to be tuned for each environment).
            loss_type="action",
            norm_q_grad=False,
            clip=False,
            use_acton_for_sample=False,
            delta=1.0,
            num_samples=32,  # Number of action samples for rejection sampling.
            flow_steps=10,  # Number of flow steps.
            use_q_loss=False,  # Whether to normalize the Q loss.
            test_guidance=False,
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
