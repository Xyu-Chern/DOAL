import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field,DOALAgent
from utils.networks import ActorVectorField, Value


from functools import partial
from agents.ifql import IFQLAgent
class DIFQLAgent(DOALAgent,IFQLAgent):
    """Implicit flow Q-learning (IFQL) agent.

    IFQL is the flow variant of implicit diffusion Q-learning (IDQL).
    """

    def actor_loss(self, batch, grad_params, rng=None,aux={}):
        """Compute the behavioral flow-matching actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        
        alpha = self.config['alpha'] 
        adjusted_actions , adjustment,hd,g, q = self.get_guided_action(  x_1, x_1,batch['observations'],alpha,delta=self.config["delta"],params=self.network.params)
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * adjusted_actions
        vel = x_1 - x_0

        v = jax.lax.stop_gradient(aux["v"])
        q = jax.lax.stop_gradient(aux["q"])
        adv = q - v

        exp_a = jnp.exp(adv * self.config["alpha_actor"])
        exp_a = jnp.expand_dims( jnp.minimum(exp_a, 100.0),1)

        pred = self.network.select('actor_flow')(batch['observations'], x_t, t, params=grad_params)

        raw_actor_loss = (pred - vel) ** 2
        actor_loss = jnp.mean(exp_a* raw_actor_loss)

        return actor_loss, {
            "raw_actor_loss":jnp.mean(raw_actor_loss),
            'actor_loss': actor_loss,
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


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='difql',  # Agent name.
            solver="linear",
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            alpha_actor=10.0,
            alpha=1.0,  # BC coefficient (need to be tuned for each environment).
            delta=1.0,
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            gn=0.0,
            num_samples=32,  # Number of action samples for rejection sampling.
            flow_steps=10,  # Number of flow steps.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
