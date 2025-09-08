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

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params, aux={}):
        """Compute the IQL value loss."""
        q1, q2 = self.network.select("target_critic")(
            batch["observations"], actions=batch["actions"]
        )
        q = jnp.minimum(q1, q2)
        v = self.network.select("value")(batch["observations"], params=grad_params)        
        lam = 1 / jax.lax.stop_gradient(jnp.abs(q).mean())
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()  


        aux.update({"v": v,"q": q,"lam": lam })
        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }, aux

    def critic_loss(self, batch, grad_params, aux={}):
        """Compute the IQL critic loss."""
        next_v = self.network.select("value")(batch["next_observations"])
        q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v

        q1, q2 = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean() 

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }, aux
    def actor_loss(self, batch, grad_params, rng=None,aux={}):
        """Compute the behavioral flow-matching actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        
        alpha = self.config['alpha'] 
        adjusted_actions , adjustment,hd, q = self.get_guided_action(  x_1, x_1,batch['observations'],alpha,delta=self.config["delta"],params=self.network.params)
        q1, q2 = self.network.select('critic')(batch['observations'], actions=adjusted_actions)
        aq = jnp.minimum(q1, q2)
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * adjusted_actions
        vel = x_1 - x_0

        pred = self.network.select('actor_flow')(batch['observations'], x_t, t, params=grad_params)
        actor_loss = self.config['alpha_actor'] *  jnp.mean((pred - vel) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'adj': jnp.mean(jnp.abs(adjustment)),
            'aq': jnp.mean(aq),
            "hd"; jnp.mean(hd),
        }


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='difql',  # Agent name.
            solver="diag_hess",
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            alpha_actor=10,
            alpha=1.0,  # BC coefficient (need to be tuned for each environment).
            delta=1.0,
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            gn=100.0,
            num_samples=32,  # Number of action samples for rejection sampling.
            flow_steps=10,  # Number of flow steps.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
