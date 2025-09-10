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
from utils.networks import ActorVectorField, Value,TimeWeight
from functools import partial
import math


class TrigFQLAgent(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent."""

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
        lam = 1 / jax.lax.stop_gradient(jnp.abs(v).mean())
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean() 

        if self.config['normalize_q_loss']:
            value_loss = lam * value_loss


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

        if self.config['normalize_q_loss']:
            critic_loss = aux["lam"] * critic_loss
        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }, aux

        
    def actor_loss(self, batch, grad_params, rng=None,aux={}):
        """Compute the FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        z = jax.random.normal(x_rng, (batch_size, action_dim))
        t = jax.random.uniform(t_rng, (batch_size, 1))  *math.pi / 2
        x_t = jnp.cos(t)* batch['actions'] + jnp.sin(t) * z

        vel =  jnp.cos(t)* z  - jnp.sin(t) * batch['actions']


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
        if self.config["distill_from_target"]:
            qs = self.network.select('target_critic')(batch['observations'], actions=pred_actions)
        else:
            qs = self.network.select('critic')(batch['observations'], actions=pred_actions)
        if self.config['q_agg'] == 'min':
            q = jnp.min(qs, axis=0)
        else:
            q = jnp.mean(qs, axis=0)


        actor_loss = -q.mean()
        # Total loss.
        total_loss = actor_loss

        out = {
                'q': q.mean(),
                "weight":weight.mean(),
                'actor_loss': actor_loss,
            }
        if self.config["vel_actor"] > 0:

            raw_bc_flow_loss = (( F_theta  - vel ) ** 2 ) .mean() #/ jnp.sin(t).clip(min=0.1)
            bc_flow_loss =  (weight* ( F_theta  - vel ) ** 2 -time_weight_logits) .mean()  #/ jnp.sin(t).clip(min=0.1)
            total_loss = total_loss + self.config['vel_actor'] * bc_flow_loss
            out["bc_flow_loss"]  = raw_bc_flow_loss
        if self.config["alpha_actor"] > 0:
            raw_zero_shot_loss = ( ( pred_actions- batch['actions'] ) ** 2).mean()   
            zero_shot_loss = ( weight*  ( pred_actions- batch['actions'] ) ** 2 -time_weight_logits).mean()   
            total_loss = total_loss  +  self.config["alpha_actor"]  *    zero_shot_loss 
            out["zero_shot_loss"]  = raw_zero_shot_loss
        
        out['total_loss'] = total_loss
        return total_loss, out 


    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info, aux = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f"value/{k}"] = v

        critic_loss, critic_info, aux = self.critic_loss(batch, grad_params, aux)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng, aux)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = value_loss + critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "critic")

        return self.replace(network=new_network, rng=new_rng), info
        
    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        orig_observations = observations
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_flow_encoder')(observations)
        action_seed, noise_seed = jax.random.split(seed)

        # Sample `num_samples` noises and propagate them through the flow.
        actions = jax.random.normal(
            action_seed,
            (
                *observations.shape[:-1],
                self.config['num_samples'],
                self.config['action_dim'],
            ),
        ) 
        n_observations = jnp.repeat(jnp.expand_dims(observations, 0), self.config['num_samples'], axis=0)
        n_orig_observations = jnp.repeat(jnp.expand_dims(orig_observations, 0), self.config['num_samples'], axis=0)
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1],self.config['num_samples'], 1), (1.0 - i / self.config['flow_steps']) *math.pi / 2)
            vels = self.network.select('actor_bc_flow')(n_observations, actions, t, is_encoded=True)
            s = jnp.full((*observations.shape[:-1], self.config['num_samples'],1), (1.0 - (i+1) / self.config['flow_steps']) *math.pi / 2)
            actions = actions * jnp.cos(t-s) - vels * jnp.sin(t-s) #* self.config["sigma"]

        actions = jnp.clip(actions, -1, 1)
        # Pick the action with the highest Q-value.
        q = self.network.select('critic')(n_orig_observations, actions=actions).min(axis=0)
        actions = actions[jnp.argmax(q)]
        return actions
    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = encoder_module()
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()


        # Define networks.
        value_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('value'),
        )
        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )

        time_weight_def = TimeWeight(
            hidden_dims=config['time_hidden_dims'],
            layer_norm=config['layer_norm'],
        )
        network_info = dict(
            value=(value_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
            time_weight = (time_weight_def, (ex_times)),
        )
        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        
        network_def = ModuleDict(networks)
        if config["gn"] > 0:
            network_tx = optax.chain(
                optax.clip_by_block_rms(config["gn"]),
                optax.adam(learning_rate=config['lr'])
            )
        else:
            network_tx =     optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
   #     config["sigma"] = jnp.std(ex_actions,axis=0,keepdims=True)
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='trigflow',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=( 512, 512,512, 512),  # Value network hidden dimensions.
            time_hidden_dims=(32,),
     #       sigma=ml_collections.config_dict.placeholder(jax.Array),
            normalize_action=False,
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='min',  # Aggregation method for target Q values.
            q_steps=10,
            return_next_actions=True,
            decode_type="ddim",
            alpha_actor=10.0,  # BC coefficient (need to be tuned for each environment).
            distill_from_target=False,  # BC coefficient (need to be tuned for each environment).
            time_weight=False,
            expectile=0.9,  # IQL expectile.
            gn=10.0,
            vel_actor = 0.0,
            alpha_critic=0.0,  # Critic BC coefficient.
            num_samples=32,  # Number of action samples for rejection sampling.
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
