import functools
import glob
import os
import pickle
from typing import Any, Dict, Mapping, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import numpy as np
nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)
#from jaxopt import linear_solve

from functools import partial
import jaxopt
from jax import jvp
def hvp(grad_f, primals, tangents):
    return jvp(grad_f, primals, tangents)[1]

@jax.jit
def svd_computation(matrix):
    """
    使用 PyTorch 计算 SVD，支持 CUDA
    正确处理 JAX tracer 对象
    """
    # 首先使用 lax.stop_gradient 来阻止梯度追踪
  #  matrix_detached = lax.stop_gradient(matrix)
    
    # 将 JAX 数组转换为 numpy 数组
   # matrix_np = np.array(matrix)
    
    
    # 计算 SVD
    U, S, V = jnp.linalg.svd(matrix)
    
    # 转换回 JAX 数组
    U_jax = jnp.array(U)
    S_jax = jnp.array(S)
    V_jax = jnp.array(V)
    
    return U_jax, S_jax, V_jax
def clip(x):
    return jnp.clip(x,-1,1)
class DOALAgent(flax.struct.PyTreeNode):
    """Implicit Q-learning (IQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    def get_guided_action(self,q_action, action,observation,alpha,delta,params):
        return getattr(self, self.config["solver"] )(q_action, action,observation,alpha,delta,params)

    @jax.jit
    def bfgs(self, q_action, action, observation, alpha, delta, params):


        @jax.jit
        @partial(jax.vmap, in_axes=(0, 0, 0, None, None, None))
        def _get_guided_action(q_action, action, observation, alpha, delta, params):

            # --- HYPERPARAMETERS FOR MANUAL BFGS ---
            num_steps = self.config["num_steps"]
            step_size = 1.0 
            
            
            def bc_loss_wrt_q_action(q_action):
                qs = self.network.select('critic')(observation, q_action, params=params)
                q = jnp.mean(qs)
                return - q + alpha * jnp.sum((q_action - action)**2)

            

            def projected_step(last_action,H,grad_action):
                dz = -  jnp.linalg.solve(H, grad_action)
                unconstrained_action = last_action + dz
                
                distance = jnp.linalg.norm(dz)
                projected_action = jnp.where(
                    distance > delta,
                    last_action + dz * (delta / distance),
                    unconstrained_action
                )
                return projected_action

            def step(H , last_action, grad_action):

                final_step_action = projected_step(last_action,H,grad_action)
                
                grad_new_action = grad_f(final_step_action)
                dg = grad_new_action - grad_action
                actual_dz = final_step_action - q_action
                epsilon = 1e-8
                
                term1_num = jnp.outer(dg, dg)
                term1_den = jnp.dot(dg, actual_dz) + epsilon
                H_dz = H @ actual_dz
                term2_num = jnp.outer(H_dz, H_dz)
                term2_den = jnp.dot(actual_dz, H_dz) + epsilon
                H_new = H + (term1_num / term1_den) - (term2_num / term2_den)
                return H_new, final_step_action,grad_new_action
            grad_f = jax.grad(bc_loss_wrt_q_action)
            action_dim = q_action.shape[0]
            H =  2 * alpha * jnp.eye(action_dim)
            value, grad_action = jax.value_and_grad(bc_loss_wrt_q_action)(q_action)
            
            for i in range(num_steps):
                H,q_action ,grad_action = step(H,q_action ,grad_action)
            final_action = projected_step(q_action, H,grad_action)
            adjusted_actions = jax.lax.stop_gradient(final_action)
            
            # --- KEY CHANGE: Calculate Eigenvalues ---
            # We compute the eigenvalues of the final Hessian approximation.
            # `eigvalsh` is used because the Hessian is a symmetric matrix.
            hessian_eigenvalues = jnp.diagonal(H)
            
            # Final calculations for other return values
            dx = jax.lax.stop_gradient(adjusted_actions - action)
            q = jax.lax.stop_gradient(-value)
            g = jax.lax.stop_gradient(grad_action)

            # --- KEY CHANGE: Return Eigenvalues instead of 0*q ---
            return adjusted_actions, dx, jax.lax.stop_gradient(hessian_eigenvalues), g, q

        return _get_guided_action(q_action, action, observation, alpha, delta, params)


    @jax.jit
    def diag(self,q_action, action,observation,alpha,delta,params):


        @jax.jit
        @partial(jax.vmap, in_axes=(0,0,0,None,None))
        def _get_guided_action(q_action, action,observation,alpha,params):

            def bc_loss_wrt_q_action(q_action):
                qs = self.network.select('critic')(observation, q_action, params=params)
                q = jnp.mean(qs)
                return q 

            grad_q = jax.grad(bc_loss_wrt_q_action) 
            v_grad_q = jax.value_and_grad(bc_loss_wrt_q_action) 
            def hvp_dot_basis_vector(v):
                # jax.jvp computes the product of the Jacobian of grad_f (which is the Hessian)
                # with the vector v. It returns the primal output (grad_f(x)) and the tangent
                # output (the HVP). We only need the second part.
                hvp_val = jax.jvp(grad_q, (q_action,), (v,))[1]
                # Calculate the final dot product: v^T @ (H @ v)
                return jnp.vdot(v, hvp_val)
            basis = jnp.eye(q_action.shape[0], dtype=q_action.dtype)
            h_diagonal = jax.vmap(hvp_dot_basis_vector)(basis)
            q, g = v_grad_q(q_action)

            gap = 1  / (2 * alpha - h_diagonal )
            b =  g * jnp.clip (gap,min=0)
            normb = jnp.linalg.norm(b)
            dx = jnp.where(normb > delta,  b / normb,   b)
            
            adjusted_actions = jax.lax.stop_gradient(clip(q_action + dx))
            dx = jax.lax.stop_gradient(adjusted_actions - action)
            q =  jax.lax.stop_gradient(q)
            return  adjusted_actions, dx, h_diagonal, g, q

        return _get_guided_action(q_action, action,observation,alpha,params)
    @jax.jit
    def auto(self,q_action, action,observation,alpha,delta,params):


        def bc_loss_wrt_q_action(q_action):
            qs = self.network.select('critic')(observation, q_action, params=params)
            q = jnp.mean(qs,axis=0)
            return jnp.sum(q) 
    
        v_grad_q = jax.value_and_grad(bc_loss_wrt_q_action) 
        q, g = v_grad_q(q_action)

        g = g +  2 * alpha * (action-q_action)
        norm = jnp.linalg.norm(g,axis=-1,keepdims=True)
        norm_mean = jnp.mean(norm)
        norm_std = jnp.std(norm)
        norm_up = norm_mean + delta * norm_std

        clipped_g = jnp.where(norm > norm_up,  g * norm_up / norm,   g)
        dx =  clipped_g / ( norm_mean * alpha )

            
        adjusted_actions = jax.lax.stop_gradient(clip(q_action + dx))
        dx = jax.lax.stop_gradient(adjusted_actions - action)
        q =  jax.lax.stop_gradient(q)
        return adjusted_actions,dx,norm * alpha *  jnp.eye(q_action.shape[0], dtype=q_action.dtype),g,q


    @jax.jit
    def mpt_auto(self,q_action, action,observation,alpha,delta,params):


        @jax.jit
        @partial(jax.vmap, in_axes=(0,0,0,None,None))
        def _get_guided_action(q_action, action,observation,alpha,params):

            def bc_loss_wrt_q_action(q_action):
                qs = self.network.select('critic')(observation, q_action, params=params)
                return qs

            gs = jax.jacrev(bc_loss_wrt_q_action) (q_action)

            g = jnp.mean(gs,axis=0)
            norm = jnp.linalg.norm(g,axis=-1,keepdims=True)
            norm_mean = jnp.mean(norm)
            g = g + 2 * alpha * (action-q_action)

            alpha = alpha * norm_mean

            cov = jnp.cov(gs.T) +  2*alpha* jnp.eye(q_action.shape[0], dtype=q_action.dtype)
            b = jnp.linalg.solve(cov, g)

            normb = jnp.linalg.norm(b)
            dx = jnp.where(normb > delta,  b * delta/ normb,   b)
            
            adjusted_actions = jax.lax.stop_gradient(clip(q_action + dx))
            dx = jax.lax.stop_gradient(adjusted_actions - action)
            q = jax.lax.stop_gradient(0 * normb)
            return  adjusted_actions, dx, 2*  alpha *  jnp.eye(q_action.shape[0], dtype=q_action.dtype) ,g, q
        return _get_guided_action(q_action, action,observation,alpha,params)
    @jax.jit
    def mpt(self,q_action, action,observation,alpha,delta,params):


        @jax.jit
        @partial(jax.vmap, in_axes=(0,0,0,None,None))
        def _get_guided_action(q_action, action,observation,alpha,params):

            def bc_loss_wrt_q_action(q_action):
                qs = self.network.select('critic')(observation, q_action, params=params)
                return qs

            gs = jax.jacrev(bc_loss_wrt_q_action) (q_action)

            g = jnp.mean(gs,axis=0)
            cov = jnp.cov(gs.T) +  2*alpha* jnp.eye(q_action.shape[0], dtype=q_action.dtype)
            b = jnp.linalg.solve(cov, g)

            normb = jnp.linalg.norm(b)
            dx = jnp.where(normb > delta,  b * delta/ normb,   b)
            
            adjusted_actions = jax.lax.stop_gradient(clip(q_action + dx))
            dx = jax.lax.stop_gradient(adjusted_actions - action)
            q = jax.lax.stop_gradient(0 * normb)
            return  adjusted_actions, dx, 2*  alpha *  jnp.eye(q_action.shape[0], dtype=q_action.dtype) ,g, q
        return _get_guided_action(q_action, action,observation,alpha,params)
    @jax.jit
    def linear(self,q_action, action,observation,alpha,delta,params):


        @jax.jit
        @partial(jax.vmap, in_axes=(0,0,0,None,None))
        def _get_guided_action(q_action, action,observation,alpha,params):

            def bc_loss_wrt_q_action(q_action):
                qs = self.network.select('critic')(observation, q_action, params=params)
                q = jnp.mean(qs)
                return q 

            v_grad_q = jax.value_and_grad(bc_loss_wrt_q_action) 
            q, g = v_grad_q(q_action)

            g = g +  2 * alpha * (action-q_action)
            b =  g / (  2 * alpha )

            normb = jnp.linalg.norm(b)
            dx = jnp.where(normb > delta,  b * delta/ normb,   b)
            
            adjusted_actions = jax.lax.stop_gradient(clip(q_action + dx))
            dx = jax.lax.stop_gradient(adjusted_actions - action)
            q =  jax.lax.stop_gradient(q)
            return  adjusted_actions, dx, 2*  alpha *  jnp.eye(q_action.shape[0], dtype=q_action.dtype) ,g, q
        return _get_guided_action(q_action, action,observation,alpha,params)
# Assume self.network is defined elsewhere

    @jax.jit
    def full(self, q_action, action, observation, alpha, delta, params):

        @jax.jit
        @partial(jax.vmap, in_axes=(0, 0, 0, None, None))
        def _get_guided_action(q_action, action, observation, alpha, params):

            # 1. Define the regularized objective function to be MINIMIZED.

            def bc_loss_wrt_q_action(q_action):
                qs = self.network.select('critic')(observation, q_action, params=params)
                q = jnp.mean(qs)
                return - q + alpha * jnp.sum((q_action - action)**2)


            def projected_step(last_action,H,grad_action):
                dz = -  jnp.linalg.solve(H, grad_action)
                unconstrained_action = last_action + dz
                
                distance = jnp.linalg.norm(dz)
                projected_action = jnp.where(
                    distance > delta,
                    last_action + dz * (delta / distance),
                    unconstrained_action
                )
                return jnp.clip(projected_action, -1.0, 1.0)



            q_final,grad_action = jax.value_and_grad(bc_loss_wrt_q_action)(q_action)
            H = jax.hessian(bc_loss_wrt_q_action)(q_action)

            adjusted_actions = projected_step(q_action,H,grad_action)

            # 4. Extract the results.
            adjusted_actions = jax.lax.stop_gradient(adjusted_actions)
            
            dx = jax.lax.stop_gradient(adjusted_actions - action)
            q = jax.lax.stop_gradient(q_final)
            
            g = jax.lax.stop_gradient(grad_action)

            eig =   jnp.diagonal(H)#jax.scipy.linalg.svd(H,full_matrices =False,compute_uv =False)
            return adjusted_actions, dx,eig, g, q

        return _get_guided_action(q_action, action, observation, alpha, params)

    @jax.jit
    def auto_trust(self, q_action, action, observation, alpha, delta, params):


        @jax.jit
        @partial(jax.vmap, in_axes=(0, 0, 0, None, None))
        def _get_svd(q_action, action, observation, alpha, params):

            # 1. Define the regularized objective function to be MINIMIZED.

            def bc_loss_wrt_q_action(q_action):
                qs = self.network.select('critic')(observation, q_action, params=params)
                q = jnp.mean(qs)
                return - q 



            q_final,grad_action = jax.value_and_grad(bc_loss_wrt_q_action)(q_action)
         #   grad_action = grad_action + 2 * (q_action-action)
            H = jax.hessian(bc_loss_wrt_q_action)(q_action)
           # H = make_hessian_psd_gershgorin(H,alpha)
            U, S, V =  jnp.linalg.svd(H)
            return U, S, V ,q_final,grad_action
        U, S, V, q_final,grad_action = _get_svd(q_action, action, observation, alpha, params)
        eigvals = jnp.abs(S)
        @jax.vmap
        def make_inv_h(U, S):
            return  jnp.dot(U, jnp.dot(jnp.diag(1.0/S), U.T))
        h_std = jnp.std(eigvals)
        inv_H = make_inv_h(U,eigvals +1e-4)

        dx = alpha * jax.lax.batch_matmul (inv_H , grad_action )
        unconstrained_action = q_action - dx
        distance = jnp.linalg.norm(dx)
        projected_action = jnp.where(
            distance > delta,
            q_action + dx * (delta / distance),
            unconstrained_action
        )
        adjusted_actions = jnp.clip(projected_action, -1.0, 1.0)


        # 4. Extract the results.
        adjusted_actions = jax.lax.stop_gradient(adjusted_actions)
        
        dx = jax.lax.stop_gradient(adjusted_actions - action)
        q = jax.lax.stop_gradient(q_final)
        
        g = jax.lax.stop_gradient(grad_action)

        eig =  jax.lax.stop_gradient(S)#jax.scipy.linalg.svd(H,full_matrices =False,compute_uv =False)

        return adjusted_actions, dx,eig, g, q
    @jax.jit
    def trust(self, q_action, action, observation, alpha, delta, params):

        @jax.jit
        @partial(jax.vmap, in_axes=(0, 0, 0, None, None))
        def _get_guided_action(q_action, action, observation, alpha, params):

            # 1. Define the regularized objective function to be MINIMIZED.

            def bc_loss_wrt_q_action(q_action):
                qs = self.network.select('critic')(observation, q_action, params=params)
                q = jnp.mean(qs)
                return - q 


            def projected_step(last_action,H,grad_action):
                dz = -  jnp.linalg.solve(H, grad_action)
                unconstrained_action = last_action + dz
                
                distance = jnp.linalg.norm(dz)
                projected_action = jnp.where(
                    distance > delta,
                    last_action + dz * (delta / distance),
                    unconstrained_action
                )
                return jnp.clip(projected_action, -1.0, 1.0)



            q_final,grad_action = jax.value_and_grad(bc_loss_wrt_q_action)(q_action)
         #   grad_action = grad_action + 2 * (q_action-action)
            H = jax.hessian(bc_loss_wrt_q_action)(q_action)
           # H = make_hessian_psd_gershgorin(H,alpha)
            S, V, D =  jnp.linalg.svd(H)
            eigvals = jnp.abs(V)
            H = jnp.dot(S, jnp.dot(jnp.diag(eigvals+2*alpha), D.T))

            adjusted_actions = projected_step(q_action,H,grad_action)

            # 4. Extract the results.
            adjusted_actions = jax.lax.stop_gradient(adjusted_actions)
            
            dx = jax.lax.stop_gradient(adjusted_actions - action)
            q = jax.lax.stop_gradient(q_final)
            
            g = jax.lax.stop_gradient(grad_action)

            eig =  V#jax.scipy.linalg.svd(H,full_matrices =False,compute_uv =False)
            return adjusted_actions, dx,eig, g, q

        return _get_guided_action(q_action, action, observation, alpha, params)
class ModuleDict(nn.Module):
    """A dictionary of modules.

    This allows sharing parameters between modules and provides a convenient way to access them.

    Attributes:
        modules: Dictionary of modules.
    """

    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        """Forward pass.

        For initialization, call with `name=None` and provide the arguments for each module in `kwargs`.
        Otherwise, call with `name=<module_name>` and provide the arguments for that module.
        """
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f'When `name` is not specified, kwargs must contain the arguments for each module. '
                    f'Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}'
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    """Custom train state for models.

    Attributes:
        step: Counter to keep track of the training steps. It is incremented by 1 after each `apply_gradients` call.
        apply_fn: Apply function of the model.
        model_def: Model definition.
        params: Parameters of the model.
        tx: optax optimizer.
        opt_state: Optimizer state.
    """

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        """Create a new train state."""
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        """Forward pass.

        When `params` is not provided, it uses the stored parameters.

        The typical use case is to set `params` to `None` when you want to *stop* the gradients, and to pass the current
        traced parameters when you want to flow the gradients. In other words, the default behavior is to stop the
        gradients, and you need to explicitly provide the parameters to flow the gradients.

        Args:
            *args: Arguments to pass to the model.
            params: Parameters to use for the forward pass. If `None`, it uses the stored parameters, without flowing
                the gradients.
            method: Method to call in the model. If `None`, it uses the default `apply` method.
            **kwargs: Keyword arguments to pass to the model.
        """
        if params is None:
            params = self.params
        variables = {'params': params}
        if method is not None:
            method_name = getattr(self.model_def, method)
        else:
            method_name = None

        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        """Helper function to select a module from a `ModuleDict`."""
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        """Apply the gradients and return the updated state."""
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, loss_fn):
        """Apply the loss function and return the updated state and info.

        It additionally computes the gradient statistics and adds them to the dictionary.
        """
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)

        grad_max = jax.tree_util.tree_map(jnp.max, grads)
        grad_min = jax.tree_util.tree_map(jnp.min, grads)
        grad_norm = jax.tree_util.tree_map(jnp.linalg.norm, grads)

        grad_max_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_max)], axis=0)
        grad_min_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_min)], axis=0)
        grad_norm_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_norm)], axis=0)

        final_grad_max = jnp.max(grad_max_flat)
        final_grad_min = jnp.min(grad_min_flat)
        final_grad_norm = jnp.linalg.norm(grad_norm_flat, ord=1)

        info.update(
            {
                'grad/max': final_grad_max,
                'grad/min': final_grad_min,
                'grad/norm': final_grad_norm,
            }
        )

        return self.apply_gradients(grads=grads), info


def save_agent(agent, save_dir, epoch):
    """Save the agent to a file.

    Args:
        agent: Agent.
        save_dir: Directory to save the agent.
        epoch: Epoch number.
    """

    save_dict = dict(
        agent=flax.serialization.to_state_dict(agent),
    )
    save_path = os.path.join(save_dir, f'params_{epoch}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f'Saved to {save_path}')


def restore_agent(agent, restore_path, restore_epoch):
    """Restore the agent from a file.

    Args:
        agent: Agent.
        restore_path: Path to the directory containing the saved agent.
        restore_epoch: Epoch number.
    """
    candidates = glob.glob(restore_path)

    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

    restore_path = candidates[0] + f'/params_{restore_epoch}.pkl'

    with open(restore_path, 'rb') as f:
        load_dict = pickle.load(f)

    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])

    print(f'Restored from {restore_path}')

    return agent
