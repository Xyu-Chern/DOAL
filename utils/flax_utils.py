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

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)
from jaxopt import linear_solve

from functools import partial
import jaxopt
from jax import jvp
def hvp(grad_f, primals, tangents):
    return jvp(grad_f, primals, tangents)[1]
def clip(x):
    return jnp.clip(x,-1,1)
class DOALAgent(flax.struct.PyTreeNode):
    """Implicit Q-learning (IQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    delta = 1.0
    def get_guided_action(self,q_action, action,observation,alpha,delta,params):
        if "solver" not in self.config or self.config["solver"] == "linear":
            return self.get_linear_action(q_action, action,observation,alpha,self.delta,params)
        elif self.config["solver"] == "diag_hess":
            return self.get_diag_hess_action(q_action, action,observation,alpha,self.delta,params)
        elif self.config["solver"] == "full":
            return self.get_cg_action(q_action, action,observation,alpha,self.delta,params)
        elif self.config["solver"] == "bfgs":
            return self.get_bfgs_action(q_action, action,observation,alpha,self.delta,params)

    @jax.jit
    def get_bfgs_action(self, q_action, action, observation, alpha, delta, params):


        @jax.jit
        @partial(jax.vmap, in_axes=(0, 0, 0, None, None, None))
        def _get_guided_action(q_action, action, observation, alpha, delta, params):

            # --- HYPERPARAMETERS FOR MANUAL BFGS ---
            step_size = self.config["step_size"]
            num_steps = self.config["num_steps"]
            
            
            @jax.value_and_grad
            def q_objective(optim_action, initial_action, obs, net_params):
                qs = self.network.select('critic')(obs, optim_action, params=net_params)
                q = jnp.mean(qs)
                regularization = alpha * jnp.sum((optim_action - initial_action)**2)
                return -q + regularization
            
            def projected_step(H_inv,grad_action):
                dz = -step_size * (H_inv @ grad_action)
                unconstrained_action = q_action + dz
                
                distance = jnp.linalg.norm(dz)
                projected_action = jnp.where(
                    distance > delta,
                    q_action + dz * (delta / distance),
                    unconstrained_action
                )
                return jnp.clip(projected_action, -1.0, 1.0)

            def bfgs_step(H, _):
                H_inv = jnp.linalg.inv(H + 1e-6 * jnp.eye(H.shape[0]))

                final_step_action = projected_step(H_inv,grad_action)
                
                value, grad_new_action = q_objective(final_step_action, action, observation, params)
                dg = grad_new_action - grad_action
                actual_dz = final_step_action - q_action
                epsilon = 1e-8
                
                term1_num = jnp.outer(dg, dg)
                term1_den = jnp.dot(dg, actual_dz) + epsilon
                H_dz = H @ actual_dz
                term2_num = jnp.outer(H_dz, H_dz)
                term2_den = jnp.dot(actual_dz, H_dz) + epsilon
                H_new = H + (term1_num / term1_den) - (term2_num / term2_den)
                
                return H_new, value

            action_dim = q_action.shape[0]
            H =  2 * alpha * jnp.eye(action_dim)
            value, grad_action = q_objective(q_action, action, observation, params)
            
            # The scan now returns the final action and the final Hessian approximation H
            H, values_over_time = jax.lax.scan(
                f=bfgs_step,
                init=H,
                xs=None,
                length=num_steps
            )
            H_inv = jnp.linalg.inv(H + 1e-6 * jnp.eye(H.shape[0]))
            final_action = projected_step(H_inv,grad_action)
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
    def get_bfgs_action1(self, q_action, action, observation, alpha, delta, params):


        @jax.jit
        @partial(jax.vmap, in_axes=(0, 0, 0, None, None, None))
        def _get_guided_action(q_action, action, observation, alpha, delta, params):

            # --- HYPERPARAMETERS FOR MANUAL BFGS ---
            step_size = self.config["step_size"]
            num_steps = self.config["num_steps"]
            
            
            @jax.value_and_grad
            def q_objective(optim_action, initial_action, obs, net_params):
                qs = self.network.select('critic')(obs, optim_action, params=net_params)
                q = jnp.mean(qs)
                regularization = alpha * jnp.sum((optim_action - initial_action)**2)
                return -q + regularization

            def bfgs_step(carry, _):
                current_action, H = carry
                value, grad_action = q_objective(current_action, q_action, observation, params)
                H_inv = jnp.linalg.inv(H + 1e-6 * jnp.eye(H.shape[0]))
                dz = -step_size * (H_inv @ grad_action)
                unconstrained_action = current_action + dz
                
                diff = unconstrained_action - q_action
                distance = jnp.linalg.norm(diff)
                projected_action = jnp.where(
                    distance > delta,
                    q_action + diff * (delta / distance),
                    unconstrained_action
                )
                final_step_action = jnp.clip(projected_action, -1.0, 1.0)
                
                _, grad_new_action = q_objective(final_step_action, q_action, observation, params)
                dg = grad_new_action - grad_action
                actual_dz = final_step_action - current_action
                epsilon = 1e-8
                
                term1_num = jnp.outer(dg, dg)
                term1_den = jnp.dot(dg, actual_dz) + epsilon
                H_dz = H @ actual_dz
                term2_num = jnp.outer(H_dz, H_dz)
                term2_den = jnp.dot(actual_dz, H_dz) + epsilon
                H_new = H + (term1_num / term1_den) - (term2_num / term2_den)
                
                new_carry = (final_step_action, H_new)
                return new_carry, value

            action_dim = q_action.shape[0]
            initial_carry = (q_action, alpha * jnp.eye(action_dim))
            
            # The scan now returns the final action and the final Hessian approximation H
            (final_action, final_H), values_over_time = jax.lax.scan(
                f=bfgs_step,
                init=initial_carry,
                xs=None,
                length=num_steps
            )
            
            adjusted_actions = jax.lax.stop_gradient(final_action)
            
            # --- KEY CHANGE: Calculate Eigenvalues ---
            # We compute the eigenvalues of the final Hessian approximation.
            # `eigvalsh` is used because the Hessian is a symmetric matrix.
            hessian_eigenvalues = jnp.diagonal(final_H)
            
            # Final calculations for other return values
            final_objective_val, final_grad_unprocessed = q_objective(adjusted_actions, q_action, observation, params)
            final_reg = alpha * jnp.sum((adjusted_actions - q_action)**2)
            q_final = -(final_objective_val - final_reg)
            dx = jax.lax.stop_gradient(adjusted_actions - action)
            q = jax.lax.stop_gradient(q_final)
            g = jax.lax.stop_gradient(-final_grad_unprocessed)

            # --- KEY CHANGE: Return Eigenvalues instead of 0*q ---
            return adjusted_actions, dx, jax.lax.stop_gradient(hessian_eigenvalues), g, q

        return _get_guided_action(q_action, action, observation, alpha, delta, params)

    @jax.jit
    def get_diag_hess_action(self,q_action, action,observation,alpha,delta,params):


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
         #   dx = linear_solve.solve_normal_cg(A,b)
            
            adjusted_actions = jax.lax.stop_gradient(clip(q_action + dx))
            dx = jax.lax.stop_gradient(adjusted_actions - action)
            q =  jax.lax.stop_gradient(q)
            return  adjusted_actions, dx, h_diagonal, g, q

        return _get_guided_action(q_action, action,observation,alpha,params)
    @jax.jit
    def get_linear_action(self,q_action, action,observation,alpha,delta,params):


        @jax.jit
        @partial(jax.vmap, in_axes=(0,0,0,None,None))
        def _get_guided_action(q_action, action,observation,alpha,params):

            def bc_loss_wrt_q_action(q_action):
                qs = self.network.select('critic')(observation, q_action, params=params)
                q = jnp.mean(qs)
                return q 

            v_grad_q = jax.value_and_grad(bc_loss_wrt_q_action) 
            q, g = v_grad_q(q_action)
            b =  g / (2 * alpha )

            normb = jnp.linalg.norm(b)
            dx = jnp.where(normb > delta,  b / normb,   b)
         #   dx = linear_solve.solve_normal_cg(A,b)
            
            adjusted_actions = jax.lax.stop_gradient(clip(q_action + dx))
            dx = jax.lax.stop_gradient(adjusted_actions - action)
            q =  jax.lax.stop_gradient(q)
            return  adjusted_actions, dx, 0*q,g, q
        return _get_guided_action(q_action, action,observation,alpha,params)


# Assume self.network is defined elsewhere

    @jax.jit
    def get_newton_action(self, q_action, action, observation, alpha, delta, params):

        @jax.jit
        @partial(jax.vmap, in_axes=(0, 0, 0, None, None))
        def _get_guided_action(q_action, action, observation, alpha, params):

            # 1. Define the regularized objective function to be MINIMIZED.

            def bc_loss_wrt_q_action(q_action):
                qs = self.network.select('critic')(observation, q_action, params=params)
                q = jnp.mean(qs)
                return - q + alpha * jnp.sum((q_action - action)**2)



            grad_z = jax.grad(bc_loss_wrt_q_action)(q_action)
            hessian_z = jax.hessian(bc_loss_wrt_q_action)(q_action)
            new_z = z - 0.8 * (jnp.linalg.inv(hessian_z + jnp.eye(2) * 1) @ grad_z)

            # 4. Extract the results.
            adjusted_actions = jax.lax.stop_gradient(clip(results.params))
            final_objective_val = results.state.value

            # To get the pure Q-value, we can re-evaluate the critic or subtract the
            # regularization term from the final objective value.
            final_reg = alpha * jnp.sum((adjusted_actions - q_action)**2)
            q_final = -(final_objective_val - final_reg)
            
            dx = jax.lax.stop_gradient(adjusted_actions - action)
            q = jax.lax.stop_gradient(q_final)
            
            final_grad = jax.grad(q_maximization_objective_regularized)(adjusted_actions, q_action, observation, params)
            g = jax.lax.stop_gradient(-final_grad)

            return adjusted_actions, dx, 0*q, g, q

        return _get_guided_action(q_action, action, observation, alpha, params)

    @jax.jit
    def get_gd_action(self, q_action, action, observation, alpha, delta, params):

        @jax.jit
        @partial(jax.vmap, in_axes=(0, 0, 0, None, None))
        def _get_guided_action(q_action, action, observation, alpha, params):

            # --- HYPERPARAMETERS FOR MANUAL GRADIENT DESCENT ---
            # These are the knobs you will need to tune for performance.
            step_size = 1  # The learning rate for each gradient step.
            num_steps = 10    # The number of optimization steps to perform.
            
            # 1. Define the objective function to be MINIMIZED (same as before).
            #    We use value_and_grad for efficiency.
            @jax.value_and_grad
            def q_objective(optim_action, initial_action, obs, net_params):
                qs = self.network.select('critic')(obs, optim_action, params=net_params)
                q = jnp.mean(qs)
                regularization = alpha * jnp.sum((optim_action - initial_action)**2)
                return -q + regularization

            # 2. Define the function for a single step of gradient descent.
            #    This is the core function that `lax.scan` will loop over.
            def gradient_descent_step(current_action, _):
                # --- 1. Standard Gradient Step ---
                value, grad = q_objective(current_action, q_action, observation, params)
                # Take a step, creating a potentially unconstrained action
                unconstrained_action = current_action - step_size * grad
                
                # --- 2a. L2 Ball Projection ---
                # Project the action to be within an L2 distance of `delta` from the ORIGINAL action.
                # This acts like a leash, keeping the action from straying too far.
                diff = unconstrained_action - q_action
                distance = jnp.linalg.norm(diff)
                
                # If the distance is > delta, scale the difference vector back to length delta.
                # Otherwise, leave the action as is. `jnp.where` is JIT-friendly.
                projected_action = jnp.where(
                    distance > delta,
                    q_action + diff * (delta / distance),
                    unconstrained_action
                )

                # --- 2b. Clipping ---
                # Enforce the absolute range limits on the action.
                final_step_action = jnp.clip(projected_action, -1.0, 1.0)
                
                return final_step_action ,value

            # 3. Run the scan loop.
            #    - `gradient_descent_step`: The function to execute.
            #    - `init=q_action`: The starting point for the optimization.
            #    - `xs=None, length=num_steps`: Tells scan to run the function `num_steps` times.
            final_action, values_over_time = jax.lax.scan(
                f=gradient_descent_step,
                init=q_action,
                xs=None,
                length=num_steps
            )
            
            # 4. Extract and process the final results.
            adjusted_actions = jax.lax.stop_gradient(final_action)

            # Calculate the final Q-value at the optimized action.
            final_objective_val, final_grad_unprocessed = q_objective(adjusted_actions, q_action, observation, params)
            final_reg = alpha * jnp.sum((adjusted_actions - q_action)**2)
            q_final = -(final_objective_val - final_reg)

            dx = jax.lax.stop_gradient(adjusted_actions - action)
            q = jax.lax.stop_gradient(q_final)
            g = jax.lax.stop_gradient(-final_grad_unprocessed)

            return adjusted_actions, dx, 0*q, g, q

        return _get_guided_action(q_action, action, observation, alpha, params)
    @jax.jit
    def get_cg_action(self, q_action, action, observation, alpha, delta, params):

        @jax.jit
        @partial(jax.vmap, in_axes=(0, 0, 0, None, None))
        def _get_guided_action(q_action, action, observation, alpha, params):

            # 1. Define the regularized objective function to be MINIMIZED.
            def q_maximization_objective_regularized(optim_action, initial_action, obs, net_params):
                """Returns the negative Q-value plus a regularization penalty."""
                qs = self.network.select('critic')(obs, optim_action, params=net_params)
                q = jnp.mean(qs)

                # Penalty for deviating from the initial action. This keeps the
                # optimized action within a "trust region" of the start.
                regularization = alpha * jnp.sum((optim_action - initial_action)**2)

                # We minimize (-Q + regularization)
                return -q + regularization

            # 2. Instantiate the L-BFGS solver.
            solver = jaxopt.NonlinearCG(fun=q_maximization_objective_regularized, maxiter=1, tol=1e-3)


            # 3. Run the optimization.
            #    Note: We now pass `initial_action=q_action` as a static argument
            #    to be used in the regularization term.
            results = solver.run(init_params=q_action,
                                initial_action=q_action,
                                obs=observation,
                                net_params=params)

            # 4. Extract the results.
            adjusted_actions = jax.lax.stop_gradient(clip(results.params))
            final_objective_val = results.state.value

            # To get the pure Q-value, we can re-evaluate the critic or subtract the
            # regularization term from the final objective value.
            final_reg = alpha * jnp.sum((adjusted_actions - q_action)**2)
            q_final = -(final_objective_val - final_reg)
            
            dx = jax.lax.stop_gradient(adjusted_actions - action)
            q = jax.lax.stop_gradient(q_final)
            
            final_grad = jax.grad(q_maximization_objective_regularized)(adjusted_actions, q_action, observation, params)
            g = jax.lax.stop_gradient(-final_grad)

            return adjusted_actions, dx, 0*q, g, q

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
