import jax
import numpy as np
import jax.numpy as jnp
from scipy.linalg import solve_continuous_are

def solve_care(A, B, Q, R):
    """
    Wrapper to solve the continuous-time Algebraic Riccati Equation (CARE).
    This uses SciPy's solver but converts results to JAX arrays.
    """
    A_np = np.asarray(A)
    B_np = np.asarray(B)
    Q_np = np.asarray(Q)
    R_np = np.asarray(R)
    P = solve_continuous_are(A_np, B_np, Q_np, R_np)
    return P

@jax.custom_vjp
def lqr_solution(A, B, Q, R):
    """
    Returns the LQR solution P (Riccati matrix) and sets up implicit differentiation.
    
    Args:
        A (jax.numpy.ndarray): State transition matrix.
        B (jax.numpy.ndarray): Control input matrix.
        Q (jax.numpy.ndarray): State cost matrix.
        R (jax.numpy.ndarray): Control cost matrix.
    
    Returns:
        P (jax.numpy.ndarray): Solution to the CARE, matrix P.
    """
    P = solve_care(A, B, Q, R)
    return P

def care_residual(P, A, B, Q, R):
    """CARE residual function, F(P; A, B, Q, R) = 0."""
    return A.T @ P + P @ A - P @ B @ jnp.linalg.inv(R) @ B.T @ P + Q

def lqr_solution_bwd(fwd_vars, out_grad):
    P, A, B, Q, R = fwd_vars  # Unpack saved values

    # Compute Jacobians of the residual function with respect to each argument
    dres_dp = jax.jacobian(care_residual, 0)(*fwd_vars)
    dres_da = jax.jacobian(care_residual, 1)(*fwd_vars)
    dres_db = jax.jacobian(care_residual, 2)(*fwd_vars)
    dres_dq = jax.jacobian(care_residual, 3)(*fwd_vars)
    dres_dr = jax.jacobian(care_residual, 4)(*fwd_vars)
    
    # Solve for the adjoint (Lagrange multiplier)
    adj = jnp.linalg.tensorsolve(dres_dp.T, out_grad.T)
    N = adj.ndim

    # Compute the gradients for A, B, Q, and R
    a_grad = -jnp.tensordot(dres_da.T, adj, N).T
    b_grad = -jnp.tensordot(dres_db.T, adj, N).T
    q_grad = -jnp.tensordot(dres_dq.T, adj, N).T
    q_grad = (q_grad + q_grad.T) / 2 
    r_grad = -jnp.tensordot(dres_dr.T, adj, N).T
    r_grad = (r_grad + r_grad.T) / 2 

    return (a_grad, b_grad, q_grad, r_grad)

def lqr_solution_fwd(A, B, Q, R):
    P = lqr_solution(A, B, Q, R)
    return P, (P, A, B, Q, R)

# Attach forward and backward passes for custom VJP
lqr_solution.defvjp(lqr_solution_fwd, lqr_solution_bwd)
