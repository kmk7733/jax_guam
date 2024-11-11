import jax.numpy as jnp
import jax.lax.linalg as lax_linalg
from jax import custom_jvp
from jax import lax
from jax.numpy.linalg import solve

from functools import partial
import jax
import ipdb
# from .eig_util import eig
ctx = jax.default_device(jax.devices("cpu")[0])
ctx.__enter__()

def lqr_continuous_time_infinite_horizon(A, B, Q, R):
  # Take the last dimension, in case we try to do some kind of broadcasting thing in the future.
  x_dim = A.shape[-1]

  # Enforce symmetricity
  # Q = (Q+Q.T)/2
  # R = (R+R.T)/2

  # See https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator#Infinite-horizon,_continuous-time_LQR.
  A1 = A 
  Q1 = Q 

  # See https://en.wikipedia.org/wiki/Algebraic_Riccati_equation#Solution.

  H = jnp.block([[A1, -B @ jnp.linalg.inv(R)@B.T], [-Q1, -A1.T]])
  # import ipdb; ipdb.set_trace()
  eigvals, eigvectors = eig(H)
  argsort = jnp.argsort(eigvals)
  ix = argsort[:x_dim]
  U = eigvectors[:, ix]
  P = U[x_dim:, :] @ jnp.linalg.inv(U[:x_dim, :])
 
  P = jnp.real(P)
  K = jnp.linalg.inv(R)@B.T @ P
  
  return K
  
@custom_jvp
def eig(a):
    w, vl, vr = lax_linalg.eig(a)
    return w, vr


@eig.defjvp
def eig_jvp_rule(primals, tangents):
    a, = primals
    da, = tangents

    w, v = eig(a)

    eye = jnp.eye(a.shape[-1], dtype=a.dtype)

    # carefully build reciprocal delta-eigenvalue matrix, avoiding NaNs.
    Fmat = (jnp.reciprocal(eye + w[..., jnp.newaxis, :] - w[..., jnp.newaxis])
            - eye)
    dot = partial(lax.dot if a.ndim == 2 else lax.batch_matmul,
                  precision=lax.Precision.HIGHEST)
    vinv_da_v = dot(solve(v, da), v)
    du = dot(v, jnp.multiply(Fmat, vinv_da_v))
    corrections = (jnp.conj(v) * du).sum(-2, keepdims=True)
    dv = du - v * corrections
    dw = jnp.diagonal(vinv_da_v, axis1=-2, axis2=-1)
    return (w, v), (dw, dv)