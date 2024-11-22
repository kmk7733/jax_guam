import functools as ft
import functools as ft

import gc
import ipdb
import jax
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import jax.numpy as jnp
import tqdm
from jax import remat
from jax_guam.functional.guam_new import FuncGUAM, GuamState
from jax_guam.subsystems.genctrl_inputs.genctrl_inputs import lift_cruise_reference_inputs
from jax_guam.utils.jax_utils import jax2np, jax_use_cpu, jax_use_double
from jax_guam.utils.logging import set_logger_format
from loguru import logger
import matplotlib.pyplot as plt
import time

import copy

import os

ctx = jax.default_device(jax.devices("cpu")[0])
ctx.__enter__()


sigma_params = {
    'baseline_alloc': {
        'W_lon': jnp.array([0.01]*12 ),
        'W_lat': jnp.array([0.01]*11)
    },
    'lqr': {
        'Q_lon': jnp.array([0.001, 0.001, 10]),
        'R_lon': jnp.array([0.1] * 3),
        'Q_lat': jnp.array([0.001, 10, 10]),
        'R_lat': jnp.array([0.1] * 3)
    }
}
def computeCost(t):
    return jnp.exp(t)

def acceptance(new_cost, prev_cost):
    if new_cost < prev_cost:
        return True
    else:
        accept = np.random.uniform(0, 1)
        print(prev_cost/new_cost)
        return (accept < (prev_cost/new_cost))

def sample_gaussian_at_step(params, pass_param, sigma = sigma_params, key = jax.random.PRNGKey(0)):
    """
    Sample from a Gaussian distribution centered at the parameter at each step.
    
    Args:
        params (dict): Dictionary of initial parameters.
        key (jax.random.PRNGKey): Random key for sampling.
        sigma (dict): Dictionary of standard deviations with the same shape as the parameters.

    Returns:
        dict: Sampled parameters.
    """
    sampled_params = {}
    
    for key_param, value in params.items():
        # print(key_param)
        # if isinstance(value, dict):
        sampled_params[key_param] = {}
        if key_param == pass_param:
            sampled_params[key_param] = value
        else:
            for sub_key, sub_value in value.items():
                sub_key_rng, key = jax.random.split(key)
                noise = jax.random.normal(sub_key_rng, shape=sub_value.shape) * sigma[key_param][sub_key]
                sampled_params[key_param][sub_key] = sub_value + noise

                if 'Q' in sub_key:
                    sampled_params[key_param][sub_key] = jnp.clip(sampled_params[key_param][sub_key] , 0.001)
                if 'R' in sub_key:
                    sampled_params[key_param][sub_key] = jnp.clip(sampled_params[key_param][sub_key] , 0.001)
                if 'W' in sub_key:
                    sampled_params[key_param][sub_key] = jnp.clip(sampled_params[key_param][sub_key] , 0.1)

    return sampled_params

def main(path, itr,pass_param):
    jax_use_cpu()
    # jax_use_double()
    set_logger_format()
    best_cost = np.inf
    prev_cost = np.inf
    new_cost = np.inf
    count_acceptance = 0
    final_time = 40

    epoch_max = 10
        
    my_param = {
        'baseline_alloc': {
            'W_lon': jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), #[omega 1-9 dele delf theta]
            'W_lat': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  #[omega 1-8 dela delr phi]
        },
        'lqr': {
            'Q_lon': np.array([0.01, 0.01, 1000.0]),
            'R_lon': np.array([1.0, 1.0, 1.0]),
            'Q_lat': np.array([0.01, 1000.0, 1000.0]),
            'R_lat': np.array([1.0, 1.0, 1.0])
        },
    }
    logger.info("Constructing GUAM...")
    guam = FuncGUAM()
    logger.info("Calling GUAM...")

    rmse_history = []
    for i in range(epoch_max):
        # Start profiling
        # jax.profiler.start_trace('profile_output')
        start = time.time()
        batch_size = 1
        state = GuamState.create()
        b_state: GuamState = jtu.tree_map(lambda x: np.broadcast_to(x, (batch_size,) + x.shape).copy(), state)

        T = int(final_time / guam.dt)

        vmap_step = jax.vmap(remat(jax.jit(ft.partial(guam.new_step, guam.dt))), in_axes=(0, None, None))
        # vmap_step = jax.vmap(jax.jit(ft.partial(guam.new_step, guam.dt)), in_axes=(0, None, None))

        global pos_des
        pos_des = []

        # @jax.checkpoint
        def simulate_batch(b_state0, my_param) -> GuamState:
            Tb_state = [b_state0]
            b_state = b_state0
            rmse=0
            global pos_des  # To store desired positions globally
            pos_des = []  # Reset pos_des for each simulation

            for kk in tqdm.trange(T):
                t = kk * guam.dt
                ref_inputs = lift_cruise_reference_inputs(t)
                b_state = vmap_step(b_state, ref_inputs, my_param)
                rmse += jnp.linalg.norm(ref_inputs.Pos_des - b_state.aircraft[0,6:9], ord = 2) ** 2
                # Tb_state.append(jax2np(b_state))
                Tb_state.append(b_state)
                pos_des.append(ref_inputs.Pos_des)
                
            bT_state = jtu.tree_map(lambda *args: jnp.stack(list(args), axis=1), *Tb_state)
            rmse = jnp.sqrt(rmse / T)
            return bT_state, rmse
        
        # Now you have both the RMSE value and the gradient with respect to my_param

        # If you also need the state, you can reuse it without recomputation
        _, rmse = simulate_batch(b_state, my_param)
        new_cost = computeCost(rmse)
        
        print(rmse)
        if i == 0 :
            prev_cost = new_cost
            prev_param= copy.deepcopy(my_param)
            rmse_history.append(rmse)
            # print(prev_cost)
        else:
            if acceptance(new_cost, prev_cost):
                prev_cost = new_cost
                prev_param = copy.deepcopy(my_param)
                count_acceptance += 1
                print("Acceptance rate: ", str(count_acceptance/epoch_max*100))
                rmse_history.append(rmse)
            else:
                print("Rejected configuration at iteration ", str(i))
                rmse_history.append(rmse_history[-1])

        if best_cost>prev_cost:
            best_cost=prev_cost
            best_param=prev_param

        if i != epoch_max-1:
            print(my_param)
            my_param = sample_gaussian_at_step(my_param, pass_param, key=jax.random.PRNGKey(itr))

    np.savez(path+"Last_param_"+str(itr)+".npz", my_param)
    np.savez(path+"Best_param_"+str(itr)+".npz", best_param)
    np.savez(path+"rmse_history_"+str(itr)+".npz", rmse_history)
    jax.clear_caches()
    gc.collect()
    
if __name__ == "__main__":

    case = 'alloc_lqr'
    trajectory = 'hover_to_cruise'
    path = "/home/mk/research/guam/jax_guam/autotune/"+trajectory+"/"+case+"/"

    if case == 'lqr':
        pass_param = 'baseline_alloc'
    if case == 'alloc_lqr':
        pass_param = None
    if case == 'alloc':
        pass_param = 'lqr'

    with ipdb.launch_ipdb_on_exception():
        for itr in range(10):
            if not os.path.isdir(path):
                os.makedirs(path)
            main(path, itr, pass_param)
