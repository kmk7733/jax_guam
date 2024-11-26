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

ctx = jax.default_device(jax.devices("cpu")[0])
ctx.__enter__()

file_path = "/home/mk/research/guam/jax_guam/"
traj = {"hover_to_cruise", "hover_to_rectangle", "sinusoidal"}
method = {"difftune", "autotune"}
tuned_param = {"alloc", "lqr", "alloc_lqr"}

def computeCost(t):
    return jnp.exp(t)

def acceptance(new_cost, prev_cost):
    if new_cost < prev_cost:
        return True
    else:
        accept = np.random.uniform(0, 1)
        print(prev_cost/new_cost)
        return (accept < (prev_cost/new_cost))

def main(my_param):
    jax_use_cpu()
    # jax_use_double()
    set_logger_format()

    final_time = 50
        
    logger.info("Constructing GUAM...")
    guam = FuncGUAM()
    logger.info("Calling GUAM...")

    # Start profiling
    # jax.profiler.start_trace('profile_output')
    start = time.time()
    batch_size = 1
    state = GuamState.create()
    b_state: GuamState = jtu.tree_map(lambda x: np.broadcast_to(x, (batch_size,) + x.shape).copy(), state)

    T = int(final_time / guam.dt)

    vmap_step = jax.vmap(remat(jax.jit(ft.partial(guam.new_step, guam.dt))), in_axes=(0, None, None))

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
    bT_state, rmse = simulate_batch(b_state, my_param)

    # Plot x, y, z for real and desired positions
    print(rmse)
    bT_positions = bT_state.aircraft[:, :, 6:9]  # Extract real positions from state (assuming aircraft positions are in [6:9])
    pos_des_np = np.array(pos_des)  # Convert desired positions to numpy array

    return bT_positions, pos_des_np

def print_best_param(traj_type, tuned_param, method_to_use):
    import os
    p = tuned_param
    m = method_to_use
    np.set_printoptions(suppress=True)
    if method_to_use == "autotune":
        rmse_history = np.zeros((10,10))
        for i in range(10):
            current_path = os.path.join(file_path,m,traj_type,p,"rmse_history_"+str(i)+".npz")
            rmse_history[i] = np.load(current_path)['arr_0']
        rmse_best_idx = np.unravel_index(np.argmin(rmse_history, axis=None), rmse_history.shape)[0]
        # print(rmse_best_idx)
        current_path = os.path.join(file_path,m,traj_type,p,"Best_param_"+str(rmse_best_idx)+".npz")
        param = np.load(current_path,allow_pickle=True)['arr_0'].item() 
    else:
        current_path = os.path.join(file_path,m,traj_type,p,"Best_param.npz")
        param = np.load(current_path,allow_pickle=True)['arr_0'].item() 
    return param

if __name__ == "__main__":

    original_param = {
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

    traj_type = "hover_to_rectangle"

    difftuned = print_best_param(traj_type, "alloc_lqr", "difftune")
    autotuned = print_best_param(traj_type, "alloc_lqr", "autotune")

    with ipdb.launch_ipdb_on_exception():
        bT_positions, pos_des_np = main(original_param)
        bT_positions_dt, _ =main(difftuned)
        bT_positions_at, _ =main(autotuned)

    plt.figure(figsize=(12, 6))
    labels = ['x', 'y', 'z']
    time_horizon = np.arange(0, len(bT_positions[0, :, 0]))*0.01
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(time_horizon, bT_positions[0, :, i], label=f'Before Tune')
        plt.plot(time_horizon,bT_positions_dt[0, :, i], label=f'DiffTune')
        plt.plot(time_horizon, bT_positions_at[0, :, i], label=f'AutoTune ')
        plt.plot(time_horizon, pos_des_np[:, i], label=f'Desired', linestyle='--')
        plt.legend(loc="upper right")
        plt.title(f'{labels[i]} component of trajectory')
        plt.xlabel('Time step')
        plt.ylabel(f'{labels[i]} position')
        # if i == 1:
        #     plt.ylim(-5, 10)
    
    plt.tight_layout()
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.set_facecolor('white')
    ax.plot3D(bT_positions[0, :, 0],bT_positions[0, :, 1], -bT_positions[0, :, 2], label = 'Before Tune')
    ax.plot3D(bT_positions_dt[0, :, 0],bT_positions_dt[0, :, 1], -bT_positions_dt[0, :, 2], label = 'DiffTune')
    ax.plot3D(bT_positions_at[0, :, 0],bT_positions_at[0, :, 1], -bT_positions_at[0, :, 2], label = 'AutoTune')
    ax.plot3D(pos_des_np[:, 0],pos_des_np[:, 1], -pos_des_np[:, 2], label = "Desired")
    # ax.set_aspect('equal') # not relevent to the problem
    ax.view_init()
    ax.set_xlabel('North [ft]')
    ax.set_ylabel('East [ft]')
    ax.set_zlabel('Height [ft]')
    
    # ax.set_ylim(-100, 100)
    # plt.autoscale(enable=True)
    plt.legend()
    # plt.axis('square')
    plt.show()

    import pdb
    pdb.set_trace()