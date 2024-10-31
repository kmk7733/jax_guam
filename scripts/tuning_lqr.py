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



# jax.config.update("jax_enable_x64", True)
def main():
    # jax_use_cpu()
    # jax_use_double()
    set_logger_format()
    best_rmse = np.inf
    learning_rate = 1
    final_time = 45.0
    epoch_max = 1
    
    logger.info("Constructing GUAM...")
    guam = FuncGUAM()
    logger.info("Calling GUAM...")

    # my_param
    my_param = {
        # 'baseline_alloc': {
        #     # 'W_lon': jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1000.0, 10000000.0, 0.1]),
        #     # 'W_lat': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1000.0, 1000.0, 1.0])
        #     'W_lon': jnp.array([1.5358679e+00, 8.3843321e-01, 1.2812005e+00, 1.0000000e-01, 1.0000000e-01, 4.8056790e-01, 1.4986095e+00, 1.5526563e+00, 1.0000000e+00, 1.0000000e+03, 1.0000000e+07, 1.0168679e-01]),
        #     'W_lat': jnp.array([1.5600196e+00, 9.0476894e-01, 1.1099764e+00, 1.0000000e-01, 1.3520601e+00, 1.4072037e+00, 5.7398933e-01, 8.1650019e-01, 1.0000000e+03, 1.0000000e+03, 9.8281962e-01])
        # },
        'lqr': {
            'Q_lon': np.array([0.01, 0.01, 1000.0, 0.0, 0.0, 0.0]),
            'R_lon': np.array([1.0, 1.0, 1.0]),
            'Q_lat': np.array([0.01, 1000.0, 1000.0, 0.0, 0.0, 0.0]),
            'R_lat': np.array([1.0, 1.0, 1.0])
        },

    }
        
    rmse_history = []
    for i in range(epoch_max):
        # Start profiling
        # jax.profiler.start_trace('profile_output')
        batch_size = 1
        # batch_size = 8192
        # batch_size = 16_384
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

        # Wrapper to compute only RMSE for gradient computation
        def rmse_only(b_state0, my_param):
            _, rmse = simulate_batch(b_state0, my_param)  # We only need RMSE for the gradient
            return rmse

        # Compute both RMSE and the gradient with respect to my_param in one pass
        rmse_val, grad_rmse = jax.value_and_grad(rmse_only, argnums=1)(b_state, my_param)

        # Now you have both the RMSE value and the gradient with respect to my_param
        print(f"RMSE: {rmse_val}")
        print(f"Gradient of RMSE with respect to my_param: {grad_rmse}")

        # If you also need the state, you can reuse it without recomputation
        bT_state, _ = simulate_batch(b_state, my_param)

        # update my params
        my_param['baseline_alloc']['W_lon'] -= learning_rate * grad_rmse['baseline_alloc']['W_lon']
        my_param['baseline_alloc']['W_lat'] -= learning_rate * grad_rmse['baseline_alloc']['W_lat']
        my_param['baseline_alloc']['W_lon'] = jnp.clip(my_param['baseline_alloc']['W_lon'], 0.1)
        my_param['baseline_alloc']['W_lat'] = jnp.clip(my_param['baseline_alloc']['W_lat'], 0.1)
        rmse_history.append(rmse_val)

        # Stop profiling
        # jax.profiler.stop_trace() # tensorboard --logdir=profile_output
        print(f"my_param: {my_param}")
        
        jax.clear_caches()
        gc.collect()

        # Stop learning
        if i >0:
            if best_rmse > rmse_history[-1]:
                best_rmse = rmse_history[-1]
                print('BEST RMSE so far: ', best_rmse)
                best_param = my_param
            if abs(rmse_history[-1] - rmse_history[-2]) < 1e-3:
                rmse_history
                break
            if rmse_history[-1] - rmse_history[-2] > 1:
                break
    # # Store the state and RMSE from the first run
    # bT_state, rmse = simulate_batch(b_state, my_param)

    # # Compute the gradient of RMSE
    # grad_rmse = jax.grad(lambda b_state0, my_param: rmse)(b_state, my_param)

    np.savez("bT_state.npz", aircraft=bT_state.aircraft)
    np.savez("Pos_des.npz", pos_des)
    np.savez("Best_param.npz", best_param)

    # Plot the real and desired trajectories for x, y, z
    bT_positions = bT_state.aircraft[:, :, 6:9]  # Extract real positions from state (assuming aircraft positions are in [6:9])
    pos_des_np = np.array(pos_des)  # Convert desired positions to numpy array

    # Plot x, y, z for real and desired positions
    plt.figure(figsize=(12, 6))
    labels = ['x', 'y', 'z']
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(bT_positions[0, :, i], label=f'Real {labels[i]}')
        plt.plot(pos_des_np[:, i], label=f'Desired {labels[i]}', linestyle='--')
        plt.legend()
        plt.title(f'{labels[i]} component of trajectory')
        plt.xlabel('Time step')
        plt.ylabel(f'{labels[i]} position')

    plt.tight_layout()
    plt.show()

    # Plot loss history
    plt.plot(rmse_history)
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.show()

if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()