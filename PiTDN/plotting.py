import numpy as np
import matplotlib.pyplot as plt
from PiTDN import scaling
from collections import defaultdict
import matplotlib.gridspec as gridspec
from matplotlib import colormaps
import matplotlib.colors as mcolors
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PiTDN import loss_func

params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)


def plot_loss(HyperParams):
    """
    Plots the history of losses during the training of the autoencoder.

    Args:
        HyperParams (namedtuple): An object containing the parameters of the autoencoder.
    """

    history = np.load(HyperParams.net_dir+'history'+HyperParams.net_run+'.npy', allow_pickle=True).item()
    history_test = np.load(HyperParams.net_dir+'history_test'+HyperParams.net_run+'.npy', allow_pickle=True).item()
    ax = plt.figure(figsize=(12, 6)).gca()
    ax.semilogy(history['l1'])
    ax.semilogy(history['l2'])
    ax.semilogy(history['l3'])
    ax.semilogy(history['l4'])
    ax.semilogy(history_test['l1'], '--')
    ax.semilogy(history_test['l2'], '--')
    ax.semilogy(history_test['l3'], '--')
    ax.semilogy(history_test['l4'], '--')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title('Loss over training epochs')
    plt.legend(['Autoencoder (train)', 'Map (train)', 'TDN (train)', 'Sparse (train)', 'Autoencoder (test)', 'Map (test)', 'TDN (test)', 'Sparse (test)'])
    plt.savefig(HyperParams.net_dir+'history_losses'+HyperParams.net_run+'.png', bbox_inches='tight', dpi=500)


def plot_latent(HyperParams, latents, latents_estimation):
    """
    Plot the original and estimated latent spaces.

    Args:
        HyperParams (obj): Object containing the autoencoder parameters.
        latents (tensor): Tensor of original latent spaces.
        latents_estimation (tensor): Tensor of estimated latent spaces.
    """
    plt.figure(figsize=(12, 4))
    for i1 in range(HyperParams.bottleneck_dim):
        plt.plot(latents[:,i1].detach(), '--')
        plt.plot(latents_estimation[:,i1].detach(),'-')
    plt.title('Evolution in the latent space')
    plt.ylabel('$u_N(\mu)$')
    plt.xlabel('Snaphots')
    plt.legend(['Autoencoder', 'Map'])
    plt.savefig(HyperParams.net_dir+'latents'+HyperParams.net_run+'.png', bbox_inches='tight', dpi=500)
    green_diamond = dict(markerfacecolor='g', marker='D')
    _, ax = plt.subplots(figsize=(12, 4))
    ax.boxplot(latents_estimation.detach().numpy(), flierprops=green_diamond)
    plt.title('Variance in the latent space')
    plt.ylabel('$u_N(\mu)$')
    plt.xlabel('Bottleneck')
    plt.savefig(HyperParams.net_dir+'box_plot_latents'+HyperParams.net_run+'.png', bbox_inches='tight', dpi=500)


def plot_error(res, VAR_all_reshaped, scaler_all_reshaped, HyperParams, mu_space, params, train_trajectories, vars, p1=0, p2=-1):
    """
    Plots the relative error between the predicted and actual results as a 3D surface.

    Args:
        res (ndarray): The predicted results.
        VAR_all_reshaped (ndarray): The actual results (reshaped).
        scaler_all_reshaped (object): The scaler object used for scaling the results.
        HyperParams (object): The HyperParams object holding the necessary hyperparameters.
        mu_space (list): List of parameter space ranges.
        params (ndarray): The input variables.
        train_trajectories (ndarray): The indices of the training data.
        vars (str): The name of the variable being plotted.
        p1 (int): Index of the first parameter dimension.
        p2 (int): Index of the second parameter dimension.
    """
    res = res.reshape(-1, res.shape[2], 1)
    u_hf = scaling.inverse_scaling(VAR_all_reshaped, scaler_all_reshaped, HyperParams.scaling_type)
    u_app = scaling.inverse_scaling(res, scaler_all_reshaped, HyperParams.scaling_type)
    error = np.linalg.norm(u_app - u_hf, axis=0) / np.linalg.norm(u_hf, axis=0)
    mu1_range = mu_space[p1]
    mu2_range = mu_space[p2]
    n_params = params.shape[1]
    tr_pt_1 = params[train_trajectories, p1]
    tr_pt_2 = params[train_trajectories, p2]
    if n_params > 2:
        rows, ind = np.unique(params[:, [p1, p2]], axis=0, return_inverse=True)
        indices_dict = defaultdict(list)
        [indices_dict[tuple(rows[i])].append(idx) for idx, i in enumerate(ind)]
        error = np.array([np.mean(error[indices]) for indices in indices_dict.values()])
        tr_pt = [i for i in indices_dict if any(idx in train_trajectories for idx in indices_dict[i])]
        tr_pt_1 = [t[0] for t in tr_pt]
        tr_pt_2 = [t[1] for t in tr_pt]
    X1, X2 = np.meshgrid(mu1_range, mu2_range, indexing='ij')
    output = np.reshape(error, (len(mu1_range), len(mu2_range)))
    fig = plt.figure('Relative Error '+vars)
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X1, X2, output, cmap=colormaps['jet'], color='blue')
    ax.contour(X1, X2, output, zdir='z', offset=output.min(), cmap=colormaps['jet'])
    ax.set(xlabel=f'$\mu_{str((p1%n_params)+1)}$',
           ylabel=f'$\mu_{str((p2%n_params)+1)}$',
           zlabel='$\\epsilon_{PiTDN}(\\mathbf{\mu})$')
    ax.plot(tr_pt_1, tr_pt_2, output.min()*np.ones(len(tr_pt_1)), '*r')
    ax.set_title('Relative Error '+vars)
    ax.zaxis.offsetText.set_visible(False)
    exponent_axis = np.floor(np.log10(max(ax.get_zticks()))).astype(int)
    ax.ticklabel_format(axis='z', style='sci', scilimits=(0, 0))
    ax.text2D(0.9, 0.82, "$\\times 10^{"+str(exponent_axis)+"}$", transform=ax.transAxes, fontsize="x-large")
    plt.subplots_adjust(right=0.8)
    plt.tight_layout()
    plt.savefig(HyperParams.net_dir+'relative_error_'+vars+HyperParams.net_run+'.png', transparent=True, dpi=500)


def plot_error_2d(res, VAR_all_reshaped, scaler_all_reshaped, HyperParams, mu_space, params, train_trajectories, var_name, p1=0, p2=-1):
    """
    Plots the relative error between the predicted and actual results in 2D.

    Args:
        res (ndarray): The predicted results.
        VAR_all_reshaped (ndarray): The actual results (reshaped).
        scaler_all_reshaped (object): The scaler object used for scaling the results.
        HyperParams (object): The HyperParams object holding the necessary hyperparameters.
        mu_space (list): List of parameter space ranges.
        params (ndarray): The input variables.
        train_trajectories (ndarray): The indices of the training data.
        var_name (str): The name of the variable being plotted.
        p1 (int): Index of the first parameter dimension.
        p2 (int): Index of the second parameter dimension.
    """
    res = res.reshape(-1, res.shape[2], 1)
    u_hf = scaling.inverse_scaling(VAR_all_reshaped, scaler_all_reshaped, HyperParams.scaling_type)
    u_app = scaling.inverse_scaling(res, scaler_all_reshaped, HyperParams.scaling_type)
    error = np.linalg.norm(u_app - u_hf, axis=0) / np.linalg.norm(u_hf, axis=0)
    # TODO: average error values at the same mu point for plotting
    mu1_range = mu_space[p1]
    mu2_range = mu_space[p2]
    n_params = params.shape[1]
    tr_pt_1 = params[train_trajectories, p1]
    tr_pt_2 = params[train_trajectories, p2]
    if n_params > 2:
        rows, ind = np.unique(params[:, [p1, p2]], axis=0, return_inverse=True)
        indices_dict = defaultdict(list)
        [indices_dict[tuple(rows[i])].append(idx) for idx, i in enumerate(ind)]
        error = np.array([np.mean(error[indices]) for indices in indices_dict.values()])
        tr_pt = [i for i in indices_dict if any(idx in train_trajectories for idx in indices_dict[i])]
        tr_pt_1 = [t[0] for t in tr_pt]
        tr_pt_2 = [t[1] for t in tr_pt]
    X1, X2 = np.meshgrid(mu1_range, mu2_range, indexing='ij')
    output = np.reshape(error, (len(mu1_range), len(mu2_range)))
    fig = plt.figure('Relative Error 2D '+var_name)
    ax = fig.add_subplot()
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    colors = output.flatten()
    sc = plt.scatter(X1.flatten(), X2.flatten(), s=(2e1*colors/output.max())**2, c=colors, cmap=colormaps['jet'])
    plt.colorbar(sc, format=fmt)
    ax.set(xlabel=f'$\mu_{str((p1%n_params)+1)}$',
           ylabel=f'$\mu_{str((p2%n_params)+1)}$')
    ax.plot(tr_pt_1, tr_pt_2, '*r')
    ax.set_title('Relative Error 2D '+var_name)
    plt.tight_layout()
    plt.savefig(HyperParams.net_dir+'relative_error_2d_'+var_name+HyperParams.net_run+'.png', transparent=True, dpi=500)


def plot_fields(index, unstruct_results, scaler_all_reshaped, HyperParams, dataset, xyz, params, cand_list, comp="_U"):
    """
    Plots the predicted field solution for a given snapshot.

    Args:
        index (int): Snapshot index to plot.
        unstruct_results (ndarray): Network output (predicted solution).
        scaler_all_reshaped (ndarray): Scaler object for inverse scaling.
        HyperParams (object): Hyperparameters object containing network architecture and training info.
        dataset (object): Dataset object containing mesh/triangulation information.
        xyz (list): List of coordinate arrays [x, y] or [x, y, z].
        params (ndarray): Parameter array associated with each snapshot.
        cand_list (list): List of candidate indices.
        comp (str): Component suffix for file naming.
    """
    res = unstruct_results
    res = res.reshape(-1, res.shape[2], 1)
    res = res.squeeze().T
    fig = plt.figure()
    Z_net = scaling.inverse_scaling(res, scaler_all_reshaped, HyperParams.scaling_type)
    z_net = Z_net[:, index]
    cand_list = np.array(cand_list)
    flat_cand_list = cand_list.flatten()
    SNAP = flat_cand_list[index]
    xx = xyz[0].squeeze()
    yy = xyz[1].squeeze()
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    if dataset.dim == 2:
        triang = np.asarray(dataset.T - 1)
        gs1 = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs1[0, 0])
        cs = ax.tricontourf(xx, yy, triang, z_net, 100, cmap=colormaps['jet'])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(cs, cax=cax, format=fmt)
    elif dataset.dim == 3:
        zz = xyz[2]
        ax = fig.add_subplot(projection='3d')
        cax = inset_axes(ax, width="5%", height="60%", loc="center left",
                         bbox_to_anchor=(1.15, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        p = ax.scatter(xx, yy, zz, c=z_net, cmap=colormaps['jet'], linewidth=0.5)
        cbar = fig.colorbar(p, ax=ax, cax=cax, format=fmt)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        ax.locator_params(axis='both', nbins=5)
    tick_locator = MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.ax.yaxis.set_offset_position('left')
    cbar.update_ticks()
    plt.tight_layout()
    ax.set_aspect('equal', 'box')
    ax.set_title('Solution field for $\mu$ = '+str(np.around(params[SNAP].detach().numpy(), 2)))
    plt.savefig(HyperParams.net_dir+'field_solution_'+str(SNAP)+''+HyperParams.net_run+comp+'.png', bbox_inches='tight', dpi=500)


def plot_truth(index, VAR_all_reshaped, scaler_all_reshaped, HyperParams, dataset, xyz, params, cand_list, comp="_U"):
    """
    Plots the ground truth solution for a given snapshot.

    Args:
        index (int): Snapshot index to plot.
        VAR_all_reshaped (ndarray): Ground truth solution (reshaped).
        scaler_all_reshaped (ndarray): Scaler object for inverse scaling.
        HyperParams (object): Hyperparameters object containing network architecture and training info.
        dataset (object): Dataset object containing mesh/triangulation information.
        xyz (list): List of coordinate arrays [x, y] or [x, y, z].
        params (ndarray): Parameter array associated with each snapshot.
        cand_list (list): List of candidate indices.
        comp (str): Component suffix for file naming.
    """

    fig = plt.figure()
    Z_net = scaling.inverse_scaling(VAR_all_reshaped, scaler_all_reshaped, HyperParams.scaling_type)
    z_net = Z_net[:, index]
    cand_list = np.array(cand_list)
    flat_cand_list = cand_list.flatten()
    SNAP = flat_cand_list[index]

    xx = xyz[0].squeeze()
    yy = xyz[1].squeeze()
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    if dataset.dim == 2:
        triang = np.asarray(dataset.T - 1)
        gs1 = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs1[0, 0])
        cs = ax.tricontourf(xx, yy, triang, z_net, 100, cmap=colormaps['jet'])
        ax.triplot(xx, yy, triang, lw=0.5, color="black")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(cs, cax=cax, format=fmt)
    elif dataset.dim == 3:
        zz = xyz[2]
        ax = fig.add_subplot(projection='3d')
        cax = inset_axes(ax, width="5%", height="60%", loc="center left",
                         bbox_to_anchor=(1.15, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        p = ax.scatter(xx, yy, zz, c=z_net, cmap=colormaps['jet'], linewidth=0.5)
        cbar = fig.colorbar(p, ax=ax, cax=cax, format=fmt)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        ax.locator_params(axis='both', nbins=5)
    tick_locator = MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.ax.yaxis.set_offset_position('left')
    cbar.update_ticks()
    plt.tight_layout()
    ax.set_aspect('equal', 'box')
    ax.set_title('HF Solution field for $\mu$ = '+str(np.around(params[SNAP].detach().numpy(), 2)))
    plt.savefig(HyperParams.net_dir+'hf_field_solution_'+str(SNAP)+''+HyperParams.net_run+comp+'.png', bbox_inches='tight', dpi=500)


def plot_error_fields(index, unstruct_results, VAR_all_reshaped, scaler_all_reshaped, HyperParams, dataset, xyz, params, cand_list, comp="_U"):
    """
    Plots a contour map of the error field for a given solution of a scalar field.
    The error is computed as the absolute difference between the true solution and the predicted solution,
    normalized by the 2-norm of the true solution.

    Args:
        index (int): Snapshot index of the solution to be plotted.
        unstruct_results (ndarray): Predicted solution.
        VAR_all_reshaped (ndarray): True solution (reshaped).
        scaler_all_reshaped (ndarray): Scaler information used in the prediction.
        HyperParams (object): Model architecture and training parameters.
        dataset (object): Dataset object containing mesh/triangulation information.
        xyz (list): List of coordinate arrays [x, y] or [x, y, z].
        params (ndarray): Model parameters associated with each snapshot.
        cand_list (list): List of candidate indices.
        comp (str): Component suffix for file naming.
    """
    res = unstruct_results
    res = res.reshape(-1, res.shape[2], 1)
    res = res.squeeze().T
    VAR_all_reshaped = VAR_all_reshaped.squeeze().T
    VAR_all_reshaped = VAR_all_reshaped.reshape(VAR_all_reshaped.shape[0], VAR_all_reshaped.shape[1]*VAR_all_reshaped.shape[2])
    Z = scaling.inverse_scaling(VAR_all_reshaped, scaler_all_reshaped, HyperParams.scaling_type)
    Z_net = scaling.inverse_scaling(res, scaler_all_reshaped, HyperParams.scaling_type)
    fig = plt.figure()
    z = Z[:, index]
    z_net = Z_net[:, index]
    cand_list = np.array(cand_list)
    flat_cand_list = cand_list.flatten()
    SNAP = flat_cand_list[index]
    error = abs(z - z_net)/np.linalg.norm(z, 2)
    xx = xyz[0].squeeze()
    yy = xyz[1].squeeze()
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    if dataset.dim == 2:
        triang = np.asarray(dataset.T - 1)
        gs1 = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs1[0, 0])
        cs = ax.tricontourf(xx, yy, triang, error, 100, cmap=colormaps['jet'])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(cs, cax=cax, format=fmt)
    elif dataset.dim == 3:
        zz = xyz[2]
        ax = fig.add_subplot(projection='3d')
        cax = inset_axes(ax, width="5%", height="60%", loc="center left",
                         bbox_to_anchor=(1.15, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        p = ax.scatter(xx, yy, zz, c=error, cmap=colormaps['jet'], linewidth=0.5)
        cbar = fig.colorbar(p, ax=ax, cax=cax, format=fmt)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        ax.locator_params(axis='both', nbins=5)
    tick_locator = MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.ax.yaxis.set_offset_position('left')
    cbar.update_ticks()
    plt.tight_layout()
    ax.set_aspect('equal', 'box')
    ax.set_title('Error field for $\mu$ = '+str(np.around(params[SNAP].detach().numpy(), 2)))
    plt.savefig(HyperParams.net_dir+'error_field_'+str(SNAP)+''+HyperParams.net_run+comp+'.png', bbox_inches='tight', dpi=500)


def plot_latent_time(HyperParams, SAMPLE, latents, mu_space, params, param_sample):
    """
    Plots the evolution of latent states over time and saves the plot as a .png file.

    Args:
        HyperParams (object): The hyperparameters object.
        SAMPLE (int): The sample index.
        latents (ndarray): The latent states tensor.
        mu_space (list): The parameter space (last element is the time axis).
        params (list): The parameter array.
        param_sample (int): Number of parameter samples.

    Returns:
        None
    """

    plt.figure()
    sequence_length = latents.shape[0] // param_sample
    start = SAMPLE * sequence_length
    end = start + sequence_length
    time = mu_space[-1]

    for i in range(HyperParams.bottleneck_dim):
        stn_evolution = latents[start:end, i]
        plt.plot(time, stn_evolution)

    plt.xlabel('$t$')
    plt.ylabel('$s(t)$')
    plt.title('Latent state evolution $\mu = $'+ str(np.around(params[start][0:2].detach().numpy(), 2)))
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(HyperParams.net_dir+'latent_evolution_'+HyperParams.net_run+str(SAMPLE)+'.png', bbox_inches='tight', dpi=500)


def plot_sample(HyperParams, mu_space, params, train_trajectories, test_trajectories, p1=0, p2=1, param_frequency=False):
    """
    Plots the train/test sample used for the training.

    Args:
        HyperParams (object): The HyperParams object holding the necessary hyperparameters.
        mu_space (list): List of parameter space ranges.
        params (ndarray): The input variables.
        train_trajectories (ndarray): The indices of the training data.
        test_trajectories (ndarray): The indices of the testing data.
        p1 (int): Index of the first parameter dimension.
        p2 (int): Index of the second parameter dimension.
        param_frequency (bool): If True, plot parameter frequency bar chart instead of scatter plot.
    """

    mu1_range = mu_space[p1]
    mu2_range = mu_space[p2]
    n_params = params.shape[1]
    tr_pt_1 = params[train_trajectories, p1]
    tr_pt_2 = params[train_trajectories, p2]
    te_pt_1 = params[test_trajectories, p1]
    te_pt_2 = params[test_trajectories, p2]

    fig = plt.figure('Sample')
    ax = fig.add_subplot()

    if param_frequency is True:
        for i in range(len(params[train_trajectories][0])):
            plot_idx=[]
            plot_val=[]
            vals, counts = np.unique(params[train_trajectories][:, i], return_counts=True)
            args = vals.argsort()
            vals = vals[args]
            counts = counts[args]
            for j in range(len(vals)):
                mu = vals[j]
                val = counts[j]
                plot_idx.append(f'$\mu_{i}={np.around(mu, 2)}$')
                plot_val.append(val)
            plt.bar(plot_idx, plot_val)
        plt.xticks(rotation=90)
        plt.xlabel('Parameter')
        plt.ylabel('Frequency in Training Set')
    else:
        if n_params > 2:
            rows, ind = np.unique(params[:, [p1, p2]], axis=0, return_inverse=True)
            indices_dict = defaultdict(list)
            [indices_dict[tuple(rows[i])].append(idx) for idx, i in enumerate(ind)]
            tr_pt = [i for i in indices_dict if any(idx in train_trajectories for idx in indices_dict[i])]
            te_pt = [i for i in indices_dict if any(idx in test_trajectories for idx in indices_dict[i])]
            tr_pt_1 = [t[0] for t in tr_pt]
            tr_pt_2 = [t[1] for t in tr_pt]
            te_pt_1 = [s[0] for s in te_pt]
            te_pt_2 = [s[1] for s in te_pt]
        ax.set(xlim=tuple([mu1_range[0], mu1_range[-1]]),
            ylim=tuple([mu2_range[0], mu2_range[-1]]),
            xlabel=f'$\mu_{str((p1%n_params)+1)}$',
            ylabel=f'$\mu_{str((p2%n_params)+1)}$')
        ax.scatter(tr_pt_1, tr_pt_2, marker='o', color="red", label='Training')
        ax.scatter(te_pt_1, te_pt_2, marker='s', color="blue", label='Testing')
        ax.legend()

    ax.set_title('Sample')
    plt.tight_layout()
    plt.savefig(HyperParams.net_dir+'sample'+HyperParams.net_run+'.png', transparent=True, dpi=500)


def plot_comparison_fields(unstruct_results, VAR_all_reshaped, scaler_all_reshaped, HyperParams, dataset, xyz, params, cand_list, grid="horizontal", comp="_U", adjust_title=None):
    """
    Plots a comparison of the predicted field solution, ground truth, and error field.

    Args:
        unstruct_results (ndarray): The predicted solution.
        VAR_all_reshaped (ndarray): The ground truth solution (reshaped).
        scaler_all_reshaped (ndarray): Scaler object for inverse scaling.
        HyperParams (object): Hyperparameters object containing network architecture and training info.
        dataset (object): Dataset object containing mesh/triangulation information.
        xyz (list): List of coordinate arrays [x, y] or [x, y, z].
        params (ndarray): Parameter array associated with each snapshot.
        cand_list (list): List of candidate indices.
        grid (str): Layout orientation ('horizontal' or 'vertical').
        comp (str): Component suffix for file naming.
        adjust_title (float, optional): Y-position adjustment for the suptitle.
    """
    res = unstruct_results
    res = res.reshape(-1, res.shape[2], 1)
    res = res.squeeze().T
    VAR_all_reshaped = VAR_all_reshaped.squeeze().T
    VAR_all_reshaped = VAR_all_reshaped.reshape(VAR_all_reshaped.shape[0], VAR_all_reshaped.shape[1]*VAR_all_reshaped.shape[2])
    cand_list = np.array(cand_list)
    flat_cand_list = cand_list.flatten()
    plt.figure()
    Z = scaling.inverse_scaling(VAR_all_reshaped, scaler_all_reshaped, HyperParams.scaling_type)
    Z_net = scaling.inverse_scaling(res, scaler_all_reshaped, HyperParams.scaling_type)
    error = np.linalg.norm(Z_net - Z, axis=0) / np.linalg.norm(Z, axis=0)
    index = np.argmax(error)
    z = Z[:, index]
    z_net = Z_net[:, index]
    SNAP = flat_cand_list[index]
    xx = xyz[0].squeeze()
    yy = xyz[1].squeeze()

    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    triang = np.asarray(dataset.T - 1)
    error_abs = abs(z - z_net)
    error_rel = error_abs/np.linalg.norm(z, 2)

    if dataset.dim == 2:
        if grid == "horizontal":
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            y0=0.7
        elif grid == "vertical":
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
            y0=1.1
    elif dataset.dim == 3:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, subplot_kw=dict(projection='3d'))
        y0=1.1
    if adjust_title is not None:
        y0 = adjust_title

    # Subplot 1
    if dataset.dim == 2:
        norm1 = mcolors.Normalize(vmin=z.min(), vmax=z.max())
        cs1 = ax1.tricontourf(xx, yy, triang, z, 100, cmap=colormaps['jet'], norm=norm1)
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        cbar1 = plt.colorbar(cs1, cax=cax1, format=fmt)
    elif dataset.dim == 3:
        zz = xyz[2]
        cax1 = inset_axes(ax1, width="5%", height="60%", loc="center left",
                         bbox_to_anchor=(1.5, 0., 1, 1), bbox_transform=ax1.transAxes, borderpad=0)
        p1 = ax1.scatter(xx, yy, zz, c=z, cmap=colormaps['jet'], linewidth=0.5)
        cbar1 = fig.colorbar(p1, ax=ax1, cax=cax1, format=fmt)
        ax1.locator_params(axis='x', nbins=2)
        ax1.yaxis.set_ticklabels([])
        ax1.zaxis.set_ticklabels([])
    tick_locator = MaxNLocator(nbins=3)
    cbar1.locator = tick_locator
    cbar1.ax.yaxis.set_offset_position('left')
    cbar1.update_ticks()
    ax1.set_aspect('equal', 'box')
    ax1.set_title('Truth')

    # Subplot 2
    if dataset.dim == 2:
        norm2 = mcolors.Normalize(vmin=z_net.min(), vmax=z_net.max())
        cs2 = ax2.tricontourf(xx, yy, triang, z_net, 100, cmap=colormaps['jet'], norm=norm2)
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        cbar2 = plt.colorbar(cs2, cax=cax2, format=fmt)
    elif dataset.dim == 3:
        zz = xyz[2]
        cax2 = inset_axes(ax2, width="5%", height="60%", loc="center left",
                         bbox_to_anchor=(1.5, 0., 1, 1), bbox_transform=ax2.transAxes, borderpad=0)
        p2 = ax2.scatter(xx, yy, zz, c=z_net, cmap=colormaps['jet'], linewidth=0.5)
        cbar2 = fig.colorbar(p2, ax=ax2, cax=cax2, format=fmt)
        ax2.locator_params(axis='x', nbins=2)
        ax2.yaxis.set_ticklabels([])
        ax2.zaxis.set_ticklabels([])
    tick_locator = MaxNLocator(nbins=3)
    cbar2.locator = tick_locator
    cbar2.ax.yaxis.set_offset_position('left')
    cbar2.update_ticks()
    ax2.set_aspect('equal', 'box')
    ax2.set_title('Prediction')

    # Subplot 3
    if dataset.dim == 2:
        norm3 = mcolors.Normalize(vmin=error_rel.min(), vmax=error_rel.max())
        cs3 = ax3.tricontourf(xx, yy, triang, error_rel, 100, cmap=colormaps['jet'], norm=norm3)
        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes("right", size="5%", pad=0.1)
        cbar3 = plt.colorbar(cs3, cax=cax3, format=fmt)
    elif dataset.dim == 3:
        zz = xyz[2]
        cax3 = inset_axes(ax3, width="5%", height="60%", loc="center left",
                         bbox_to_anchor=(1.5, 0., 1, 1), bbox_transform=ax3.transAxes, borderpad=0)
        p3 = ax3.scatter(xx, yy, zz, c=error_rel, cmap=colormaps['jet'], linewidth=0.5)
        cbar3 = fig.colorbar(p3, ax=ax3, cax=cax3, format=fmt)
        ax3.locator_params(axis='x', nbins=2)
        ax3.yaxis.set_ticklabels([])
        ax3.zaxis.set_ticklabels([])
    tick_locator = MaxNLocator(nbins=3)
    cbar3.locator = tick_locator
    cbar3.ax.yaxis.set_offset_position('left')
    cbar3.update_ticks()
    ax3.set_aspect('equal', 'box')
    ax3.set_title('Error')

    # Adjust layout
    plt.tight_layout()
    fig.suptitle('Maximum error for $\mu$ = '+str(np.around(params[SNAP].detach().numpy(), 2)), y=y0)
    plt.savefig(HyperParams.net_dir+'comparison_field_'+str(SNAP)+''+HyperParams.net_run+comp+'.png', bbox_inches='tight', dpi=500)


def plot_error_3d(reconstruction, solution, scaler, HyperParams, mu_space, params, train_trajectories, var_name, test_trajectories=None):
    """
    Plots the relative error between the predicted and actual results in 3D.

    Args:
        reconstruction (ndarray): The predicted results.
        solution (ndarray): The actual results.
        scaler (object): The scaler object used for scaling the results.
        HyperParams (object): The HyperParams object holding the necessary hyperparameters.
        mu_space (list): List of parameter space ranges.
        params (ndarray): The input variables.
        train_trajectories (ndarray): The indices of the training data.
        var_name (str): The name of the variable being plotted.
        test_trajectories (ndarray, optional): The indices of the test data.
    """

    u_hf = scaling.inverse_scaling(solution, scaler, HyperParams.scaling_type)
    u_app = scaling.inverse_scaling(reconstruction, scaler, HyperParams.scaling_type)
    error = np.linalg.norm(u_app - u_hf, axis=0) / np.linalg.norm(u_hf, axis=0)
    p1 = 0
    p2 = 1
    p3 = 2
    mu1_range = mu_space[p1]
    mu2_range = mu_space[p2]
    mu3_range = mu_space[p3]
    n_params = params.shape[0]
    tr_pt_1 = params[train_trajectories, p1]
    tr_pt_2 = params[train_trajectories, p2]
    tr_pt_3 = params[train_trajectories, p3]
    if test_trajectories:
        pt_1 = params[test_trajectories, p1]
        pt_2 = params[test_trajectories, p2]
        pt_3 = params[test_trajectories, p3]
    else:
        pt_1 = params[:, p1]
        pt_2 = params[:, p2]
        pt_3 = params[:, p3]
    fig = plt.figure('Relative Error 3D '+var_name)
    ax = fig.add_subplot(projection='3d')
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    colors = error.flatten()
    sc = ax.scatter(pt_1, pt_2, pt_3, s=1000*colors, c=colors, cmap=colormaps['jet'])
    cbar = plt.colorbar(sc, format=fmt, shrink=0.5, pad=0.1)
    tick_locator = MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.ax.yaxis.set_offset_position('left')
    cbar.update_ticks()
    ax.set(xlim=tuple([mu1_range[0], mu1_range[-1]]),
           ylim=tuple([mu2_range[0], mu2_range[-1]]),
           zlim=tuple([mu3_range[0], mu3_range[-1]]),
           xlabel=f'$\mu_{str((p1%n_params)+1)}$',
           ylabel=f'$\mu_{str((p2%n_params)+1)}$',
           zlabel=f'$\mu_{str((p3%n_params)+1)}$')
    ax.scatter(tr_pt_1, tr_pt_2, tr_pt_3, marker="*", color="red", s=10)
    ax.set_title('Relative Error 3D '+var_name)
    plt.tight_layout()
    plt.savefig(HyperParams.net_dir+'relative_error_3d_'+var_name+HyperParams.net_run+'.png', transparent=True, dpi=500)


def create_animation(SAMPLE, VAR_all, scaler_all, HyperParams, dataset, xyz, params, param_sample, comp="_U"):
    """
    Creates an animation for time-dependent solutions.

    Args:
        SAMPLE (int): Sample index indicating which parameter sample to animate.
        VAR_all (ndarray): Ground truth solution array.
        scaler_all (object): Scaler object used to scale the data.
        HyperParams (object): Hyperparameters object containing network architecture and training info.
        dataset (object): Dataset object containing mesh/triangulation information.
        xyz (list): List of coordinate arrays [x, y] or [x, y, z].
        params (ndarray): Parameter array associated with each snapshot.
        param_sample (int): Number of parameter samples.
        comp (str): Component suffix for file naming.

    Returns:
        animation.FuncAnimation: The created animation object.
    """


    fig = plt.figure()
    Z = scaling.inverse_scaling(VAR_all, scaler_all, HyperParams.scaling_type)
    xx = xyz[0].squeeze()
    yy = xyz[1].squeeze()
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    triang = np.asarray(dataset.T - 1)
    gs1 = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs1[0, 0])
    sequence_length = VAR_all.shape[0] // param_sample
    start = SAMPLE * sequence_length
    cs = ax.tricontourf(xx, yy, triang, Z[:, start].squeeze(), 100, cmap=colormaps['jet'])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(cs, cax=cax, format=fmt)
    tick_locator = MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.ax.yaxis.set_offset_position('left')
    cbar.update_ticks()

    def update_animation(i):
        cs = ax.tricontourf(xx, yy, triang, Z[:, i+start].squeeze(), 100, cmap=colormaps['jet'])
        plt.tight_layout()
        ax.set_aspect('equal', 'box')
        ax.set_title('Rollout Solution field for $\mu$ = '+str(np.around(params[SAMPLE][:-1].detach().numpy(), 2)))
        return cs

    anim = animation.FuncAnimation(fig=fig, func=update_animation, frames=sequence_length, interval=30)
    gif = animation.PillowWriter(fps=10)
    anim.save(HyperParams.net_dir+'rollout_field_solution_'+str(SAMPLE)+''+HyperParams.net_run+comp+'.gif', writer=gif)
    plt.close()
    return anim

def plot_struc_result(index, struct_results, scaler_all_reshaped, HyperParams, dataset, xyz, params, cand_list, mask_result, comp="_U"):
    """
    Plots the structured (grid) solution field for a given snapshot.

    Args:
        index (int): Snapshot index to plot.
        struct_results (tensor): Structured grid results from the network.
        scaler_all_reshaped (object): Scaler object for inverse scaling.
        HyperParams (object): Hyperparameters object containing network architecture and training info.
        dataset (object): Dataset object containing mesh/triangulation information.
        xyz (list): List of coordinate arrays [x, y] or [x, y, z].
        params (ndarray): Parameter array associated with each snapshot.
        cand_list (list): List of candidate indices.
        mask_result (tuple): Mask tuple containing (inmask, outmask, grid_points, X, Y).
        comp (str): Component suffix for file naming.
    """
    inmask = mask_result[0]
    outmask = mask_result[1]
    grid_points = mask_result[2]
    X = mask_result[3]
    Y = mask_result[4]
    res = struct_results
    res_perm = res.permute(1, 2, 0, 3)
    res_reshped = res_perm.reshape(res_perm.shape[0], res_perm.shape[1], -1)
    temp_data = res_reshped.reshape(-1, res_reshped.shape[2])
    rescaled_data = scaling.inverse_scaling(temp_data, scaler_all_reshaped, HyperParams.scaling_type)
    res_reshped = rescaled_data.reshape(res_perm.shape[0], res_perm.shape[1], -1)
    fig = plt.figure()
    z_net_temp = np.squeeze(res_reshped[:, :, index])
    z_net_temp = z_net_temp.ravel()
    outmask = outmask.astype(bool)  # NumPy
    z_net_temp[outmask.ravel()] = np.nan
    z_net = z_net_temp.reshape(X.shape)
    cand_list = np.array(cand_list)
    flat_cand_list = cand_list.flatten()
    SNAP = flat_cand_list[index]

    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    if dataset.dim == 2:
        gs1 = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs1[0, 0])
        cs = ax.imshow(
            z_net,
            origin='lower',
            cmap='viridis',
            aspect='equal'  # or 'equal', adjust aspect ratio as needed
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(cs, cax=cax, format=fmt)
    elif dataset.dim == 3:
        zz = xyz[2]
        ax = fig.add_subplot(projection='3d')
        cax = inset_axes(ax, width="5%", height="60%", loc="center left",
                         bbox_to_anchor=(1.15, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        p = ax.scatter(xx, yy, zz, c=z_net, cmap=colormaps['jet'], linewidth=0.5)
        cbar = fig.colorbar(p, ax=ax, cax=cax, format=fmt)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        ax.locator_params(axis='both', nbins=5)
    tick_locator = MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.ax.yaxis.set_offset_position('left')
    cbar.update_ticks()
    plt.tight_layout()
    ax.set_aspect('equal', 'box')
    ax.set_title('struc_solution field for $\mu$ = '+str(np.around(params[SNAP].detach().numpy(), 2)))
    plt.savefig(HyperParams.net_dir+'struc_solution_'+str(SNAP)+''+HyperParams.net_run+comp+'.png', bbox_inches='tight', dpi=500)


def plot_modes(index, modes, HyperParams, mask_result, comp="_U"):
    """
    Plots the decomposition modes and their temporal frequency spectrum.

    Args:
        index (int): Mode set index for file naming.
        modes (tuple): Tuple of tensor factors (U_r, V_r, W_r, core_r).
        HyperParams (object): Hyperparameters object containing network architecture and training info.
        mask_result (tuple): Mask tuple containing (inmask, outmask, grid_points, X, Y).
        comp (str): Component suffix for file naming.
    """
    U_r = modes[0]
    V_r = modes[1]
    W_r = modes[2]
    core_r = modes[3]
    inmask = mask_result[0]
    outmask = mask_result[1]
    grid_points = mask_result[2]
    X = mask_result[3]
    Y = mask_result[4]
    mod_num = HyperParams.mode_num
    mods = loss_func.mode_func(mod_num, U_r, core_r, V_r)
    col_num = HyperParams.col_num
    row_num = int(np.ceil(mod_num / col_num))
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))

    # One colorbar per subplot
    plt.figure(figsize=(30, 20))
    for i in range(mod_num):
        ax = plt.subplot(row_num, col_num, i + 1)
        temp_data = np.squeeze(mods[:, :, i])
        z_net_temp = temp_data.ravel()
        outmask = outmask.astype(bool)  # NumPy
        z_net_temp[outmask.ravel()] = np.nan
        z_net = z_net_temp.reshape(X.shape)
        contour = ax.imshow(
            z_net,
            origin='lower',
            cmap='viridis',
            aspect='equal'  # or 'equal', adjust aspect ratio as needed
        )
        plt.title(f'Modes_{i + 1}', fontsize=30)
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(contour, cax=cax, format=fmt)
        tick_locator = MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.ax.yaxis.set_offset_position('left')
        cbar.update_ticks()
        plt.tight_layout()
        ax.set_aspect('equal', 'box')
    plt.savefig(HyperParams.net_dir + 'modes_' + str(index) + '' + HyperParams.net_run + comp + '.png',
                bbox_inches='tight', dpi=500)

    time_steps = W_r.shape[0]
    num_dims = W_r.shape[1]
    sampling_interval = 0.01  # Sampling interval (s)
    sampling_rate = 1 / sampling_interval  # Sampling rate (Hz)
    time = np.linspace(0, (time_steps - 1) * sampling_interval, time_steps)  # Time axis
    data = W_r
    # Compute single-sided spectrum
    frequencies = np.fft.fftfreq(time_steps, d=sampling_interval)  # Frequency axis
    positive_frequencies = frequencies[frequencies >= 0]  # Positive frequency part
    plt.figure(figsize=(15, 12))
    # Subplot 1: Time series
    ax1 = plt.subplot(2, 1, 1)
    for i in range(num_dims):
        plt.plot(time, data[:, i], label=f"mode {i + 1}")
    plt.title("Time Series")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc='best', ncol=3)  # Automatically select the best location
    plt.grid()
    # Subplot 2: Frequency spectrum
    ax2 = plt.subplot(2, 1, 2)
    for i in range(num_dims):
        # Compute Fourier transform
        fft_values = np.fft.fft(data[:, i])
        fft_magnitude = np.abs(fft_values) / time_steps  # Amplitude normalization
        positive_magnitude = fft_magnitude[frequencies >= 0]
        # Plot single-sided spectrum
        plt.plot(positive_frequencies, positive_magnitude, label=f"mode {i + 1}")
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend(loc='best', ncol=3)  # Automatically select the best location
    plt.grid()
    # Layout adjustment
    plt.tight_layout()
    plt.savefig(HyperParams.net_dir + 'temporal_frequency_' + str(index) + '' + HyperParams.net_run + comp + '.png',
                bbox_inches='tight', dpi=500)
