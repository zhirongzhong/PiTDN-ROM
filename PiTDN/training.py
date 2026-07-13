"""Training loop for the PiTDN-ROM model.

Implements the main training procedure with multi-component loss:
- MSE reconstruction loss
- Parameter-to-latent mapping loss
- Sparse encoder alignment loss
- Tensor decomposition regularization (TDN loss)
- Geometry-aware boundary loss
"""

import torch
import gc
import torch.nn.functional as nF
import numpy as np
from tqdm import tqdm
from PiTDN import loss_func, resample_func, scaling


def train(model, optimizer, device, scheduler, params, train_loader,
          test_loader, train_trajectories, test_trajectories, HyperParams,
          sub_train_loader, sub_test_loader, mask_result,
          scaler_all_reshaped, interpolator, centre):
    """Train the PiTDN-ROM autoencoder model.

    The total loss is a weighted combination of:
    - MSE loss (lambda_mse): reconstruction error on unstructured mesh
    - Map loss (lambda_map): latent space prediction error
    - Sparse loss (lambda_sp): sparse encoder alignment
    - TDN loss (lambda_tdn): tensor decomposition regularization
    - Geo loss (lambda_geo): boundary/geometry constraint

    Args:
        model: PiTDN-ROM network.
        optimizer: PyTorch optimizer.
        device: Compute device ('cuda' or 'cpu').
        scheduler: Learning rate scheduler.
        params: Parameter array for latent mapping.
        train_loader: DataLoader for full training graphs.
        test_loader: DataLoader for full test graphs.
        train_trajectories: Training trajectory indices.
        test_trajectories: Test trajectory indices.
        HyperParams: Configuration object.
        sub_train_loader: DataLoader for sparse training graphs.
        sub_test_loader: DataLoader for sparse test graphs.
        mask_result: Geometry mask data.
        scaler_all_reshaped: Scalers for inverse transformation.
        interpolator: Mesh interpolator (structured/unstructured).
        centre: Initial core tensor for TDN.

    Returns:
        Updated core tensor after training.
    """
    train_history = dict(train=[], l1=[], l2=[], l3=[], l4=[], l5=[])
    test_history = dict(test=[], l1=[], l2=[], l3=[], l4=[], l5=[])
    outmask = mask_result[1]
    min_test_loss = np.inf
    mod_num = HyperParams.r_3
    w = HyperParams.w
    model.train()
    loop = tqdm(range(HyperParams.max_epochs))

    for epoch in loop:
        train_rmse = sum_loss = 0
        train_rmse_1 = train_rmse_2 = train_rmse_3 = train_rmse_4 = train_rmse_5 = 0
        start_ind = 0

        if HyperParams.minibatch:
            total_batches = 0
            for data in train_loader:
                optimizer.zero_grad()
                data = data.to(device)
                out, z, z_estimation, U_out, V_out, W_out, core_out = model(
                    data, params[train_trajectories[start_ind], :])
                loss_train_mse = nF.mse_loss(out, data.x, reduction='mean')
                loss_train_map = nF.mse_loss(z_estimation, z, reduction='mean')
                loss_train = (loss_train_mse
                              + HyperParams.lambda_map * loss_train_map)
                loss_train.backward()
                optimizer.step()
                train_rmse += loss_train.item()
                train_rmse_1 += loss_train_mse.item()
                train_rmse_2 += loss_train_map.item()
                total_batches += 1
            train_rmse /= total_batches
            train_rmse_1 /= total_batches
            train_rmse_2 /= total_batches
        else:
            optimizer.zero_grad()

            loss_train_mse_total = 0
            loss_train_map_total = 0
            loss_train_tdn_total = 0
            loss_train_sp_total = 0
            loss_train_geo_total = 0
            loss_train_total = 0

            train_trajectories_1d = [
                item for sublist in train_trajectories
                for item in sublist]
            start_ind = 0

            for i, (data, spdata) in enumerate(
                    tqdm(zip(train_loader, sub_train_loader),
                         total=len(train_loader), desc="Training")):
                data = data.to(device)
                spdata = spdata.to(device)

                index_tensor = torch.tensor(
                    train_trajectories[start_ind], device='cpu')
                scale_train = [t[index_tensor] for t in scaler_all_reshaped]

                out, z, z_estimation, U_out, V_out, W_out, core_out, z_sp = \
                    model(centre, data,
                          params[train_trajectories[start_ind], :], spdata)
                centre = core_out

                # Resample structured output to unstructured mesh (no grad)
                with torch.no_grad():
                    resample_out = interpolator.batch_interpolate(out)
                    resample_1d = torch.tensor(
                        resample_out.reshape(-1, 1), device=device)

                # Compute loss components
                loss_mse = torch.norm(resample_1d - data.x, 2) / (
                    len(train_trajectories_1d) * HyperParams.num_nodes)
                loss_map = nF.mse_loss(z_estimation, z, reduction='sum') / (
                    len(train_trajectories_1d) * HyperParams.bottleneck_dim)
                loss_sp = nF.mse_loss(z_sp, z, reduction='sum') / (
                    len(train_trajectories_1d) * HyperParams.bottleneck_dim)
                loss_tdn = loss_func.loss_tdn(
                    U_out, V_out, W_out, core_out, mod_num, w, outmask) / (
                    len(train_trajectories_1d) * HyperParams.n_1
                    * HyperParams.n_2 * HyperParams.r_3)

                # Geometry loss with inverse scaling (no grad)
                out_reshape = out.reshape(-1, out.shape[2]).to('cpu')
                with torch.no_grad():
                    out_rescale = scaling.inverse_scaling(
                        out_reshape, scale_train,
                        HyperParams.scaling_type).to(device)
                out_rescale = out_rescale.reshape_as(out)
                loss_geo = loss_func.loss_geom(out_rescale, outmask)

                # Aggregate total loss
                loss = (HyperParams.lambda_mse * loss_mse
                        + HyperParams.lambda_map * loss_map
                        + HyperParams.lambda_sp * loss_sp
                        + HyperParams.lambda_tdn * loss_tdn
                        + HyperParams.lambda_geo * loss_geo)

                loss_train_total += loss

                # Record detached losses
                loss_train_mse_total += loss_mse.detach()
                loss_train_map_total += loss_map.detach()
                loss_train_tdn_total += loss_tdn.detach()
                loss_train_sp_total += loss_sp.detach()
                loss_train_geo_total += loss_geo.detach()

                start_ind += 1

                # Clean up
                del data, spdata, index_tensor, scale_train
                del out, z, z_estimation, U_out, V_out, W_out, z_sp
                del resample_out, resample_1d, out_reshape, out_rescale
                torch.cuda.empty_cache()

            # Backward pass and optimizer step
            loss_train_total.backward()
            optimizer.step()

            # Accumulate epoch metrics
            train_rmse += loss_train_total.item()
            train_rmse_1 += loss_train_mse_total.item()
            train_rmse_2 += loss_train_map_total.item()
            train_rmse_3 += loss_train_tdn_total.item()
            train_rmse_4 += loss_train_sp_total.item()
            train_rmse_5 += loss_train_geo_total.item()

            gc.collect()
            torch.cuda.empty_cache()

        scheduler.step()

        train_history['train'].append(train_rmse)
        train_history['l1'].append(train_rmse_1)
        train_history['l2'].append(train_rmse_2)
        train_history['l3'].append(train_rmse_3)
        train_history['l4'].append(train_rmse_4)
        train_history['l5'].append(train_rmse_5)

        # Cross-validation
        if HyperParams.cross_validation:
            with torch.no_grad():
                model.eval()
                test_rmse = test_rmse_1 = test_rmse_2 = test_rmse_3 = \
                    test_rmse_4 = test_rmse_5 = 0

                loss_test_mse = 0
                loss_test_map = 0
                loss_test_tdn = loss_test_sp = loss_test_geo = 0

                start_ind = 0
                test_trajectories_1d = [
                    item for sublist in test_trajectories
                    for item in sublist]

                for i, (data, spdata) in enumerate(
                        zip(test_loader, sub_test_loader)):
                    data = data.to(device)
                    spdata = spdata.to(device)
                    centre = core_out

                    index_tensor = torch.tensor(
                        test_trajectories[start_ind], device='cpu')
                    scale_test = [t[index_tensor]
                                  for t in scaler_all_reshaped]

                    param_input = params[
                        test_trajectories[start_ind], :].to(device)

                    out, z, z_estimation, U_out, V_out, W_out, core_out, \
                        z_sp = model(centre, data, param_input, spdata)


                    test_resample_out = interpolator.batch_interpolate(
                            out)
                    test_resample_1d = torch.tensor(
                        test_resample_out.reshape(-1, 1), device=device)

                    loss_test_tdn += loss_func.loss_tdn(
                        U_out, V_out, W_out, core_out,
                        mod_num, w, outmask).detach() / (
                        len(test_trajectories_1d) * HyperParams.r_3)
                    loss_test_mse += torch.norm(
                        test_resample_1d - data.x, 2).detach() / (
                        len(test_trajectories_1d) * HyperParams.num_nodes)
                    loss_test_map += nF.mse_loss(
                        z_estimation, z, reduction='sum').detach() / (
                        len(test_trajectories_1d)
                        * HyperParams.bottleneck_dim)
                    loss_test_sp += nF.mse_loss(
                        z_sp, z, reduction='sum').detach() / (
                        len(test_trajectories_1d)
                        * HyperParams.bottleneck_dim)

                    out_reshape = out.reshape(-1, out.shape[2]).to('cpu')
                    out_rescale = scaling.inverse_scaling(
                        out_reshape, scale_test,
                        HyperParams.scaling_type).to(device)
                    out_rescale = out_rescale.reshape_as(out)
                    loss_test_geo += loss_func.loss_geom(
                        out_rescale, outmask).detach()

                    start_ind += 1

                    del data, spdata, index_tensor, scale_test, param_input
                    del out, z, z_estimation, U_out, V_out, W_out, z_sp
                    del test_resample_out, test_resample_1d, out_reshape, \
                        out_rescale
                    torch.cuda.empty_cache()

                # Aggregate test loss
                loss_test = (HyperParams.lambda_mse * loss_test_mse
                             + HyperParams.lambda_map * loss_test_map
                             + HyperParams.lambda_sp * loss_test_sp
                             + HyperParams.lambda_tdn * loss_tdn
                             + HyperParams.lambda_geo * loss_geo)

                test_rmse += loss_test.item()
                test_rmse_1 += loss_test_mse.item()
                test_rmse_2 += loss_test_map.item()
                test_rmse_3 += loss_test_tdn.item()
                test_rmse_4 += loss_test_sp.item()
                test_rmse_5 += loss_test_geo.item()

                test_history['test'].append(test_rmse)
                test_history['l1'].append(test_rmse_1)
                test_history['l2'].append(test_rmse_2)
                test_history['l3'].append(test_rmse_3)
                test_history['l4'].append(test_rmse_4)
                test_history['l5'].append(test_rmse_5)

                gc.collect()
                torch.cuda.empty_cache()

            loop.set_postfix({
                "Loss(training)": train_history['train'][-1],
                "Loss(validation)": test_history['test'][-1]})
        else:
            test_rmse = train_rmse
            loop.set_postfix({
                "Loss(training)": train_history['train'][-1]})

        if test_rmse < min_test_loss:
            min_test_loss = test_rmse
            best_epoch = epoch
            torch.save(
                model.state_dict(),
                HyperParams.net_dir + HyperParams.net_name
                + HyperParams.net_run + '.pt')
            torch.save(
                model,
                HyperParams.net_dir + HyperParams.net_name
                + HyperParams.net_run + '_all.pt')
            np.save(
                HyperParams.net_dir + HyperParams.net_name
                + HyperParams.net_run + '_core_out.npy',
                core_out.detach().cpu().numpy())

        if HyperParams.tolerance >= train_rmse:
            print('Early stopping!')
            break

        np.save(HyperParams.net_dir + 'history' + HyperParams.net_run
                + '.npy', train_history)
        np.save(HyperParams.net_dir + 'history_test'
                + HyperParams.net_run + '.npy', test_history)

    print("\nLoading best network for epoch: ", best_epoch)
    model.load_state_dict(torch.load(
        HyperParams.net_dir + HyperParams.net_name + HyperParams.net_run
        + '.pt', map_location=torch.device('cpu')))
    return core_out
