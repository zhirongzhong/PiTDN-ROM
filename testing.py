import torch
from tqdm import tqdm
import numpy as np
from PiTDN import resample_func


def evaluate(VAR, model, loader, params, HyperParams, test_trajectories, interpolator, core):
    """Evaluate a trained PiTDN-ROM model on test data.

    Encodes input data using both the model's encoder and the parameter-to-latent
    mapping function, then decodes latent representations to obtain predicted
    solutions. Also computes the relative error between the two latent representations.

    Args:
        VAR: Ground truth solution array.
        model: Trained PiTDN-ROM model.
        loader: DataLoader for the input graph data.
        params: Model parameters array.
        HyperParams: Model architecture and training hyperparameters.
        test_trajectories: Indices of test trajectories.
        interpolator: Mesh interpolator for structured/unstructured conversion.
        core: Core tensor for TDN reconstruction.

    Returns:
        latents_map: Latent representations from the parameter-to-latent mapping.
        latents_encoder: Latent representations from the autoencoder encoder.
        unstruct_results: Reconstructed unstructured field results.
        struct_results: Reconstructed structured field results.
        modes_results: TDN mode decomposition results (U, V, W, core).
    """
    serial_var = VAR[test_trajectories, :]
    unstruct_results = torch.zeros(serial_var.shape[0], serial_var.shape[1], serial_var.shape[2], HyperParams.comp)
    struct_results = torch.zeros(serial_var.shape[0], HyperParams.n_1, HyperParams.n_2, HyperParams.n_3)
    latents_map = torch.zeros(serial_var.shape[0], serial_var.shape[1]*HyperParams.bottleneck_dim)
    latents_encoder = torch.zeros(serial_var.shape[0], serial_var.shape[1]*HyperParams.bottleneck_dim)
    index = 0
    latents_error = list()
    modes_results = list()
    with torch.no_grad():
        for data in tqdm(loader):
            z_net = model.solo_encoder(data)
            z_map = model.mapping(params[test_trajectories[index], :])
            latents_map[index, :] = z_map.flatten()
            latents_encoder[index, :] = z_net.flatten()
            lat_err = np.linalg.norm(z_net.detach().numpy() - z_map.detach().numpy())/np.linalg.norm(z_net.detach().numpy())
            latents_error.append(lat_err)
            out, U_out, V_out, W_out, core_out = model.TDN_decoder(core, z_map)
            test_resample_out = resample_func.resample_NNR(out, HyperParams)

            unstruct_results[index, :, :] = torch.tensor(test_resample_out.T).unsqueeze(-1)
            struct_results[index, :, :, :] = out
            modes = [U_out, V_out, W_out, core_out]
            modes_results.append(modes)
            index += 1
        np.savetxt(HyperParams.net_dir+'latents'+HyperParams.net_run+'.csv', latents_map.detach(), delimiter =',')
        latents_error = np.array(latents_error)
    return latents_map, latents_encoder, unstruct_results, struct_results, modes_results