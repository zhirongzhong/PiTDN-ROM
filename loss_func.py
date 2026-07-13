"""Loss functions for the PiTDN-ROM training pipeline.

Includes:
- Orthogonality loss for tensor decomposition modes
- Frequency-domain regularization
- Diagonal energy concentration loss
- Geometry-aware boundary loss
- Composite TDN loss combining all regularizers
"""

import torch
import torch.nn.functional as nF
import numpy as np


def mode_func(mod_num, U_out_, core_out, V_out_):
    """Compute tensor decomposition modes from factor matrices.

    Args:
        mod_num: Number of modes.
        U_out_: U factor matrix.
        core_out: Core tensor.
        V_out_: V factor matrix.

    Returns:
        Modes tensor of shape (U_rows, V_rows, mod_num).
    """
    mods = torch.zeros(U_out_.shape[0], V_out_.shape[0], mod_num,
                       dtype=torch.double)
    for j in range(mod_num):
        mods[:, :, j] = torch.matmul(
            U_out_, torch.matmul(core_out[:, :, j], V_out_.T))
    return mods


def loss_ort(mods, mod_num):
    """Compute orthogonality loss via Gram matrix deviation from identity.

    Args:
        mods: Mode matrices.
        mod_num: Number of modes.

    Returns:
        Frobenius norm of (Gram - I).
    """
    b, h, w = mods.shape
    mods_2d = mods.reshape(b * h, mod_num)
    gram_matrix = torch.mm(mods_2d.T, mods_2d)
    identity_matrix = torch.eye(gram_matrix.shape[0], device=mods_2d.device)
    loss_orth = torch.norm(gram_matrix - identity_matrix, p='fro')
    return loss_orth


def kl_divergence(p, q):
    """Compute KL divergence P || Q for probability distributions.

    Args:
        p, q: Input probability distributions (each column is a spectrum).

    Returns:
        KL divergence loss.
    """
    p = p / p.sum(dim=0, keepdim=True)
    q = q / q.sum(dim=0, keepdim=True)
    kl_loss = nF.kl_div(torch.log(q), p, reduction='none')
    return kl_loss


def fft_time_factor(W_Out):
    """Compute single-sided amplitude spectrum along the time dimension.

    Args:
        W_Out: Time-factor matrix.

    Returns:
        Single-sided amplitude spectrum.
    """
    fft_result = torch.fft.fft(W_Out, dim=0)
    amplitude_spectrum = torch.abs(fft_result)
    m = W_Out.shape[0]
    single_sided_spectrum = amplitude_spectrum[:m // 2 + 1, :]
    single_sided_spectrum[1:] *= 2
    return single_sided_spectrum


def loss_fre(single_sided_spectrum, n):
    """Compute frequency-domain sparsity loss.

    Args:
        single_sided_spectrum: Single-sided FFT spectrum.
        n: Number of frequency bins to penalize.

    Returns:
        Average L1 norm over the first n frequency components.
    """
    loss_freq = 0.0
    for i in range(n):
        loss = torch.norm(single_sided_spectrum[:i], p=1)
        loss_freq += loss
    loss_freq = loss_freq / n
    return loss_freq


def diagonal_loss(centre):
    """Compute energy concentration loss on diagonal of core tensor.

    Encourages the core tensor to have energy concentrated on its main
    diagonal in decreasing order.

    Args:
        centre: Core tensor of shape (r1, r2, r3).

    Returns:
        Combined MSE loss for diagonal sorting.
    """
    r1, r2, r3 = centre.shape
    min_dim = min(r1, r2, r3)
    diagonal_elements = torch.tensor(
        [centre[i, i, i] for i in range(min_dim)], device=centre.device)
    sorted_diagonal, _ = torch.sort(diagonal_elements, descending=True)
    new_tensor = torch.zeros_like(centre)
    for i in range(min_dim):
        new_tensor[i, i, i] = sorted_diagonal[i]
    loss_p = nF.mse_loss(new_tensor, centre)
    diag_sort_loss = nF.mse_loss(diagonal_elements, sorted_diagonal)
    return loss_p + diag_sort_loss


def loss_geom(modes, outmask):
    """Compute geometry-aware boundary loss.

    Penalizes non-zero values outside the physical domain defined by outmask.

    Args:
        modes: Mode tensor of shape (H, W, depth).
        outmask: Boolean mask where True = outside domain.

    Returns:
        Mean absolute value of out-of-domain entries.
    """
    if isinstance(outmask, np.ndarray):
        outmask = torch.from_numpy(outmask)

    outmask = outmask.to(dtype=torch.bool, device=modes.device)
    mask_flat = outmask.reshape(-1)
    masked_cnt = mask_flat.sum()
    assert masked_cnt > 0, "outmask is all False; check input data."

    loss_geo = 0.0
    depth = modes.shape[2]
    for i in range(depth):
        matrix = modes[:, :, i]
        z_flat = matrix.reshape(-1)
        loss_geo += torch.sum(torch.abs(z_flat[mask_flat]))
    loss_geo = loss_geo / (depth * masked_cnt)
    return loss_geo


def loss_tdn(U_Out, V_Out, W_Out, core_Out, mod_num, w, outmask):
    """Composite TDN loss combining orthogonality, frequency, and energy terms.

    Args:
        U_Out, V_Out, W_Out: Factor matrices from TDN.
        core_Out: Core tensor from TDN.
        mod_num: Number of decomposition modes.
        w: List of 3 weights for [orthogonality, frequency, energy] losses.
        outmask: Geometry mask for optional boundary loss.

    Returns:
        Weighted sum of regularization losses.
    """
    loss_total = 0.0
    w_tensor = torch.tensor(w).float()
    w_normalized = torch.softmax(w_tensor, dim=0)

    if w_tensor[0] != 0:
        modes = mode_func(mod_num, U_Out, core_Out, V_Out)
        loss_orth_val = loss_ort(modes, mod_num)
        loss_total += w_normalized[0] * loss_orth_val

    if w_tensor[1] != 0:
        single_sided_spectrum = fft_time_factor(W_Out)
        loss_freq_val = loss_fre(single_sided_spectrum, mod_num)
        loss_freq_val = loss_freq_val.to(w_normalized.device)
        loss_total += w_normalized[1] * loss_freq_val

    if w_tensor[2] != 0:
        loss_enge = diagonal_loss(core_Out)
        loss_total += w_normalized[2] * loss_enge

    return loss_total
