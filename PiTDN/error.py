"""Error computation and reporting utilities for PiTDN-ROM."""

import numpy as np
from PiTDN import scaling


def save_error(error, norm, HyperParams, var_name):
    """Save relative error statistics to a text file.

    Args:
        error: List of absolute error values.
        norm: List of norm values (same length as error).
        HyperParams: Configuration object (provides net_dir, net_run).
        var_name: Variable name string for the output filename.
    """
    error = np.array(error)
    norm = np.array(norm)
    rel_error = error / norm
    np.savetxt(
        HyperParams.net_dir + 'relative_errors' + HyperParams.net_run
        + var_name + '.txt',
        [max(rel_error), sum(rel_error) / len(rel_error), min(rel_error)])


def print_error(error, norm, var_name):
    """Print absolute and relative error statistics.

    Args:
        error: List of absolute error values.
        norm: List of norm values (same length as error).
        var_name: Variable name for display.
    """
    error = np.array(error)
    norm = np.array(norm)
    rel_error = error / norm
    print("\nMaximum absolute error for field " + var_name + " = ",
          max(error))
    print("Mean absolute error for field " + var_name + " = ",
          sum(error) / len(error))
    print("Minimum absolute error for field " + var_name + " = ",
          min(error))
    print("\nMaximum relative error for field " + var_name + " = ",
          max(rel_error))
    print("Mean relative error for field " + var_name + " = ",
          sum(rel_error) / len(rel_error))
    print("Minimum relative error for field " + var_name + " = ",
          min(rel_error))


def compute_error(res, VAR, scaler, HyperParams):
    """Compute absolute and relative errors between reconstructed and true data.

    Args:
        res: Reconstructed data from the autoencoder.
        VAR: Original ground truth data.
        scaler: Scaler object for inverse transformation.
        HyperParams: Configuration object (provides scaling_type).

    Returns:
        Tuple of (error_abs_list, norm_z_list).
    """
    error_abs_list = []
    norm_z_list = []
    Z = scaling.inverse_scaling(VAR, scaler, HyperParams.scaling_type)
    Z_net = scaling.inverse_scaling(res, scaler, HyperParams.scaling_type)
    for snap in range(VAR.shape[0]):
        error_abs = np.linalg.norm(abs(Z[:, snap] - Z_net[:, snap]))
        norm_z = np.linalg.norm(Z[:, snap], 2)
        error_abs_list.append(error_abs)
        norm_z_list.append(norm_z)
    return error_abs_list, norm_z_list
