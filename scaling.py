"""Data scaling and normalization utilities for PiTDN-ROM.

Provides configurable scaling strategies (min-max, robust, standard, z-score)
for preprocessing field data before training the autoencoder.
"""

from sklearn import preprocessing
import torch


def scaler_functions(k):
    """Return a scikit-learn scaler and its name based on index.

    Args:
        k: Scaler selector (1=MinMax, 2=Robust, 3=Standard).

    Returns:
        Tuple of (scaler_instance, scaler_name_string).
    """
    if k == 1:
        sc_fun = preprocessing.MinMaxScaler()
        sc_name = "minmax"
    elif k == 2:
        sc_fun = preprocessing.RobustScaler()
        sc_name = "robust"
    elif k == 3:
        sc_fun = preprocessing.StandardScaler()
        sc_name = "standard"
    return sc_fun, sc_name


def tensor_scaling(tensor, scaling_type, scaler_name):
    """Apply scaling to a tensor using the specified strategy.

    Supports four scaling modes:
    1 - Sample scaling (scale each sample independently)
    2 - Feature scaling (scale each feature across samples)
    3 - Feature-sample two-stage scaling
    4 - Z-score normalization (mean/std)

    Args:
        tensor: Input data tensor.
        scaling_type: Integer 1-4 specifying the scaling strategy.
        scaler_name: Integer 1-3 selecting the base scaler type.

    Returns:
        Tuple of (scaler_object, scaled_tensor).
    """
    scaling_fun_1, _ = scaler_functions(int(scaler_name))
    scaling_fun_2, _ = scaler_functions(int(scaler_name))

    if scaling_type == 1:
        scale = scaling_fun_1.fit(tensor)
        scaled_data = torch.unsqueeze(
            torch.tensor(scale.transform(tensor)), 0).permute(2, 1, 0)
    elif scaling_type == 2:
        scale = scaling_fun_1.fit(torch.t(tensor))
        scaled_data = torch.unsqueeze(
            torch.tensor(scale.transform(torch.t(tensor))), 0).permute(1, 2, 0)
    elif scaling_type == 3:
        scaler_f = scaling_fun_1.fit(torch.t(tensor))
        temp = torch.tensor(scaler_f.transform(torch.t(tensor)))
        scaler_s = scaling_fun_2.fit(temp)
        scaled_data = torch.unsqueeze(
            torch.tensor(scaler_s.transform(temp)), 0).permute(1, 2, 0)
        scale = [scaler_f, scaler_s]
    elif scaling_type == 4:
        mean = tensor.mean(axis=0)
        std = tensor.std(axis=0)
        scaled_data = (tensor - mean) / std
        scale = [mean, std]
        scaled_data = torch.unsqueeze(
            torch.tensor(scaled_data), 0).permute(2, 1, 0)

    return scale, scaled_data


def inverse_scaling(tensor, scale, scaling_type):
    """Reverse the scaling operation to recover original-scale data.

    Args:
        tensor: Scaled tensor to be inverse-transformed.
        scale: Scaler object (or [mean, std] list for type 3/4).
        scaling_type: Integer 1-4 specifying which scaling was used.

    Returns:
        Tensor in the original data scale.
    """
    if scaling_type == 1:
        rescaled_data = torch.tensor(
            scale.inverse_transform(
                torch.t(torch.tensor(
                    tensor[:, :, 0].detach().numpy().squeeze()))))
    elif scaling_type == 2:
        rescaled_data = torch.tensor(
            torch.t(torch.tensor(
                scale.inverse_transform(
                    tensor[:, :, 0].detach().numpy().squeeze()))))
    elif scaling_type == 3:
        mean = scale[0]
        std = scale[1]
        rescaled_data = tensor * std + mean
    elif scaling_type == 4:
        mean = scale[0]
        std = scale[1]
        rescaled_data = tensor * std + mean

    return rescaled_data
