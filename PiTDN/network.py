import torch
import numpy as np
from torch import nn
from PiTDN import modules, scaling
import torch.nn.functional as F


class HyperParams:
    """Class that holds the hyperparameters for the autoencoder model.

    Args:
        sparse_method (str): The method to use for sparsity constraint.
        rate (int): Amount of data used in training.
        seed (int): Seed for the random number generator.
        bottleneck_dim (int): The dimension of the bottleneck layer.
        tolerance (float): The tolerance value for stopping the training.
        lambda_map (float): The weight for the map loss.
        learning_rate (float): The learning rate for the optimizer.
        ffn (int): The number of feed-forward layers.
        in_channels (int): The number of input channels.
        hidden_channels (list): The number of hidden channels for each layer.
        act (function): The activation function to use.
        nodes (int): The number of nodes in each hidden layer.
        skip (int): The number of skipped connections.
        layer_vec (list): The structure of the network.
        net_name (str): The name of the network.
        scaler_name (str): The name of the scaler used for preprocessing.
        weight_decay (float): The weight decay for the optimizer.
        max_epochs (int): The maximum number of epochs to run training for.
        miles (list): The miles for learning rate update in scheduler.
        gamma (float): The gamma value for the optimizer.
        num_nodes (int): The number of nodes in the network.
        scaling_type (int): The type of scaling to use for preprocessing.
        net_dir (str): The directory to save the network in.
        cross_validation (bool): Whether to perform cross-validation.
    """

    def __init__(self, argv, **kwargs):
        self.net_name = argv[0]
        self.variable = argv[1]
        self.scaling_type = int(argv[2])
        self.scaler_number = int(argv[3])
        _, self.scaler_name = scaling.scaler_functions(self.scaler_number)
        self.skip = int(argv[4])
        self.rate = int(argv[5])
        self.sparse_method = 'L1_mean'
        self.ffn = int(argv[6])
        self.nodes = int(argv[7])
        self.bottleneck_dim = int(argv[8])
        self.lambda_map = float(argv[9])
        self.in_channels = int(argv[10])
        self.seed = 3407
        self.tolerance = 1e-6
        self.learning_rate = 0.001
        self.map_act = 'tanh'
        self.layer_vec=[argv[11], self.nodes, self.nodes, self.nodes, self.nodes, self.bottleneck_dim]
        self.net_run = '_' + self.scaler_name
        self.weight_decay = 0.00001
        self.max_epochs = argv[12]
        self.comp = argv[13]
        self.hidden_channels = [self.comp]*self.in_channels
        self.miles = []
        self.gamma = 0.0001
        self.num_nodes = 0
        self.conv = 'GMMConv'
        self.ae_act = 'elu'
        self.batch_size = np.inf
        self.minibatch = False
        self.net_dir = './' + self.net_name + '/' + self.net_run + '/' + self.variable + '_' + self.net_name + '_lmap' + str(self.lambda_map) + '_btt' + str(self.bottleneck_dim) \
                            + '_seed' + str(self.seed) + '_lv' + str(len(self.layer_vec)-2) + '_hc' + str(len(self.hidden_channels)) + '_nd' + str(self.nodes) \
                            + '_ffn' + str(self.ffn) + '_skip' + str(self.skip) + '_lr' + str(self.learning_rate) + '_sc' + str(self.scaling_type) + '_rate' + str(self.rate) + '_conv' + self.conv + '/'
        self.cross_validation = True
        self.timesteps_portion = 0.8
        self.w_decay = 3
        self.lr_real = 0.0001
        self.down = [2, 2, 5]  # 核心张量在各维度的缩放尺度
        self.omega = 2
        self.n_1 = 100
        self.n_2 = 90
        self.n_3 = 64 # snapshot number
        self.mid_channel = int(self.n_2)
        self.r_1 = int(self.n_1 / self.down[0])
        self.r_2 = int(self.n_2 / self.down[1])
        self.r_3 = int(self.n_3 / self.down[2])
        self.w = [2, 1, 1]  # 物理损失权重
        self.Width = None
        self.Height = None
        self.dataset_dir = None
        self.mask_dataset_dir = None
        self.infeature = self.bottleneck_dim * self.n_3
        self.lambda_tdn = 0.01
        self.lambda_sp = 0.1
        self.lambda_mesh = 1
        self.lambda_mse = 1
        self.ifsparse = True
        self.num_samples = 50
        self.snap_nums = 64
        self.col_num = 4  # 模态列数
        self.lambda_geo = 0.1  # 几何损失权重
        self.ifneuralmap = True  # 是否考虑稀疏传感器

def get_activation(act_str):
    return getattr(F, act_str)

class Net(torch.nn.Module):
    """PiTDN-ROM main network combining graph autoencoder with tensor decomposition.

    The network consists of:
    - A graph autoencoder (encoder + decoder) for unstructured field compression
    - A sparse encoder for sensor-sparse observations
    - A vector mapping network (vecmap) to map latent codes to factor matrices
    - A Tensor Decomposition Network (TDN) for structured field reconstruction
    - A parameter-to-latent mapping MLP for parametric predictions

    Args:
        HyperParams: Configuration object with all model hyperparameters.
    """

    def __init__(self, HyperParams):
        super().__init__()
        self.encoder = modules.Encoder(HyperParams.hidden_channels, HyperParams.bottleneck_dim, HyperParams.num_nodes, ffn=HyperParams.ffn, skip=HyperParams.skip, act=get_activation(HyperParams.ae_act), conv=HyperParams.conv)
        self.spencoder = modules.Encoder(HyperParams.hidden_channels, HyperParams.bottleneck_dim, HyperParams.num_samples, ffn=HyperParams.ffn, skip=HyperParams.skip, act=get_activation(HyperParams.ae_act), conv=HyperParams.conv)
        self.decoder = modules.Decoder(HyperParams.hidden_channels, HyperParams.bottleneck_dim, HyperParams.num_nodes, ffn=HyperParams.ffn, skip=HyperParams.skip, act=get_activation(HyperParams.ae_act), conv=HyperParams.conv)
        # self.vecmap = modules.vecmap(HyperParams.n_1, HyperParams.n_2, HyperParams.n_3, HyperParams.infeature, core_dim=HyperParams.r_1+HyperParams.r_2+HyperParams.r_3)
        self.vecmap = modules.vecmap(HyperParams.n_1, HyperParams.n_2, HyperParams.n_3, HyperParams.infeature)
        self.TDN = modules.TDN(HyperParams.r_1, HyperParams.r_2, HyperParams.r_3, HyperParams.mid_channel)
        # self.MLPmapping = modules.MLPmapping(HyperParams.n_1*HyperParams.n_2, HyperParams.num_nodes, 2*HyperParams.num_nodes)
        self.act_map = get_activation(HyperParams.map_act)
        self.layer_vec = HyperParams.layer_vec
        self.steps = len(self.layer_vec) - 1
        self.num_nodes = HyperParams.num_nodes
        self.snaps_num = HyperParams.n_3
        self.maptovec = nn.ModuleList()
        for k in range(self.steps):
            self.maptovec.append(nn.Linear(self.layer_vec[k], self.layer_vec[k+1]))

    def solo_encoder(self, data):
        x = self.encoder(data)
        return x

    def sp_encoder(self, data):
        x = self.spencoder(data)
        return x

    def solo_decoder(self, x, data):
        x = self.decoder(x, data)
        return x

    def TDN_decoder(self, centre, x):
        U_i, V_i, W_i = self.vecmap(x)
        centre, U, V, W, core = self.TDN(centre, U_i, V_i, W_i)
        return centre, U, V, W, core

    def mapping(self, x):
        idx = 0
        for layer in self.maptovec:
            if (idx==self.steps): x = layer(x)
            else: x = self.act_map(layer(x))
            idx += 1
        return x

    def forward(self, centre, data, parameters, spdata):
        z = self.solo_encoder(data)
        z_sp = self.sp_encoder(spdata)
        z_estimation = self.mapping(parameters)
        centre, U_out, V_out, W_out, core_out = self.TDN_decoder(centre, z)
        return centre, z, z_estimation, U_out, V_out, W_out, core_out, z_sp
