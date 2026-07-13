import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from PiTDN import scaling
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from scipy.spatial import ConvexHull
import scipy


# Get the deduplicated set of edges
def get_edges(tri):
    edges = set()

    # Iterate through all triangles (each triangle has 3 vertices, forming 3 edges)
    for simplex in tri.simplices:
        # Get the three vertices of the triangle
        v1, v2, v3 = simplex
        # Deduplicate each edge
        edges.add(tuple(sorted([v1, v2])))  # sorted ensures consistent edge ordering
        edges.add(tuple(sorted([v2, v3])))
        edges.add(tuple(sorted([v3, v1])))

    # Convert the edge set to a NumPy array
    edges_np = np.array(list(edges), dtype=int)
    return edges_np


def graphs_dataset(dataset, HyperParams, param_sample=None):
    """
    graphs_dataset: function to process and scale the input dataset for graph autoencoder model.

    Inputs:
    dataset: an object containing the dataset to be processed.
    HyperParams: an object containing the hyperparameters of the graph autoencoder model.

    Outputs:
    dataset_graph: an object containing the processed and scaled dataset.
    loader: a DataLoader object of the processed and scaled dataset.
    train_loader: a DataLoader object of the training set.
    test_loader: a DataLoader object of the test set.
    val_loader: a DataLoader object of the validation set.
    scaler_all: a scaler object to scale the entire dataset.
    scaler_test: a scaler object to scale the test set.
    xyz: a list containing arrays of the x, y and z-coordinates of the nodes.
    var: an array of the node features.
    VAR_all: an array of the scaled node features of the entire dataset.
    VAR_test: an array of the scaled node features of the test set.
    train_snapshots: a list of indices of the training set.
    test_snapshots: a list of indices of the test set.
    """

    xx = dataset.xx
    yy = dataset.yy
    xyz = [xx, yy]
    if dataset.dim == 3:
       zz = dataset.zz
       xyz.append(zz)
    if HyperParams.comp == 1:
        var = dataset.U
    else:
        var1 = dataset.VX
        var2 = dataset.VY
        var = torch.stack((dataset.VX, dataset.VY), dim=2)

    # PROCESSING DATASET
    num_nodes = var.shape[0]
    num_graphs = var.shape[1]

    print("Number of nodes processed: ", num_nodes)
    print("Number of graphs processed: ", num_graphs)
    rate = HyperParams.rate/100
    total_sims = int(num_graphs)

    if param_sample is None:
        snapshots_num = HyperParams.snapshots_num
        cand_list = [
            list(range(i, i + snapshots_num))
            for i in range(0, total_sims - snapshots_num + 1, snapshots_num)  # Step size set to snapshots_num
        ]
        np.random.shuffle(cand_list)
        sample_num = HyperParams.sample_num
        train_sims = int(rate * sample_num)
        res_sims = sample_num - train_sims
        val_sims = int(0.5 * res_sims)
        test_sims = res_sims - val_sims

        indices = np.arange(sample_num)
        train_indices = sorted(indices[:train_sims].tolist())
        val_indices = sorted(indices[train_sims:(train_sims+val_sims)].tolist())
        test_indices = sorted(indices[(train_sims+val_sims):].tolist())

        train_snapshots = [cand_list[i] for i in train_indices]
        val_snapshots = [cand_list[i] for i in val_indices]
        test_snapshots = [cand_list[i] for i in test_indices]

        val_snapshots.sort()
        train_snapshots.sort()
        test_snapshots.sort()
    else:
        snapshots_num = HyperParams.snapshots_num
        cand_list = [
            list(range(i, i + snapshots_num))
            for i in range(0, param_sample - snapshots_num + 1, snapshots_num)  # Step size set to snapshots_num
        ]
        np.random.shuffle(cand_list)
        sample_num = HyperParams.sample_num
        train_sims = int(rate * sample_num)
        res_sims = sample_num - train_sims
        val_sims = int(0.5 * res_sims)
        test_sims = res_sims - val_sims

        indices = np.arange(sample_num)
        train_indices = sorted(indices[:train_sims].tolist())
        val_indices = sorted(indices[train_sims:(train_sims+val_sims)].tolist())
        test_indices = sorted(indices[(train_sims+val_sims):].tolist())

        train_snapshots = [cand_list[i] for i in train_indices]
        val_snapshots = [cand_list[i] for i in val_indices]
        test_snapshots = [cand_list[i] for i in test_indices]

        val_snapshots.sort()
        train_snapshots.sort()
        test_snapshots.sort()

    ## SCALING
    scaling_type = HyperParams.scaling_type
    if HyperParams.comp == 1:
        var_test = dataset.U[:, test_snapshots]
        var_test_reshaped = var_test.reshape(var_test.shape[0], -1)
        var_val = dataset.U[:, val_snapshots]
        var_val_reshaped = var_val.reshape(var_val.shape[0], -1)

        scaler_all, VAR_all = scaling.tensor_scaling(var, scaling_type, HyperParams.scaler_number)
        # Additional outputs: VAR_all_reshaped, scaler_all_reshaped
        VAR_cand = var.T[cand_list,:]
        VAR_cand = VAR_cand.reshape(-1, VAR_cand.shape[2]).T
        scaler_all_reshaped, VAR_all_reshaped = scaling.tensor_scaling(VAR_cand, scaling_type, HyperParams.scaler_number)

        scaler_test, VAR_test_reshaped = scaling.tensor_scaling(var_test_reshaped, scaling_type, HyperParams.scaler_number)
        VAR_test = VAR_test_reshaped.reshape(len(test_snapshots), snapshots_num, VAR_test_reshaped.shape[1],VAR_test_reshaped.shape[2])
        scaler_val, VAR_val_reshaped = scaling.tensor_scaling(var_val_reshaped, scaling_type,
                                                                HyperParams.scaler_number)
        VAR_val = VAR_val_reshaped.reshape(len(val_snapshots), snapshots_num, VAR_val_reshaped.shape[1],
                                             VAR_val_reshaped.shape[2])

    else:
        var1_test = var1[:, test_snapshots]
        var1_test_reshaped = var1_test.reshape(var1_test.shape[0], -1)
        var2_test = var2[:, test_snapshots]
        var2_test_reshaped = var2_test.reshape(var2_test.shape[0], -1)
        scaler_var1_all, VAR1_all = scaling.tensor_scaling(var1, scaling_type, HyperParams.scaler_number)
        scaler_var1_test, VAR1_test_reshaped = scaling.tensor_scaling(var1_test_reshaped, scaling_type, HyperParams.scaler_number)
        scaler_var2_all, VAR2_all = scaling.tensor_scaling(var2, scaling_type, HyperParams.scaler_number)
        scaler_var2_test, VAR2_test_reshaped = scaling.tensor_scaling(var2_test_reshaped, scaling_type, HyperParams.scaler_number)
        VAR1_test = VAR1_test_reshaped.reshape(len(test_snapshots), snapshots_num, VAR1_test_reshaped.shape[1],VAR1_test_reshaped.shape[2])
        VAR2_test = VAR2_test_reshaped.reshape(len(test_snapshots), snapshots_num, VAR2_test_reshaped.shape[1],VAR2_test_reshaped.shape[2])
        VAR_all = torch.cat((VAR1_all, VAR2_all), dim=2)
        VAR_test = torch.cat((VAR1_test, VAR2_test), dim=3)
        scaler_all = [scaler_var1_all, scaler_var2_all]
        scaler_test = [scaler_var1_test, scaler_var2_test]

    graphs = []
    edge_index = torch.t(dataset.E) - 1
    for graph in range(num_graphs):
        if dataset.dim == 2:
            pos = torch.cat((xx, yy), 1)
        elif dataset.dim == 3:
            pos = torch.cat((xx, yy, zz), 1)
        ei = torch.index_select(pos, 0, edge_index[0, :])
        ej = torch.index_select(pos, 0, edge_index[1, :])
        edge_attr = torch.abs(ej - ei)
        if dataset.dim == 2:
            edge_weight = torch.sqrt(torch.pow(edge_attr[:, 0], 2) + torch.pow(edge_attr[:, 1], 2)).unsqueeze(1)
        elif dataset.dim == 3:
            edge_weight = torch.sqrt(torch.pow(edge_attr[:, 0], 2) + torch.pow(edge_attr[:, 1], 2) + torch.pow(edge_attr[:, 2], 2)).unsqueeze(1)
        if HyperParams.comp == 1:
            node_features = VAR_all[graph, :]
        else:
            node_features = VAR_all[graph, :, :]
        dataset_graph = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr, pos=pos)
        graphs.append(dataset_graph)

    if HyperParams.ifsparse:
        np.random.seed(HyperParams.seed)
        num_samples = HyperParams.num_samples
        sample_indices = np.random.choice(num_nodes, num_samples, replace=False)
        tri_xx = xx[sample_indices, :]
        tri_yy = yy[sample_indices, :]
        sampled_points = np.vstack([tri_xx[:,0], tri_yy[:,0]]).T
        tri = Delaunay(sampled_points)
        edge = torch.tensor(get_edges(tri))
        edge_index = torch.t(edge)
        subgraphs = []
        for graph in range(num_graphs):
            if dataset.dim == 2:
                sub_pos = torch.cat((tri_xx, tri_yy), 1)
            elif dataset.dim == 3:
                sub_pos = torch.cat((tri_xx, tri_yy, zz), 1)
            ei = torch.index_select(sub_pos, 0, edge_index[0, :])
            ej = torch.index_select(sub_pos, 0, edge_index[1, :])
            edge_attr = torch.abs(ej - ei)
            if dataset.dim == 2:
                edge_weight = torch.sqrt(torch.pow(edge_attr[:, 0], 2) + torch.pow(edge_attr[:, 1], 2)).unsqueeze(1)
            elif dataset.dim == 3:
                edge_weight = torch.sqrt(
                    torch.pow(edge_attr[:, 0], 2) + torch.pow(edge_attr[:, 1], 2) + torch.pow(edge_attr[:, 2],
                                                                                              2)).unsqueeze(1)
            if HyperParams.comp == 1:
                node_features = VAR_all[graph, sample_indices]
            else:
                node_features = VAR_all[graph, sample_indices, sample_indices]
            sub_graph = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr,
                                 pos=sub_pos)
            subgraphs.append(sub_graph)

    HyperParams.num_nodes = dataset_graph.num_nodes
    train_dataset = [graphs[i] for sublist in train_snapshots for i in sublist]
    val_dataset = [graphs[i] for sublist in val_snapshots for i in sublist]
    test_dataset = [graphs[i] for sublist in test_snapshots for i in sublist]
    all_dataset = [graphs[i] for sublist in cand_list for i in sublist]
    sub_train_dataset = [subgraphs[i] for sublist in train_snapshots for i in sublist]
    sub_val_dataset = [subgraphs[i] for sublist in val_snapshots for i in sublist]
    sub_test_dataset = [subgraphs[i] for sublist in test_snapshots for i in sublist]
    sub_all_dataset = [subgraphs[i] for sublist in cand_list for i in sublist]

    print("Length of train dataset: ", len(train_dataset))
    print("Length of val dataset: ", len(val_dataset))
    print("Length of test dataset: ", len(test_dataset))

    loader = DataLoader(all_dataset, batch_size=snapshots_num)
    train_loader = DataLoader(train_dataset, batch_size=snapshots_num if train_sims<HyperParams.batch_size else HyperParams.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=snapshots_num if test_sims<HyperParams.batch_size else HyperParams.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=snapshots_num if test_sims<HyperParams.batch_size else HyperParams.batch_size, shuffle=False)

    sub_loader = DataLoader(sub_all_dataset, batch_size=snapshots_num)
    sub_train_loader = DataLoader(sub_train_dataset,
                              batch_size=snapshots_num if train_sims < HyperParams.batch_size else HyperParams.batch_size,
                              shuffle=False)
    sub_test_loader = DataLoader(sub_test_dataset,
                             batch_size=snapshots_num if test_sims < HyperParams.batch_size else HyperParams.batch_size,
                             shuffle=False)
    sub_val_loader = DataLoader(sub_val_dataset,
                             batch_size=snapshots_num if test_sims < HyperParams.batch_size else HyperParams.batch_size,
                             shuffle=False)

    return loader, train_loader, test_loader, \
                val_loader, scaler_all, scaler_test, xyz, VAR_all, VAR_test, \
                    train_snapshots, test_snapshots, sub_loader, sub_train_loader, sub_test_loader, \
                        sub_val_loader, scaler_all_reshaped, VAR_all_reshaped, cand_list, scaler_val, VAR_val, val_snapshots


def delete_initial_condition(dataset, params, mu_space, n_comp, n_snap_time):
    params = params[params[:, -1] != 0.]
    mu_space[-1] = np.delete(mu_space[-1], 0)
    if n_comp == 1:
        indices = torch.ones(dataset.U.shape[1], dtype=torch.bool)
        indices[::n_snap_time] = 0
        dataset.U = dataset.U[:, indices]
    elif n_comp == 2:
        indices = torch.ones(dataset.VX.shape[1], dtype=torch.bool)
        indices[::n_snap_time] = 0
        dataset.VX = dataset.VX[:, indices]
        dataset.VY = dataset.VY[:, indices]
    else:
        print("Invalid dimension. Please enter 1 or 2.")

    dataset.xx = dataset.xx[:, indices]
    dataset.yy = dataset.yy[:, indices]
    return dataset, params, mu_space


def shrink_dataset(dataset, mu_space, n_sim, n_snap2keep, n_comp):
    time = mu_space[-1]
    n_time = len(time)
    idx_time = np.round(np.linspace(0, n_time-1, n_snap2keep)).astype(int)
    mu_space[-1] = time[idx_time]

    idx = np.copy(idx_time)
    for i in range(1, n_sim):
        idx_time += n_time
        idx = np.hstack((idx, idx_time))

    if n_comp == 1:
        dataset.U = dataset.U[:, idx]
    elif n_comp == 2:
        dataset.VX = dataset.VX[:, idx]
        dataset.VY = dataset.VY[:, idx]
    dataset.xx = dataset.xx[:, idx]
    dataset.yy = dataset.yy[:, idx]

    return dataset, mu_space


def mask_gen(HyperParams):
    mask_dataset_dir = HyperParams.mask_dataset_dir
    data_mat = scipy.io.loadmat(mask_dataset_dir)
    mask = data_mat['mask']
    inmask, outmask = ~mask, mask
    mask_result = list()
    mask_result.append(inmask)
    mask_result.append(outmask)
    mask_result.append(data_mat['grid_points'])
    mask_result.append(data_mat['X'])
    mask_result.append(data_mat['Y'])
    return mask_result
