"""Dataset loader for graph-structured PDE simulation data.

Provides a PyTorch Geometric Dataset wrapper for loading .mat files
containing unstructured mesh data (node coordinates, connectivity,
and solution fields).
"""

import torch
import scipy
from torch_geometric.data import Dataset


class LoadDataset(Dataset):
    """Custom Dataset for loading graph-structured simulation data from .mat files.

    Attributes:
        data_mat: Raw data loaded via scipy.io.loadmat.
        U: Solution field tensor (shape: [num_nodes, num_snapshots]).
        xx, yy, zz: Node coordinate tensors.
        T: Adjacency/triangle connectivity matrix.
        E: Edge connection matrix.
        dim: Spatial dimensionality (2 or 3).
        n_comp: Number of field components (1 or 2).
    """

    def __init__(self, root_dir, variable, dim_pde, n_comp):
        """Load simulation data from a .mat file.

        Args:
            root_dir: Path to the .mat file.
            variable: Variable name to load (e.g., 'U', 'VX', 'VY').
            dim_pde: Spatial dimensionality (2 or 3).
            n_comp: Number of solution components (1=scalar, 2=vector).
        """
        self.data_mat = scipy.io.loadmat(root_dir)
        self.dim = dim_pde
        self.n_comp = n_comp
        self.xx = torch.tensor(self.data_mat['xx'])
        self.yy = torch.tensor(self.data_mat['yy'])
        self.T = torch.tensor(self.data_mat['T'].astype(int))
        self.E = torch.tensor(self.data_mat['E'].astype(int))

        if self.n_comp == 1:
            self.U = torch.tensor(self.data_mat[variable])
        elif self.n_comp == 2:
            self.VX = torch.tensor(self.data_mat['VX'])
            self.VY = torch.tensor(self.data_mat['VY'])

        if self.dim == 3:
            self.zz = torch.tensor(self.data_mat['zz'])

    def len(self):
        pass

    def get(self):
        pass
