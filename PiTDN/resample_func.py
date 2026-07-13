"""Mesh resampling utilities for structured/unstructured grid conversion.

Provides interpolation between unstructured triangular meshes and structured
Cartesian grids, including Natural Neighbor Region (NNR) based methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
from scipy.interpolate import RegularGridInterpolator
import scipy
import torch
import pickle
import os


def resample(out, dataset_dir, new_D, new_H, method='linear'):
    """Resample structured grid tensor to unstructured mesh points.

    Args:
        out: Structured grid data of shape (D, H, W).
        dataset_dir: Path to .mat file with unstructured mesh coordinates.
        new_D: Target D dimension (grid height).
        new_H: Target H dimension (grid width).
        method: Interpolation method ('linear', 'nearest', 'cubic').

    Returns:
        Resampled data array of shape (num_nodes, W).
    """
    D, H, W = out.shape
    data_mat = scipy.io.loadmat(dataset_dir)
    tri_xx = data_mat['xx']
    tri_yy = data_mat['yy']
    resampled_tensor = np.zeros((len(tri_yy), W))
    x_new = np.linspace(0, new_D, H)
    y_new = np.linspace(0, new_H, D)
    for w in range(W):
        tensor_flat = out[:, :, w].detach().cpu().numpy()
        interpolator_tri = RegularGridInterpolator(
            (x_new, y_new), tensor_flat.T, method='linear')
        xx = tri_xx[:, w]
        yy = tri_yy[:, w]
        tri_values = interpolator_tri((xx, yy))
        resampled_tensor[:, w] = tri_values
    return resampled_tensor


def triangular_to_uniform(tri_data, out, dataset_dir, new_D, new_H,
                          scale_train):
    """Convert unstructured triangular mesh data to uniform Cartesian grid.

    Args:
        tri_data: Unstructured triangular mesh data.
        out: Reference structured grid of shape (D, H, W).
        dataset_dir: Path to .mat file with mesh info.
        new_D: Target D dimension.
        new_H: Target H dimension.
        scale_train: [mean, std] scaling factors for fill value computation.

    Returns:
        Uniform grid data array of shape (D, H, W).
    """
    D, H, W = out.shape
    data_mat = scipy.io.loadmat(dataset_dir)
    tri_xx = data_mat['xx'][:, 0]
    tri_yy = data_mat['yy'][:, 0]
    triangles = data_mat['T'] - 1
    tri = Triangulation(tri_xx, tri_yy, triangles)
    x_uniform = np.linspace(0, new_D, H)
    y_uniform = np.linspace(0, new_H, D)
    grid_x, grid_y = np.meshgrid(x_uniform, y_uniform)
    uniform_tensor = np.zeros((D, H, W))
    num_nodes = len(tri_xx)
    mean = scale_train[0]
    std = scale_train[1]
    for w in range(W):
        tri_values = tri_data[
            (w * num_nodes):((w + 1) * num_nodes)]
        tri_values = tri_values.detach().cpu().numpy().squeeze(-1)
        interpolator = LinearTriInterpolator(tri, tri_values)
        grid_z = interpolator(grid_x, grid_y)
        grid_z = np.array(grid_z)
        fill_value = (0 - mean[w]) / std[w]
        grid_z.fill(fill_value)
        uniform_tensor[:, :, w] = grid_z
    return uniform_tensor


class NNRInterpolator:
    """Natural Neighbor Region (NNR) interpolator with topology caching.

    Pre-computes the mapping between structured grid points and unstructured
    mesh nodes for efficient batch interpolation during training.

    Args:
        mat_path: Path to .mat file with mesh and grid data.
        cache_path: Path to cache pre-computed topology (.pkl).
        device: Compute device ('cuda' or 'cpu').
        K_min: Minimum number of neighbors for interpolation fallback.
    """

    def __init__(self, mat_path, cache_path=None, device='cuda', K_min=4):
        self.device = torch.device(device)
        self.K_min = K_min
        self.cache_path = cache_path

        mat = scipy.io.loadmat(mat_path)
        xx = mat['xx'].reshape(-1)
        yy = mat['yy'].reshape(-1)
        T = mat['T'].astype(int) - 1
        self.grid_coords = torch.tensor(
            mat['grid_points'], dtype=torch.float32, device=self.device)
        self.unstructured_nodes = torch.stack([
            torch.tensor(xx, dtype=torch.float32),
            torch.tensor(yy, dtype=torch.float32)], dim=1).to(self.device)
        self.unstructured_cells = torch.tensor(
            T, dtype=torch.long, device=self.device)

        if cache_path and os.path.exists(cache_path):
            self.nnr_point_indices = self._load_topology(cache_path)
        else:
            self.nnr_point_indices = self._build_topology()
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                self._save_topology(cache_path)

    def _point_in_polygon(self, polygon, points):
        """Ray-casting algorithm for point-in-polygon test.

        Args:
            polygon: Polygon vertices of shape (K, 2).
            points: Query points of shape (M, 2).

        Returns:
            Boolean tensor of shape (M,) indicating inside/outside.
        """
        num_vertices = polygon.shape[0]
        x, y = points[:, 0], points[:, 1]
        poly_x = polygon[:, 0]
        poly_y = polygon[:, 1]
        inside = torch.zeros(
            points.shape[0], dtype=torch.bool, device=points.device)
        j = num_vertices - 1
        for i in range(num_vertices):
            xi, yi = poly_x[i], poly_y[i]
            xj, yj = poly_x[j], poly_y[j]
            intersect = ((yi > y) != (yj > y)) & (
                x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
            inside ^= intersect
            j = i
        return inside

    def _build_topology(self):
        """Build NNR topology mapping for all unstructured nodes.

        Returns:
            List of index tensors, one per unstructured node.
        """
        N = self.unstructured_nodes.shape[0]
        C = self.unstructured_cells.shape[0]
        node2cells = [[] for _ in range(N)]
        for ci in range(C):
            for v in self.unstructured_cells[ci]:
                node2cells[v.item()].append(ci)

        indices = []
        for vi in range(N):
            v = self.unstructured_nodes[vi]
            rows = torch.tensor(
                node2cells[vi], dtype=torch.long, device=self.device)
            connected = self.unstructured_cells[rows]
            centroids = torch.mean(
                self.unstructured_nodes[connected], dim=1)
            neighbors = torch.unique(connected.flatten())
            neighbors = neighbors[neighbors != vi]
            edge_midpoints = (self.unstructured_nodes[neighbors] + v) * 0.5
            nnr_pts = torch.cat([centroids, edge_midpoints], dim=0)
            center = nnr_pts.mean(dim=0)
            rel = nnr_pts - center
            angles = torch.atan2(rel[:, 1], rel[:, 0])
            order = torch.argsort(angles)
            polygon = nnr_pts[order]
            mask = self._point_in_polygon(polygon, self.grid_coords)
            idx = torch.where(mask)[0]
            indices.append(idx)
        return indices

    def _save_topology(self, path):
        """Save pre-computed topology to a pickle file."""
        cpu_indices = [x.cpu().numpy() for x in self.nnr_point_indices]
        with open(path, 'wb') as f:
            pickle.dump(cpu_indices, f)

    def _load_topology(self, path):
        """Load pre-computed topology from a pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return [torch.tensor(x, dtype=torch.long, device=self.device)
                for x in data]

    def interpolate(self, X_hat):
        """Interpolate from structured grid to unstructured mesh.

        Args:
            X_hat: Structured grid values of shape (M,).

        Returns:
            Interpolated values at unstructured nodes of shape (N,).
        """
        N = len(self.nnr_point_indices)
        mapped_X = torch.zeros(N, dtype=X_hat.dtype, device=self.device)
        device = X_hat.device
        for vi in range(N):
            v = self.unstructured_nodes[vi]
            idx = self.nnr_point_indices[vi]
            if idx.numel() < self.K_min:
                dists = torch.cdist(v[None, :], self.grid_coords)[0]
                nearest = torch.topk(
                    dists, self.K_min, largest=False).indices
                idx = torch.unique(torch.cat([idx, nearest]))
            matched_pts = self.grid_coords[idx]
            dists = torch.norm(matched_pts - v, dim=1) + 1e-8
            weights = 1.0 / dists
            weights = weights / weights.sum()
            idx = idx.to(device)
            weights = weights.to(device)
            mapped_X[vi] = torch.sum(weights * X_hat[idx])
        return mapped_X

    def batch_interpolate(self, X_tensor):
        """Batch interpolate from structured to unstructured mesh.

        Args:
            X_tensor: Tensor of shape (D, H, W) or (B, C, H, W).

        Returns:
            Interpolated tensor of shape (N, W) or (B, C, N, W).
        """
        shape = X_tensor.shape
        if len(shape) == 3:  # [D, H, W]
            D, H, W = shape
            flat = X_tensor.reshape(D * H, W)
            out = torch.stack(
                [self.interpolate(flat[:, w]) for w in range(W)], dim=1)
        elif len(shape) == 4:  # [B, C, H, W]
            B, C, H, W = shape
            out = torch.zeros(
                B, C, len(self.nnr_point_indices), W,
                device=self.device, dtype=X_tensor.dtype)
            for b in range(B):
                for c in range(C):
                    flat = X_tensor[b, c].reshape(-1, W)
                    for w in range(W):
                        out[b, c, :, w] = self.interpolate(flat[:, w])
        else:
            raise ValueError("Unsupported input shape.")
        return out
