"""
Geometry mask generation utility.

This module provides functionality to generate 2D masks defined by a convex
hull boundary with optional circular cutouts. It loads unstructured point
data from a MATLAB .mat file, computes the convex hull of those points to
define a boundary polygon, then evaluates a regular grid against that polygon
(optionally excluding points inside a circular region) to produce boolean masks.
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from scipy.spatial import ConvexHull, Delaunay
from typing import Optional, Tuple


def generate_mask(
    dataset_path: str,
    grid_x_range: Tuple[float, float] = (0, 4),
    grid_y_range: Tuple[float, float] = (0, 2 * np.sqrt(3)),
    grid_nx: int = 100,
    grid_ny: int = 90,
    circle_center: Optional[Tuple[float, float]] = None,
    circle_radius: float = 0.0,
    variable_name_xx: str = 'xx',
    variable_name_yy: str = 'yy',
    visualize: bool = False,
    output_path: Optional[str] = None,
) -> dict:
    """
    Generate a boolean mask defined by a convex hull boundary and optional circular hole.

    The function loads unstructured point data from a .mat file, computes
    their convex hull to define a boundary polygon, samples a regular 2D grid,
    and returns masks for points inside the boundary (and optionally outside
    a circular cutout).

    Parameters
    ----------
    dataset_path : str
        Path to the .mat file containing unstructured point coordinates.
    grid_x_range : tuple of float, optional
        (min, max) for the x-axis of the uniform grid. Default (0, 4).
    grid_y_range : tuple of float, optional
        (min, max) for the y-axis of the uniform grid. Default (0, 2*sqrt(3)).
    grid_nx : int, optional
        Number of grid points along the x-axis. Default 100.
    grid_ny : int, optional
        Number of grid points along the y-axis. Default 90.
    circle_center : tuple of float or None, optional
        (x, y) center of the circular hole. If None, no hole is applied.
    circle_radius : float, optional
        Radius of the circular hole. Ignored if circle_center is None.
    variable_name_xx : str, optional
        Variable name for x-coordinates in the .mat file. Default 'xx'.
    variable_name_yy : str, optional
        Variable name for y-coordinates in the .mat file. Default 'yy'.
    visualize : bool, optional
        If True, create and save visualizations of the mask. Default False.
    output_path : str or None, optional
        Directory to save output figures. Required if visualize is True.

    Returns
    -------
    dict
        Dictionary containing:
        - 'inmask' : ndarray of bool, shape (grid_ny, grid_nx)
            True for points inside the boundary (and outside the hole if applicable).
        - 'outmask' : ndarray of bool, shape (grid_ny, grid_nx)
            Logical NOT of inmask.
        - 'X' : ndarray, shape (grid_ny, grid_nx)
            X-coordinates of the 2D grid.
        - 'Y' : ndarray, shape (grid_ny, grid_nx)
            Y-coordinates of the 2D grid.
        - 'grid_points' : ndarray, shape (grid_ny * grid_nx, 2)
            Flattened grid point coordinates.
        - 'boundary_points' : ndarray, shape (n_vertices, 2)
            Vertices of the convex hull boundary polygon.
        - 'tri_xx' : ndarray
            Original unstructured x-coordinates from the .mat file.
        - 'tri_yy' : ndarray
            Original unstructured y-coordinates from the .mat file.

    Raises
    ------
    FileNotFoundError
        If dataset_path does not exist.
    KeyError
        If variable_name_xx or variable_name_yy is not found in the .mat file.
    """
    # ------------------------------------------------------------------
    # 1. Generate the uniform grid
    # ------------------------------------------------------------------
    x = np.linspace(grid_x_range[0], grid_x_range[1], grid_nx)
    y = np.linspace(grid_y_range[0], grid_y_range[1], grid_ny)
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack((X.ravel(), Y.ravel()))

    # ------------------------------------------------------------------
    # 2. Load unstructured point data from .mat file
    # ------------------------------------------------------------------
    data_mat = scipy.io.loadmat(dataset_path)
    tri_xx = data_mat[variable_name_xx][:, 0]
    tri_yy = data_mat[variable_name_yy][:, 0]
    unstruc_points = np.column_stack((tri_xx, tri_yy))

    # ------------------------------------------------------------------
    # 3. Compute convex hull boundary polygon
    # ------------------------------------------------------------------
    hull = ConvexHull(unstruc_points)
    boundary_points = unstruc_points[hull.vertices]
    polygon = Polygon(boundary_points)

    # ------------------------------------------------------------------
    # 4. Determine mask: inside boundary, optionally excluding a hole
    # ------------------------------------------------------------------
    in_polygon = np.array([
        polygon.contains(Point(p[0], p[1])) for p in grid_points
    ])

    if circle_center is not None and circle_radius > 0:
        in_circle = (
            (grid_points[:, 0] - circle_center[0])**2
            + (grid_points[:, 1] - circle_center[1])**2
            <= circle_radius**2
        )
        final_mask = in_polygon & ~in_circle
    else:
        final_mask = in_polygon

    inmask = final_mask.reshape(X.shape)
    outmask = ~final_mask.reshape(X.shape)

    # ------------------------------------------------------------------
    # 5. Visualization (optional)
    # ------------------------------------------------------------------
    if visualize:
        if output_path is None:
            raise ValueError("output_path is required when visualize=True")

        _plot_mask(grid_points, final_mask, boundary_points, tri_xx, tri_yy,
                   X, Y, inmask, output_path)

    return {
        'inmask': inmask,
        'outmask': outmask,
        'X': X,
        'Y': Y,
        'grid_points': grid_points,
        'boundary_points': boundary_points,
        'tri_xx': tri_xx,
        'tri_yy': tri_yy,
    }


def _plot_mask(grid_points, final_mask, boundary_points, tri_xx, tri_yy,
               X, Y, inmask, output_path):
    """Create and save visualizations of the mask classification and a filled contour."""
    # Figure 1: scatter plot of mask classification with triangulation overlay
    plt.figure()
    plt.scatter(grid_points[final_mask, 0], grid_points[final_mask, 1],
                color='g', label='Inside Boundary', s=10)
    plt.scatter(grid_points[~final_mask, 0], grid_points[~final_mask, 1],
                color='r', label='Outside Boundary', s=10)
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'b-', linewidth=2)

    triang = Delaunay(np.column_stack((tri_xx, tri_yy)))
    plt.triplot(tri_xx.flatten(), tri_yy.flatten(), triang.simplices,
                color='black', alpha=0.3)

    plt.axis('equal')
    plt.title('Points inside and outside the boundary')
    plt.legend()
    plt.savefig(f'{output_path}/mask_boundary.png', bbox_inches='tight', dpi=500)
    plt.close()

    # Figure 2: filled contour of a random field over the masked region
    mod = np.zeros(X.size)
    mod[inmask.ravel()] = np.random.rand(int(inmask.sum()))
    mod = mod.reshape(X.shape)

    plt.figure(figsize=(30, 20))
    plt.contourf(X, Y, mod, 20, cmap='twilight')
    plt.axis('equal')
    plt.savefig(f'{output_path}/mask_contour.png', bbox_inches='tight', dpi=500)
    plt.close()


if __name__ == '__main__':
    # Example usage (adjust dataset_path to point to a valid .mat file)
    result = generate_mask(
        dataset_path='poisson_data.mat',
        circle_center=(2, np.sqrt(3)),
        circle_radius=0.5,
        visualize=True,
        output_path='.',
    )
    print(f"Mask shape: {result['inmask'].shape}")
    print(f"Inside points: {result['inmask'].sum()}")
    print(f"Outside points: {result['outmask'].sum()}")
