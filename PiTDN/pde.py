"""Problem definitions and parameter spaces for supported PDE cases.

Each case defines the problem name, variable, parameter ranges, and
dimensionality for the reduced-order modeling pipeline.

Supported cases:
  1 - Cylinder (Navier-Stokes, Re parameter)
  2 - Airfoil (RD parameter)
  3 - SST (year/week parameters)
  4 - Bifurcating flow (eta/T parameters)
  5 - 2D channel cross-section (time parameter only)
"""

import numpy as np


def problem(argument):
    """Return problem configuration for the specified case number.

    Args:
        argument: Integer case selector (1-5).

    Returns:
        Tuple of (problem_name, variable, mu_space, n_param, dim_pde, n_comp).
    """
    match argument:
        case 1:
            problem_name = "cylinder"
            variable = 'U'
            mu1 = np.arange(60, 201, 5)        # Reynolds number
            dt = 0.1
            mu2 = np.arange(0.1, 30.51, dt)
            mu_space = [mu1, mu2]
            n_param = 2
            dim_pde = 2
            n_comp = 1
        case 2:
            problem_name = "airfoil"
            variable = 'U'
            mu1 = np.arange(0, 30.5, 1)         # RD number
            dt = 0.1
            mu2 = np.arange(0, 11.85, dt)
            mu_space = [mu1, mu2]
            n_param = 2
            dim_pde = 2
            n_comp = 1
        case 3:
            problem_name = "sst"
            variable = 'U'
            mu1 = np.arange(0, 32.5, 1)         # year
            dt = 1
            mu2 = np.arange(0, 51.5, dt)         # weeks
            mu_space = [mu1, mu2]
            n_param = 2
            dim_pde = 2
            n_comp = 1
        case 4:
            problem_name = "bifurcating"
            variable = 'U'
            mu1 = np.arange(0.4, 0.61, 0.02)     # eta
            dt = 0.001
            mu2 = np.arange(0.031, 0.0995 + dt, dt)  # T
            mu_space = [mu1, mu2]
            n_param = 2
            dim_pde = 2
            n_comp = 1
        case 5:
            problem_name = "ch2Dxysec"
            variable = 'U'
            dt = 0.001
            mu2 = np.arange(0, 9.985 + dt, dt)
            mu_space = [mu2]
            n_param = 1
            dim_pde = 2
            n_comp = 1

    return problem_name, variable, mu_space, n_param, dim_pde, n_comp
