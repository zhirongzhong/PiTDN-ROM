"""Hyperparameter preset configurations for different physical problems.

Each problem (cylinder, airfoil, sst, bifurcating, ch2Dxysec) has a set of
default hyperparameters encoded as indices into preset option arrays.
"""

import numpy as np


def hyperparameters_selection(name, var, n, comp):
    """Select hyperparameter presets for a given problem.

    Args:
        name: Problem name (cylinder, airfoil, sst, bifurcating, ch2Dxysec).
        var: Variable name (e.g., 'U', 'VX', 'VY', 'P').
        n: Number of parameters for the param-to-latent mapping input.
        comp: Number of field components.

    Returns:
        argv: List of hyperparameter values formatted for HyperParams init.
    """

    def default_values(name):
        if name == "cylinder":
            preset = [3, 2, 2, 2, 1, 4, 3, 2]
        elif name == "airfoil":
            preset = [3, 2, 2, 5, 6, 5, 3, 2]
        elif name == "bifurcating":
            preset = [3, 2, 0, 2, 5, 5, 2, 2]
        elif name == "sst":
            preset = [3, 2, 0, 2, 3, 3, 2, 2]
        elif name == "ch2Dxysec":
            preset = [3, 2, 2, 2, 1, 4, 3, 2]
        return preset

    preset = default_values(name)
    preset_options_1 = ["sample", "feature", "feature-sampling",
                        "sampling-feature"]
    preset_options_2 = ["minmax", "robust", "standard"]
    preset_options_3 = ["10", "20", "30", "40", "50"]
    preset_options_4 = ["50", "100", "200", "300", "400", "3000"]
    preset_options_5 = ["25", "50", "75", "100", "125", "2000", "1000"]
    preset_options_6 = ["10", "15", "20", "25", "30", "500"]
    preset_options_7 = ["0.01", "0.1", "1", "10", "100"]
    preset_options_8 = ["1", "2", "3", "4", "5"]
    preset_options_9 = ["1000"]

    argv = [name,
            var,
            preset[0] + 1,
            preset[1] + 1,
            1,
            int(preset_options_3[preset[2]]),
            int(preset_options_4[preset[3]]),
            int(preset_options_5[preset[4]]),
            int(preset_options_6[preset[5]]),
            float(preset_options_7[preset[6]]),
            int(preset_options_8[preset[7]]),
            n,
            int(preset_options_9[0]),
            comp]

    return argv
