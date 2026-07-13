"""PiTDN-ROM: Physics-informed Tensor Decomposition Network for Reduced Order Modeling.

A hybrid framework combining graph autoencoders with tensor decomposition
for parametric reduced-order modeling of spatio-temporal physical systems.
"""

from PiTDN import modules
from PiTDN import scaling
from PiTDN import loss_func
from PiTDN import network
from PiTDN import training
from PiTDN import testing
from PiTDN import preprocessing
from PiTDN import resample_func
from PiTDN import initialization
from PiTDN import loader
from PiTDN import error
from PiTDN import plotting
from PiTDN import pde
from PiTDN import paraset
from PiTDN import utils
from PiTDN import submesh
from PiTDN import mapneural

__all__ = [
    'modules',
    'scaling',
    'loss_func',
    'network',
    'training',
    'testing',
    'preprocessing',
    'resample_func',
    'initialization',
    'loader',
    'error',
    'plotting',
    'pde',
    'paraset',
    'utils',
    'submesh',
    'mapneural',
]
