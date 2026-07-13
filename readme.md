# PiTDN-ROM: Physics-informed Tensor Decomposition Network for Reduced Order Modeling

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.3+-green.svg)](https://pyg.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

PiTDN-ROM combines **graph autoencoders** for compressing unstructured field data with **tensor decomposition networks** for structured, interpretable low-rank representations.

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric 2.3+

### Setup

```bash
git clone https://github.com/yourusername/PiTDN-ROM.git
cd PiTDN-ROM
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
torch-geometric>=2.3.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
tqdm>=4.65.0
shapely>=2.0.0
h5py>=3.8.0
```

## Usage

### Data Preparation

Simulation data should be stored in MATLAB `.mat` format. For example, the cylinder dataset should be stored in this format:

```
dataset/
├── cylinder_unstructured.mat    # Unstructured mesh data
├── cylinder_structured.mat      # Structured grid data
└── cylinder_mask.mat            # Geometry mask
```

Each `.mat` file should contain:
- `xx`, `yy`: Node coordinates (unstructured)
- `T`: Triangle connectivity matrix
- `E`: Edge connectivity matrix
- `U` (or `VX`, `VY`): Solution field(s)
- `X`, `Y`: Structured grid coordinates
- `grid_points`: Structured grid point coordinates
- `mask`: Domain geometry mask

### Training

```python
from PiTDN import network, training, preprocessing, initialization
from PiTDN import loader, pde, paraset
from para.ManuPara_cylinder import ParaSet

# Select problem and hyperparameters
problem_name, variable, mu_space, n_param, dim_pde, n_comp = pde.problem(1)
argv = paraset.hyperparameters_selection(
    problem_name, variable, n_param, n_comp)
hyperparams = network.HyperParams(argv)
hyperparams = ParaSet(hyperparams, problem_name)

# Setup
device = initialization.set_device()
initialization.set_reproducibility(hyperparams)
initialization.set_path(hyperparams)

# Load and preprocess data
dataset = loader.LoadDataset(
    hyperparams.dataset_dir, variable, dim_pde, n_comp)
(loader, train_loader, test_loader, val_loader,
 scaler_all, scaler_test, xyz, VAR_all, VAR_test,
 train_trajectories, test_trajectories,
 sub_loader, sub_train_loader, sub_test_loader,
 sub_val_loader, scaler_all_reshaped, VAR_all_reshaped,
 cand_list, scaler_val, VAR_val, val_snapshots
) = preprocessing.graphs_dataset(dataset, hyperparams)

# Initialize and train model
model = network.Net(hyperparams).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.lr_real)
core_out = training.train(
    model, optimizer, device, scheduler, params,
    train_loader, test_loader,
    train_trajectories, test_trajectories,
    hyperparams, sub_train_loader, sub_test_loader,
    mask_result, scaler_all_reshaped, interpolator, centre)
```

### Evaluation

```python
from PiTDN import testing
latents_map, latents_encoder, unstruct_results, \
    struct_results, modes_results = testing.evaluate(
        VAR_all, model, loader, params, hyperparams,
        test_trajectories, interpolator, core_out)
```

## Citation

If you use PiTDN-ROM in your research, please cite:

```bibtex
@article{zhong2026inherently,
  title={Inherently interpretable physic-informed tensor decomposition network for reduced order modeling},
  author={Zhong, Zhirong and Ren, Hongfei and Zhai, Zhi and Ma, Meng and Liu, Jinxin},
  journal={Journal of Computational Physics},
  year = {2026},
  issn = {0021-9991},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
