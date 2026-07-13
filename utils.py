"""Utility functions for saving and loading evaluation results."""

import os
import pickle


def save_evaluation_results(save_path, results_dict):
    """Save evaluation results dictionary to a pickle file.

    Args:
        save_path: Path to the output .pkl file.
        results_dict: Dictionary of evaluation results.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"Evaluation results saved to: {save_path}")


def load_evaluation_results(save_path):
    """Load evaluation results dictionary from a pickle file.

    Args:
        save_path: Path to the .pkl file.

    Returns:
        Dictionary of evaluation results.
    """
    with open(save_path, 'rb') as f:
        results_dict = pickle.load(f)
    print(f"Evaluation results loaded from: {save_path}")
    return results_dict
