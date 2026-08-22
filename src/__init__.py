"""Meta-learning framework with Neural Processes and Gradient Boosting."""

# Define the package version
__version__ = "0.1.0"

# Import key classes to make them accessible at the top level
from .data.datasets import NPBDataset
from .models.neural_process import NeuralProcess

# Define the public API for wildcard imports (e.g., from src import *)
__all__ = [
    "NeuralProcess",
    "NPBDataset",
]
