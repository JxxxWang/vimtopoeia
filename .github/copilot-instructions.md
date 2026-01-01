# Project: Vimtopoeia

## Project Structure & Architecture
- **Separation of Concerns**: The project is strictly divided into data creation and model development.
  - `data_generation/`: Scripts responsible for creating, cleaning, or preprocessing data.
  - `model_training/`: Scripts for model architecture, training loops, and evaluation.
- **Data Flow**: 
  - Generators in `data_generation/` produce HDF5 files (e.g., `dataset_10k.h5`) in the project root.
  - Training scripts in `model_training/` consume these HDF5 files.
  - **Constraint**: Do not hardcode absolute paths. Use `pathlib` to reference the root-level `.h5` files.

## Development Workflow
1. **Environment**: Active virtual environment in `.venv`.
2. **Pipeline**: 
   - Always ensure data generation scripts are runnable to reproduce `dataset_10k.h5`.
   - Training scripts should validate the existence of the dataset before execution.

## Coding Conventions
- **Data Handling**: 
  - Use `h5py` or `pandas` (HDFStore) for interacting with `.h5` files.
  - Document input/output tensor shapes in docstrings.
- **Python**:
  - Use strict type hinting.
  - Prefer `pathlib.Path` over `os.path`.
