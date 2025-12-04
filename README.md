# AF Code Repository

This repository contains code for Atrial Fibrillation (AF) detection using ECG data. It includes models for training and validation, along with necessary preprocessing and utility functions.

## Project Structure

- `modeling_MIMIC_AfibChallengeDataset.ipynb`: Main notebook for training the AF detection model using the MIMIC Afib Challenge Dataset.
- `externalValidationShaoxing.ipynb`: Notebook for external validation of the trained model using the Shaoxing ECG dataset.
- `src/`: Directory containing Python modules for models, preprocessing, and utilities.
    - `se_resnet_models.py`: SE-ResNet model definitions.
    - `resnet_models.py`: ResNet model definitions.
    - `preprocessing.py`: Signal preprocessing functions (cleaning, QRS detection).
    - `modeling_results.py`: Functions for evaluating and plotting model results.
    - `read_gcp.py`: Utilities for Google Cloud Storage (if needed).
- `data/`: Directory to store input datasets (see Data Preparation).
- `models/`: Directory to save and load trained models.
- `requirements.txt`: List of Python dependencies.

    ```

## Usage

1.  **Training:**
    Open `modeling_MIMIC_AfibChallengeDataset.ipynb` in Jupyter Notebook or JupyterLab.
    Run the cells to preprocess data, train the ResNet/SE-ResNet models, and save the best weights to the `models/` directory.

2.  **Validation:**
    Open `externalValidationShaoxing.ipynb`.
    Ensure the trained model weights (e.g., `SE_resnet_34_smallBatch.hdf5`) are present in the `models/` directory.
    Run the cells to load the model and evaluate it on the external dataset.

## Models

The project implements ResNet and SE-ResNet architectures optimized for 1D ECG signal classification. The model definitions can be found in `src/resnet_models.py` and `src/se_resnet_models.py`.

## License

[Insert License Here]
