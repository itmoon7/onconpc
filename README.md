# OncoNPC

We developed **OncoNPC** (**Onco**logy **N**GS-based **P**rimary cancer type **C**lassifier), a molecular cancer type classifier trained on multicenter targeted panel sequencing data. OncoNPC utilizes somatic alterations including mutations (single nucleotide variants and indels), mutational signatures, copy number alterations, as well as patient age at the time of sequencing and sex to jointly predict cancer type.

## Tutorial Video and Visualization Tool
We have created a short tutorial video to guide you through the setup and usage of OncoNPC's GitHub-based visualization tool, which helps visualize cancer type predictions based on user-provided inputs and model explanations.

<div align="center">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/I0mmmvuC5ug?si=46OA3-bGYbKUO4Uh" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

You can also directly access the visualization tool [here](https://itmoon7.github.io/onconpc/).

We'll walk you through the installation of the OncoNPC pipeline for feature processing and inference.

## Table of Contents
- [Setting up the Conda Environment](#setting-up-the-conda-environment)
- [Utilizing Public Tumor Sequencing Data from AACR GENIE](#utilizing-public-tumor-sequencing-data-from-aacr-genie)
- [Training and Validating the XGBoost-based OncoNPC Model](#training-and-validating-the-xgboost-based-onconpc-model)
- [Notebook Examples for Predicting Cancer Type and Visualizing Prediction Explanation](#notebook-examples-for-predicting-cancer-type-and-visualizing-prediction-explanation)
- [Link to Manuscript](#link-to-manuscript)

## Setting up the Conda Environment

To ensure consistent execution of the code, we recommend using the conda environment specified in `onconpc_conda.yml`. This file contains all the necessary dependencies with their specific versions.

### 1. Install Conda
If you do not have Conda installed, please install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

### 2. Clone the Repository
```bash
git clone https://github.com/itmoon7/onconpc.git
cd onconpc
```

### 3. Create Conda Environment
Use the `onconpc_conda.yml` file to create a Conda environment.
```bash
conda env create -f onconpc_conda.yml
```

### 4. Activate the Environment
Once the environment is created, you can activate it using:
```bash
conda activate onconpc_conda_env
```

For further details on the software and package versions, please refer to the `onconpc_conda.yml` file.
After setting up the Conda environment and installing the necessary Python packages, you will also need to install R packages crucial for processing mutation data and generating features for the OncoNPC model.

The script `install_r_packages_onconpc.sh` is provided in the main repository for installing these R packages.

### 5. Making the Script Executable
Before running the script, you need to make it executable. In the root directory of the project, run:
```bash
chmod +x install_r_packages_onconpc.sh
```

### 6. Running the Script
Execute the script to install the necessary R packages:
```bash
./install_r_packages_onconpc.sh
```

This script will handle the installation of R packages required for the analysis, including the setup for SNV in tri-nucleotide context and mutation signature features using the `deconstructSigs` R library. With the environment activated, you can now run the code within this repository.

## Utilizing Public Tumor Sequencing Data from AACR GENIE

### Introduction to AACR GENIE Data

OncoNPC was originally trained using data from multiple sources, including publicly available data from AACR Project GENIE, specifically from two cancer centers (MSK and VICC), as well as private institutional data from DFCI (Dana-Farber Cancer Institute). This repository provides the flexibility to process and train the OncoNPC model using solely the publicly available data from AACR Project GENIE.

### Required Data Files from AACR GENIE

For integrating AACR GENIE data with OncoNPC, you will need:

1. **data_mutations_extended.txt**
2. **data_clinical_patient.txt**
3. **data_clinical_sample.txt**
4. **data_CNA.txt**

### Integrating AACR GENIE Data with OncoNPC

1. **Accessing AACR GENIE Data**: Begin by obtaining the AACR GENIE data as described in their [Data Guide](https://www.aacr.org/wp-content/uploads/2023/09/14.0-data_guide.pdf).

2. **Preparing Mutataion Signature features**: 
   - The pre-processing code then uses weight matrices from the [COSMIC Sanger Signatures](https://cancer.sanger.ac.uk/signatures/sbs/) to generate continuous values for mutation signatures.

3. **Setting up Data Directories**: In the `load_genie_data()` of `process_features.py` script, specify directories for the AACR GENIE data files.

4. **Run the Pre-processing Script**: To process the AACR GENIE data, use the command:

   ```bash
   python process_features.py --config=genie --filename_suffix=[UNIQUE FILENAME SUFFIX]
   ```

   This creates processed feature and label dataframes in the /data/ directory, appended with the provided filename suffix.

## Training and Validating the XGBoost-based OncoNPC Model

After processing features with `process_features.py`, you can train and validate the OncoNPC model. To do this, follow these steps:

1. **Specify Data Locations in `train_evaluate_onconpc.py`**:
   - Open the `train_evaluate_onconpc.py` file.
   - Set the locations of the processed feature and label data. This can be done around lines 50-53. You will need to specify the file paths for both the features and labels. For example:
     ```python
     # Tab separated feature data for CKPs
     feature_data_name = os.path.join(DATA_PATH, 'features_genie_')  # Replace with your feature file path
     # Tab separated label data for CKPs
     label_data_name = os.path.join(DATA_PATH, 'labels_genie_')  # Replace with your label file path
     ```
   - Ensure that `DATA_PATH` is correctly defined to point to the directory where your processed data files are located.
   - Replace `'features_genie_'` and `'labels_genie_'` with the names of your actual processed feature and label files.

2. **Training and Validation**: Use the following command for model training and validation:
   ```bash
   python train_evaluate_onconpc.py --config=genie --save_model_name=[UNIQUE FILENAME SUFFIX OF YOUR CHOICE] --k_fold=10
   ```
   - The script supports k-fold cross-validation for model assessment. Setting `k_fold` to a specific value (e.g., 10) enables this feature.
   - If `k_fold` is set to 0, the model trains on the entire dataset and saves using `save_model_name` as a filename suffix.

3. **Outputs**: The script outputs performance results and cancer type predictions for tumor samples. The trained model is saved with the specified filename suffix.

### Note

Adapting the OncoNPC model to AACR GENIE data may result in variations in performance or results due to differences in data sources from the original DFCI training set.

## Notebook Examples for Predicting Cancer Type and Visualizing Prediction Explanation

### 1. OncoNPC Model Application for CUP Tumors
The [OncoNPC Prediction and Explanation for CUP Tumors notebook](https://github.com/itmoon7/onconpc/blob/main/onconpc_prediction_and_explanation_for_cup_tumors.ipynb) in this repository provides a practical application of the OncoNPC model. Key highlights include:

   - **Cancer Type Prediction for CUP Tumors**: Demonstrates using the trained OncoNPC model to predict the primary cancer type of CUP tumors based on molecular data.

   - **Feature Visualization**: Showcases how to visualize important features contributing to each cancer type prediction using SHAP (SHapley Additive exPlanations) values.

   - **Detailed Case Study**: Presents a specific case to illustrate the model's cancer type predictions and how SHAP values offer insights.

This notebook is a resource for researchers and clinicians to understand the model's predictions and the key features influencing these predictions in cancer genomics.

### 2. Direct Data Loading and OncoNPC Prediction
The [Direct Data Loading and OncoNPC Prediction notebook](https://github.com/itmoon7/onconpc/blob/main/onconpc_prediction_and_explanation_for_cup_tumors_from_cbio_raw.ipynb) adds functionality for:

   - **Loading Raw Data**: Utilizes raw cBioPortal-like or GENIE AACR public data, streamlining the process of data preparation.

   - **Automated Cancer Type Prediction**: Integrates a function to automatically predict cancer types using the OncoNPC model.

   - **Visualization of Prediction Explanation**: Visualizing the prediction explanation, offering clarity on how the model arrives at its conclusions for each tumor sample.

This notebook is designed to simplify the process for users who want to apply the OncoNPC model directly to raw datasets.

## Link to Manuscript

- [Manuscript](https://rdcu.be/drq1a)

### Citation

```
@article{moon2023machine,
  title={Machine learning for genetics-based classification and treatment response prediction in cancer of unknown primary},
  author={Moon, Intae and LoPiccolo, Jaclyn and Baca, Sylvan C and Sholl, Lynette M and Kehl, Kenneth L and Hassett, Michael J and Liu, David and Schrag, Deborah and Gusev, Alexander},
  journal={Nature Medicine},
  pages={1--11},
  year={2023},
  publisher={Nature Publishing Group US New York}
}
```
