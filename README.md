# OncoNPC

We developed **OncoNPC** (**Onco**logy **N**GS-based **P**rimary cancer type **C**lassifier), a molecular cancer type classifier trained on multicenter targeted panel sequencing data. OncoNPC utilizes somatic alterations including mutations (single nucleotide variants and indels), mutational signatures, copy number alterations, as well as patient age at the time of sequencing and sex to jointly predict cancer type.

To ensure consistent execution of the code, we recommend using the conda environment specified in `onconpc_conda.yml`. This file contains all the necessary dependencies with their specific versions.

## Setting up the Conda Environment

1. **Install Conda**: If you do not have Conda installed, please install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

2. **Clone the Repository**: 
   ```bash
   git clone https://github.com/your-github-username/onconpc.git
   cd onconpc
   ```

3. **Create Conda Environment**: Use the `onconpc_conda.yml` file to create a conda environment.
   ```bash
   conda env create -f onconpc_conda.yml
   ```

4. **Activate the Environment**: Once the environment is created, you can activate it using:
   ```bash
   conda activate onconpc_conda_env
   ```

5. **Running the Code**: With the environment activated, you can now run the code within this repository.

For further details on the software and package versions, please refer to the `onconpc_conda.yml` file.

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
   - Execute `codes/deconstructSigs_trinucs_data.R` with `data_mutations_extended.txt`. This step is crucial for processing SNVs in tri-nucleotide contexts, essential for prepraring mutation signature features.
   - The pre-processing code then uses weight matrices from the [COSMIC Sanger Signatures](https://cancer.sanger.ac.uk/signatures/sbs/) to generate continuous values for mutation signatures.

3. **Setting up Data Directories**: In the `load_genie_data()` of `process_features.py` script, specify directories for the AACR GENIE data files.

4. **Run the Pre-processing Script**: To process the AACR GENIE data, use the command:

   ```bash
   python process_features.py --config=genie --filename_suffix=[UNIQUE FILENAME SUFFIX]
   ```

   This creates processed feature and label dataframes in the /data/ directory, appended with the provided filename suffix.

## Training and Validating the XGBoost-based OncoNPC Model

After processing features with `process_features.py`, you can train and validate the OncoNPC model:

1. **Specify Data Locations**: In `train_evaluate_onconpc.py`, set the locations of the processed features.

2. **Training and Validation**: Use the following command for model training and validation:
   ```bash
   python train_evaluate_onconpc.py --config=genie --save_model_name=[UNIQUE FILENAME SUFFIX OF YOUR CHOICE] --k_fold=10
   ```
   - The script supports k-fold cross-validation for model assessment. Setting `k_fold` to a specific value (e.g., 10) enables this feature.
   - If `k_fold` is set to 0, the model trains on the entire dataset and saves using `save_model_name` as a filename suffix.

3. **Outputs**: The script outputs performance results and cancer type predictions for tumor samples. The trained model is saved with the specified filename suffix.

### Note

Adapting the OncoNPC model to AACR GENIE data may result in variations in performance or results due to differences in data sources from the original DFCI training set.


## Notebook Example for Predicting Cancer Type and Visualizaing Prediction Explanation.

The [notebook example](https://github.com/itmoon7/onconpc/blob/main/onconpc_prediction_and_explanation_for_cup_tumors.ipynb) in this repository illustrates the practical application of the OncoNPC model. It includes:

1. **Cancer Type Prediction for CUP Tumors**: Using the trained OncoNPC model, the notebook demonstrates how to predict the primary cancer type of Cancers of Unknown Primary (CUP) based on molecular data.

2. **Feature Visualization**: It showcases how to visualize the most important features contributing to each cancer type prediction. This is achieved through the calculation and plotting of SHAP (SHapley Additive exPlanations) values.

3. **Detailed Case Study**: A specific case may be presented to demonstrate how the model predicts the cancer type and how SHAP values provide insights into the prediction.

This notebook serves as a guide for researchers and clinicians to understand the model's predictions and the key features influencing these predictions in the context of cancer genomics.

## Additional Resources

- [Manuscript](https://www.nature.com/articles/s41591-023-02482-6)

## Citation

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