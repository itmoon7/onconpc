import warnings
import pandas as pd
import xgboost as xgb
import sys
sys.path.append('../codes')
import utils as utils
import gradio as gr
import deconstruct_sigs_from_user_input as deconstructSigs
import pickle
import gradio as gr

global image # path to explanation plot, defined as global for the purposes of update 
global features

markdown_text = """
                **Feature Category Explanations**

                **Somatic mutations**: Single nucleotide variants (SNV) and indels. For each gene, the total count of a somatic mutation is encoded as a positive integer feature.

                **Copy number alterations (CNA)**: CNA event for each gene was encoded as a categorical variable with the following five levels:
                - `-2`: Deep loss
                - `-1`: Single-copy loss
                - `0`: No event
                - `1`: Low-level gain
                - `2`: High-level amplification.

                **Mutation signatures (single-base substitution (SBS))**: Mutation signatures were inferred by calculating the dot product of COSMIC-derived weights and 96 single-base substitutions in a trinucleotide context.
                - SBS4: Tobacco smoking
                - SBS7: Ultraviolet light exposure
                - SBS22: Aristolochic acid exposure
                - SBS24: Aflatoxin exposure
                - See more details in [COSMIC](https://cancer.sanger.ac.uk/signatures/sbs/).

                **Age/Sex**: Patient age at the time of sequencing and biological sex.

                **SHAP**: impact on OncoNPC prediction (larger magnitude indicates greater impact)
                """

# defining global variables
xgb_onconpc = xgb.Booster()
xgb_onconpc.load_model('../model/xgboost_v1.7.6_OncoNPC_full.json')
cancer_types_to_consider = ['Acute Myeloid Leukemia', 'Bladder Urothelial Carcinoma', 'Cholangiocarcinoma',
                            'Colorectal Adenocarcinoma', 'Diffuse Glioma', 'Endometrial Carcinoma',
                            'Esophagogastric Adenocarcinoma', 'Gastrointestinal Neuroendocrine Tumors', 'Gastrointestinal Stromal Tumor',
                            'Head and Neck Squamous Cell Carcinoma',
                            'Invasive Breast Carcinoma', 'Melanoma', 'Meningothelial Tumor',
                            'Non-Hodgkin Lymphoma', 'Non-Small Cell Lung Cancer', 'Ovarian Epithelial Tumor', 'Pancreatic Adenocarcinoma',
                            'Pancreatic Neuroendocrine Tumor', 'Pleural Mesothelioma', 'Prostate Adenocarcinoma', 'Renal Cell Carcinoma',
                            'Well-Differentiated Thyroid Cancer']   

all_features = pd.read_csv('../data/onconpc_features.csv').drop('Unnamed: 0', axis=1).columns.tolist()

def get_preds(patients_file, samples_file, mutations_file, cna_file, tumor_id):

    """
    Generates predictions and explanations for given tumor samples using OncoNPC model.

    This function processes patient, sample, mutation, and CNA data to predict primary sites of 
    Cancer of Unknown Primary (CUP) tumors. It also provides a bar chart of SHAP values to explain the predictions.

    Args:
        patients_file: A csv file object representing patient data.
        samples_file: A csv file object representing sample data.
        mutations_file: A csv file object representing mutation data.
        cna_file: A csv file object representing CNA (Copy Number Alterations) data.
        tumor_id: The ID of the tumor.

    Returns:
        A tuple containing:
            A string containing the top 3 most probable cancers along with their predicted probabilities. 
            The filepath to the SHAP value bar chart explaining the prediction for the given tumor ID.
    """
    
    # convert files to data frames
    patients_df = pd.read_csv(patients_file.name, sep='\t')
    samples_df = pd.read_csv(samples_file.name, sep='\t')
    mutations_df = pd.read_csv(mutations_file.name, sep='\t')
    cna_df = pd.read_csv(cna_file.name, sep='\t') 

    
    patients_columns = set(['PATIENT_ID', 'SEX'])
    for col in patients_columns:
        if col not in set(patients_df.columns.to_list()):
            raise gr.Error(f'Patients File: Expected columns: ' + str(list(patients_columns)) + ', Received: ' + str(patients_df.columns.to_list()))

    samples_columns = set(['PATIENT_ID', 'AGE_AT_SEQ_REPORT', 'SAMPLE_ID'])
    for col in samples_columns:
        if col not in set(samples_df.columns.to_list()):
            raise gr.Error(f'Samples File: Expected columns: ' + str(list(samples_columns)) + ', Received: ' + str(samples_df.columns.to_list()))
    
    mutations_columns = set(['Tumor_Sample_Barcode', 'Hugo_Symbol', 'Chromosome', 'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele2'])
    for col in mutations_columns:
        if col not in set(mutations_df.columns.to_list()):
            raise gr.Error(f'Mutations File: Expected columns: ' + str(list(mutations_columns)) + ', Received: ' + str(mutations_df.columns.to_list()))
    
    cna_columns = set(['Hugo_Symbol'])
    for col in cna_columns:
        if col not in set(cna_df.columns.to_list()):
            raise gr.Error(f'CNA File: Expected columns: ' + str(list(cna_columns)) + ', Received: ' + str(cna_df.columns.to_list()))
    
    # declared as global variables to generate plots in update_image function
    global sample_id 
    global features
    global predictions

    # get features and labels for OncoNPC predictive inference
    df_features_genie_final, df_labels_genie = utils.get_onconpc_features_from_raw_data(
        patients_df,
        samples_df,
        mutations_df,
        cna_df,
        features_onconpc_path='../data/features_onconpc.pkl',
        combined_cohort_age_stats_path='../data/combined_cohort_age_stats.pkl',
        mut_sig_weights_filepath='../data/mutation_signatures/sigProfiler*.csv'
    )
    df_features_genie_final.to_csv('../data/onconpc_features2.csv')

    sample_id = tumor_id
    features = df_features_genie_final

    # load fully trained OncoNPC model
    xgb_onconpc = xgb.Booster()
    xgb_onconpc.load_model('../model/xgboost_v1.7.6_OncoNPC_full.json')
    
    # predict primary sites of CUP tumors
    predictions = utils.get_xgboost_latest_cancer_type_preds(xgb_onconpc,
                                                          df_features_genie_final,
                                                          cancer_types_to_consider)

    # get SHAP values for CUP tumors
    warnings.filterwarnings('ignore')
    shaps = utils.obtain_shap_values_with_latest_xgboost(xgb_onconpc, df_features_genie_final)
    

    query_ids = list(samples_df.SAMPLE_ID.values)

    # results is structured such that:
    # results_dict[query_id] = {'pred_prob': pred_prob,'pred_cancer': pred_cancer,'explanation_plot': full_filename}
    results = utils.get_onconpc_prediction_explanations(query_ids, predictions, shaps,
                                                        df_features_genie_final,
                                                        cancer_types_to_consider,
                                                        save_plot=True) 

    return get_top3(predictions, tumor_id),  gr.Markdown(markdown_text), results[tumor_id]['explanation_plot']

# def parse_inputs(age, gender, CNA_events, mutations, sample_id):

#     # Normalization of age input
#     combined_cohort_age_stats = None
#     combined_cohort_age_stats_path = '../data/combined_cohort_age_stats.pkl'
#     with open(combined_cohort_age_stats_path, "rb") as fp:
#         combined_cohort_age_stats = pickle.load(fp)
#     age =  (age - combined_cohort_age_stats['Age_mean']) / combined_cohort_age_stats['Std_mean']

#     # Initialize data dictionary and populate with age, sex, and sample id
#     data = {'AGE_AT_SEQ_REPORT': age, 'SEX':gender, 'SAMPLE_ID':sample_id} 
#     df_patients = pd.DataFrame([data])
    
#     # Process CNA events: Expected format [[CNA, val], [CNA, val]]
#     if len(CNA_events) > 0:
#         CNA_events = CNA_events.split('|')
#         for i in range(len(CNA_events)):
#             # Split each event into CNA and value, and cast the value to integer
#             CNA, val = CNA_events[i].split()
#             CNA_events[i] = [CNA + ' CNA', int(val)] # Cast val to integer
#     else:
#         CNA_events = []

#     CNA_dict = {}
#     # Add CNA events to data dictionary
#     for CNA, val in CNA_events:
#         CNA_dict[CNA] = val

#     df_cna = pd.DataFrame([CNA_dict])

#     # Process mutations: Expected format [mut1, mut2, etc.]
#     if len(mutations) > 0:
#         mutations = mutations.split('| ')
#         for i in range(len(mutations)):
#             # Split each mutation entry and strip white space
#             mutations[i] = ['manual input'] + mutations[i].split(', ')
#             mutations[i] = [m.strip() for m in mutations[i]] # Strip white space
#     else:
#         mutations = []

#     # Define mutation columns for DataFrame and create mutation DataFrame
#     mutation_columns = ["UNIQUE_SAMPLE_ID", "HUGO_SYMBOL", "CHROMOSOME", "POSITION", "REF_ALLELE", "ALT_ALLELE"]
#     mutation_full_df = pd.DataFrame(mutations, columns=mutation_columns) if mutations else pd.DataFrame(columns=mutation_columns)
#     mutation_partial_df = mutation_full_df[['UNIQUE_SAMPLE_ID', 'HUGO_SYMBOL']]
#     mutation_partial_df = mutation_partial_df.rename(columns={'UNIQUE_SAMPLE_ID':'Tumor_Sample_Barcode', 'HUGO_SYMBOL':'Hugo_Symbol'})
#     mutation_df = mutation_full_df.drop('HUGO_SYMBOL', axis=1, inplace=False)
    
#     # Save mutation data to a CSV file
#     mutation_df.to_csv('./mutation_input.csv', index=False)

#     # Get base substitution file and read into DataFrame
#     base_sub_file = deconstructSigs.get_base_substitutions() 
#     df_trinuc_feats = pd.read_csv(base_sub_file) 
#     # Obtain mutation signatures
#     mutation_signatures = utils.obtain_mutation_signatures(df_trinuc_feats)  


#     # Return processed features
#     return utils.pre_process_features_genie(mutation_partial_df, df_cna, mutation_signatures, df_patients)[0] # just features, not labels


def parse_inputs(age, gender, CNA_events, mutations):
    # Normalization of age input
    age = age 
    combined_cohort_age_stats = None
    combined_cohort_age_stats_path = '../data/combined_cohort_age_stats.pkl'
    with open(combined_cohort_age_stats_path, "rb") as fp:
        combined_cohort_age_stats = pickle.load(fp)
    age =  (age - combined_cohort_age_stats['Age_mean']) / combined_cohort_age_stats['Std_mean']

    # Convert gender to numerical value: male as 1, otherwise -1
    gender = 1 if gender == 'Male' else -1
    
    # Process CNA events: Expected format [[CNA, val], [CNA, val]]
    if len(CNA_events) > 0:
        CNA_events = CNA_events.split('|')
        for i in range(len(CNA_events)):
            # Split each event into CNA and value, and cast the value to integer
            CNA, val = CNA_events[i].split()
            CNA_events[i] = [CNA + ' CNA', int(val)] # Cast val to integer
    else:
        CNA_events = []

    # Process mutations: Expected format [mut1, mut2, etc.]
    if len(mutations) > 0:
        mutations = mutations.split('| ')
        for i in range(len(mutations)):
            # Split each mutation entry and strip white space
            mutations[i] = ['manual input'] + mutations[i].split(', ')
            mutations[i] = [m.strip() for m in mutations[i]] # Strip white space
    else:
        mutations = []

    # Define mutation columns for DataFrame and create mutation DataFrame
    mutation_columns = ["UNIQUE_SAMPLE_ID", "HUGO_SYMBOL", "CHROMOSOME", "POSITION", "REF_ALLELE", "ALT_ALLELE"]
    mutation_full_df = pd.DataFrame(mutations, columns=mutation_columns) if mutations else pd.DataFrame(columns=mutation_columns)
    mutation_df = mutation_full_df.drop('HUGO_SYMBOL', axis=1, inplace=False)
    
    # Save mutation data to a CSV file
    mutation_df.to_csv('./mutation_input.csv', index=False)

    # Get base substitution file and read into DataFrame
    base_sub_file = deconstructSigs.get_base_substitutions() 
    df_trinuc_feats = pd.read_csv(base_sub_file) 
    # Obtain mutation signatures
    mutation_signatures = utils.obtain_mutation_signatures(df_trinuc_feats)    

    # Initialize data dictionary and populate with age and mutation signatures
    data = {'Age': age} # TODO: Check how this is normalized 
    for column in mutation_signatures.columns:
        data[column] = mutation_signatures.loc[0][column]
    
    # Add CNA events to data dictionary
    for CNA, val in CNA_events:
        data[CNA] = val
    
    # Add zero values for missing features in data dictionary
    for column in all_features:
        if column not in data.keys():
            data[column] = 0

    # Return the data as a DataFrame
    return pd.DataFrame([data])


def get_preds_min_info(age, gender, CNA_events, mutations, output='Top Prediction'):
    """
    Generate predictions and explanations for cancer type based on input features.

    Parameters:
    age (int or float): The age of the individual.
    gender (str): The gender of the individual, either 'male' or 'female'.
    CNA_events (str): A string of CNA events, formatted appropriately.
    mutations (str): A string of mutation data, formatted appropriately.
    output (str, optional): Specifies the type of output; default is 'Top Prediction'.

    Returns:
    tuple: A tuple containing top 3 predictions and the path to the explanation plot.
    """
    global sample_id
    global features
    global predictions

    # Parse input features
    features = parse_inputs(age, gender, CNA_events, mutations)
    features.to_csv('../genie1.csv', index=False)
    all_features = pd.read_csv('../data/onconpc_features.csv').drop('Unnamed: 0', axis=1).columns.tolist()
    columns_to_add = {column: [0] * len(features) for column in all_features if column not in features.columns}
    print(columns_to_add)
    for column in features.columns:
        if column not in all_features:
            features = features.drop(column, axis=1)
    new_columns_df = pd.DataFrame(columns_to_add)
    features = pd.concat([features, new_columns_df], axis=1)
    
    # Generate predictions using the XGBoost model
    predictions = pd.DataFrame(utils.get_xgboost_latest_cancer_type_preds(xgb_onconpc, features, cancer_types_to_consider))
    
    # Compute SHAP values for model explanation
    shaps = utils.obtain_shap_values_with_latest_xgboost(xgb_onconpc, features)

    # Assuming a single sample is being processed
    query_ids = [0]
    sample_id = 0
    
    # Generate explanations for the predictions
    results = utils.get_onconpc_prediction_explanations(query_ids, predictions, shaps,
                                                        features, cancer_types_to_consider,
                                                        save_plot=True)

    # Return the top 3 predictions and the path to the explanation plot
    return get_top3(predictions, 0), gr.Markdown(markdown_text,elem_classes="markdown-box"), results[0]['explanation_plot']

def get_top3(predictions, tumor_sample_id):
    """
    Extracts and formats the top three cancer type predictions for a given tumor sample.

    Args:
        predictions: A DataFrame containing the cancer type predictions for various samples.
        tumor_sample_id: The ID of the tumor sample for which to extract the top three predictions.

    Returns:
        A string that lists the top three predicted cancer types and their probabilities.
    """
    # select the row corresponding to the tumor sample ID
    result = predictions.loc[tumor_sample_id]

    # transpose the row for easier processing, each row has columns cancer type, cancer probability 
    transposed_row = result.transpose()

    # remove unnecessary rows
    transposed_row = transposed_row.drop(['cancer_type', 'max_posterior'])

    # convert the series to a DataFrame and rename the column
    transposed_row = transposed_row.to_frame()
    transposed_row.columns = ['probability']

    # make sure the probability column is numeric
    transposed_row['probability'] = pd.to_numeric(transposed_row['probability'], errors='coerce')

    # get the top 3 predictions and their probabilities
    top3df = transposed_row.nlargest(3, columns=['probability'])
    top3 = top3df.index.tolist() # cancer types are indices
    top3probs = top3df['probability'].tolist()

    # build a formatted string with the top 3 predictions
    build = ''
    for cancer, prob in zip(top3, top3probs):
        build += f'{cancer}: {prob:.2f}\n'
    build = build.rstrip('\n')

    return build


def extract_sample_ids(samples_file):
    # Read the file into a DataFrame
    if samples_file is None:
        return []

    df = pd.read_csv(samples_file.name, sep='\t')
    # Assuming the column containing the sample IDs is named 'SampleID'
    sample_ids = df['SAMPLE_ID'].unique().tolist()
    return  gr.Dropdown.update(choices=sample_ids)

def update_image(target):
    global image
    global features
    global predictions
    
    shaps_cup = utils.obtain_shap_values_with_latest_xgboost(xgb_onconpc, features) # get shap values 

    target_idx = cancer_types_to_consider.index(target) # index of cancer type prediction 
    
    # Get SHAP-based explanation for the prediction
    feature_sample_df = features.loc[sample_id] # find the exact tumor sample we're predicting for 
    shap_pred_cancer_df = pd.DataFrame(shaps_cup[target_idx],
                                       index=features.index,
                                       columns=features.columns)
    shap_pred_sample_df = shap_pred_cancer_df.loc[sample_id]
    probability = predictions.loc[sample_id][target]
    
    # Generate explanation plot
    sample_info = f'Prediction: {target}\nPrediction probability: {probability:.3f}'
    feature_group_to_features_dict, feature_to_feature_group_dict = utils.partition_feature_names_by_group(features.columns)
    explanation_plot = utils.get_individual_pred_interpretation(shap_pred_sample_df, feature_sample_df, feature_group_to_features_dict, feature_to_feature_group_dict,sample_info=sample_info, filename=f'{sample_id}_{target}_plot', filepath='../others_prediction_explanation', save_plot=True)
    return explanation_plot

def show_row(value):
    """
    Toggle the visibility of UI elements based on the selected input method.

    This function is designed to work with a graphical user interface (GUI).
    It changes the visibility of elements depending on the user's choice of input method.

    Parameters:
    value (str): The input method selected by the user. Expected values are "CSV File" or "Manual Inputs".

    Returns:
    tuple: A pair of commands to update the visibility of GUI elements.
           Each command is a tuple itself, containing a method to update an element and a visibility state.
    """

    # If the user selects "CSV File" as the input method
    if value == "CSV File":
        # Make the CSV file upload element visible and hide the manual input fields
        return (gr.update(visible=True), gr.update(visible=False))

    # If the user selects "Manual Inputs" as the input method
    if value == "Manual Inputs":
        # Hide the CSV file upload element and make the manual input fields visible
        return (gr.update(visible=False), gr.update(visible=True))

    # If the user's selection does not match any expected input method
    return (gr.update(visible=False), gr.update(visible=False))  # Hide both elements


def launch_gradio(server_name, server_port):
    """
    Launches a Gradio application for cancer type prediction based on various clinical and genomic data.

    Args:
    server_name (str): Name of the server where the app will be hosted.
    server_port (int): Port number for the server.

    This function sets up a Gradio interface with options to input either manual data or CSV files.
    The application supports prediction for a predefined list of cancer types based on the input data.
    """
    cancer_types_to_consider = ['Acute Myeloid Leukemia', 'Bladder Urothelial Carcinoma', 'Cholangiocarcinoma',
                                'Colorectal Adenocarcinoma', 'Diffuse Glioma', 'Endometrial Carcinoma',
                                'Esophagogastric Adenocarcinoma', 'Gastrointestinal Neuroendocrine Tumors', 'Gastrointestinal Stromal Tumor',
                                'Head and Neck Squamous Cell Carcinoma',
                                'Invasive Breast Carcinoma', 'Melanoma', 'Meningothelial Tumor',
                                'Non-Hodgkin Lymphoma', 'Non-Small Cell Lung Cancer', 'Ovarian Epithelial Tumor', 'Pancreatic Adenocarcinoma',
                                'Pancreatic Neuroendocrine Tumor', 'Pleural Mesothelioma', 'Prostate Adenocarcinoma', 'Renal Cell Carcinoma',
                                'Well-Differentiated Thyroid Cancer']   

    markdown_text = """
                    **Prediction Explanations**

                    **Somatic mutations**: Single nucleotide variants (SNV) and indels. For each gene, the total count of a somatic mutation is encoded as a positive integer feature.

                    **Copy number alterations (CNA)**: CNA event for each gene was encoded as a categorical variable with the following five levels:
                    - `-2`: Deep loss
                    - `-1`: Single-copy loss
                    - `0`: No event
                    - `1`: Low-level gain
                    - `2`: High-level amplification.

                    **Mutation signatures (single-base substitution (SBS))**: Mutation signatures were inferred by calculating the dot product of COSMIC-derived weights and 96 single-base substitutions in a trinucleotide context.
                    - SBS4: Tobacco smoking
                    - SBS7: Ultraviolet light exposure
                    - SBS22: Aristolochic acid exposure
                    - SBS24: Aflatoxin exposure
                    - See more details in [COSMIC](https://cancer.sanger.ac.uk/signatures/sbs/).

                    **Age/Sex**: Patient age at the time of sequencing and biological sex.

                    **SHAP Values**: impact on OncoNPC prediction 
                    - larger magnitude indicates greater impact
                    """
    manual_input_description = 'In this option, you will manually enter patient and tumor data including age at sequencing, sex, copy number alterations (CNA) events, and somatic mutations.'
    csv_description = 'In this option, you can directly upload relevant CSV files, including clinical patient, clinical sample, mutations, and CNA data from [cBioPortal](https://www.cbioportal.org) or [AACR GENIE](https://www.aacr.org/professionals/research/aacr-project-genie/). Please see the example data in this [link](https://github.com/itmoon7/onconpc/tree/main/data/mock_n_3_data) <br> Note that column names `Tumor_Sample_Barcode` and `SAMPLE_ID` both refer to the sample identifier column and should be the same.'

    with gr.Blocks(css=".markdown-box { border: 1px solid #E0E0E0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); padding: 15px; margin-bottom: 20px; }") as demo:

        d = gr.Dropdown(["Manual Inputs", "CSV File"])

        with gr.Row(visible=False) as r3:
            gr.Markdown(csv_description,elem_classes="markdown-box")
        with gr.Row(visible=False) as r4:
            gr.Markdown(manual_input_description,elem_classes="markdown-box")

        with gr.Row(visible=False) as r1:
            with gr.Column():
                gr.Markdown("**Clinical Patient Data File:** Upload a file containing patient data. Required columns: `PATIENT_ID`, `SEX`.")
                patients_file = gr.File(label="Upload clinical patients data")
                gr.Markdown("**Clinical Sample Data File:** Upload a file containing sample data. Required columns: `PATIENT_ID`, `AGE_AT_SEQ_REPORT`, `SAMPLE_ID`.")
                samples_file = gr.File(label="Upload clinical samples data")
                gr.Markdown("**Mutations Data File:** Upload a file containing mutation data. Required columns: `Tumor_Sample_Barcode`, `Hugo_Symbol`, `Chromosome`, `Start_Position`, `Reference_Allele`, `Tumor_Seq_Allele2`.")
                mutations_file = gr.File(label="Upload mutations data")
                gr.Markdown("**CNA Data File:** Upload a file containing CNA data. Required column: `Hugo_Symbol`.")
                cna_file = gr.File(label="Upload CNA data")
                tumor_sample_id = gr.Dropdown(choices=[], label="Tumor Sample ID")  # Changed to Dropdown
                submit_button = gr.Button("Submit")

            with gr.Column():
                predictions_output = gr.Textbox(label="Top 3 Predicted Cancer Types")
                category_explanation = gr.Markdown(markdown_text,elem_classes="markdown-box")
                image = gr.Image(label="Image Display") 
                output_selector = gr.Dropdown(choices=cancer_types_to_consider, label="Output Options", filterable=True)
                gr.Markdown('For more details regarding OncoNPC feature processing, OncoNPC performance, and clinical implications of OncoNPC predictions, please see our [manuscript](https://www.nature.com/articles/s41591-023-02482-6).', elem_classes="markdown-box")

            samples_file.change(extract_sample_ids, inputs=samples_file, outputs=tumor_sample_id)
            submit_button.click(get_preds, inputs=[patients_file, samples_file, mutations_file, cna_file, tumor_sample_id], outputs=[predictions_output, category_explanation, image])
            output_selector.change(update_image, inputs=output_selector, outputs=[image])

        with gr.Row(visible=False) as r2:
            with gr.Column():
                age = gr.Number(label="Age")
                gender = gr.Radio(choices=["Male", "Female"], label="Gender")
                cna_events = gr.Textbox(lines=7, placeholder="Copy Number Alterations\n\nPlease enter the name of each gene followed by its corresponding CNA level. If you are entering data for multiple genes, separate each gene and its CNA level with a vertical bar '|'.\n\nFor example: RAF1 2 | PPARG 2", label="Genes with CNA Events")
                mutations = gr.Textbox(lines=7, placeholder="Somatic Mutations\n\nPlease enter the gene name, chromosome number, position of the variant, allele in the reference genome, and alternate allele. If you are entering data for multiple genes, separate each set of information with a vertical bar '|'.\n\nFor example: ERBB2, chr17, 37868208, C, T | CDKN1A, chr6, 36651880, -, GTCAGAACCGGCTGGGGATGTCC", label="MUTATIONS")
                submit_button = gr.Button("Submit")
            with gr.Column():
                predictions_output = gr.Textbox(label="Top 3 Predicted Cancer Types")
                category_explanation = gr.Markdown(markdown_text,elem_classes="markdown-box")
                image = gr.Image(label="Image Display") 
                output_selector = gr.Dropdown(choices=cancer_types_to_consider, label="Output Options", filterable=True)
                gr.Markdown('For more details regarding OncoNPC feature processing, OncoNPC performance, and clinical implications of OncoNPC predictions, please see our [manuscript](https://www.nature.com/articles/s41591-023-02482-6).', elem_classes="markdown-box")

            submit_button.click(get_preds_min_info, inputs=[age, gender, cna_events, mutations], outputs=[predictions_output, category_explanation, image])
            output_selector.change(update_image, inputs=output_selector, outputs=[image])

        d.change(show_row, d, [r1, r2])
        d.change(show_row, d, [r3, r4])
        
    demo.launch(debug=True, share=True,server_name=server_name, server_port=server_port)