"""
Author: Intae Moon

Date : November 20th, 2022
Description : 
The following script provides features and labels for a supervised training of classifying tumors with known primaries based on samples'
1. Clinical information (e.g. age and sex)
2. Somatic mutation profile (i.e. frequence of mutated genes)
3. Somatic copy number alteration events (i.e. genes associated with CNA : -2 (deep loss), -1 (single-copy loss), 0 (diploid), 1 (low-level gain), 2 (high-level amplification))
4. Somatic mutation signatures.
"""

import os
from absl import app
from absl import flags
from typing import List, Any, Tuple

import glob
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

# import warnings
# warnings.filterwarnings("ignore", message="Numerical issues were encountered ")
FLAGS = flags.FLAGS
_FILEPATH = '../data/seq_panel_data'
_FILENAME_SUFFIX = flags.DEFINE_string('filename_suffix', 'default', 'Filename suffix for processed files.')

def obtain_mutation_signatures(df_trinuc_feats: pd.DataFrame) -> pd.DataFrame:
	"""Transforms tri-nucleotide features into mutation signature based features.

	Args:
		df_trinuc_feats: DataFrame containing tri-nucleotide features.
	Returns:
		mut_sig_based_df: DataFrame containing mutation signature based features.
	"""
	# Load Mutational Signatures COSMIC data.
	file_names = glob.glob('../data/mutation_signatures/sigProfiler*.csv')
	# Check if the directory exists.
	if len(file_names) == 0:
		raise ValueError('No mutation signatures data.')
	file_names_key = [file_name.split('_')[-1][:-4] for file_name in file_names]
	sbs_conversion_df_dict = {}	
	for i in range(len(file_names)):
		df = pd.read_csv(file_names[i])
		df.dropna(inplace = True)
		# process base names s.t. they compatible with actual feature names
		new_name_list = []
		for change, subtype in zip(df.Type.values, df.Subtype.values):
			new_name = subtype[0] + '[' + change + ']' + subtype[2]
			new_name_list.append(new_name)

		df['Type_new'] = new_name_list
		df.set_index('Type_new', inplace = True)
		# check if column contains human-based experiment results
		merged_colum_names = ''.join(df.columns)
		if 'GRCh38' in merged_colum_names:
			df['values_oi'] = df[file_names_key[i] + '_GRCh38'].values
		elif 'GRCh37' in merged_colum_names:
			df['values_oi'] = df[file_names_key[i] + '_GRCh37'].values
		else:
			raise ValueError('No human-based experimental results.')
		sum_vals = np.sum(df.values_oi.values)
		if abs(sum_vals - 1.0) > 0.1:
			raise ValueError('Values do not sum up to 1.')
		sbs_conversion_df_dict[file_names_key[i]] = df
	mut_sig_based_df = pd.DataFrame([], index = df_trinuc_feats.index)
	for key, mut_df in sbs_conversion_df_dict.items():
		common_substitute_feats = set(mut_df.index) & set(df_trinuc_feats.columns)
		if len(common_substitute_feats) != 96:
			raise ValueError('Number of substitution features does not sum up to 96.')
		# Get mutation signature values
		mut_sig_vals = np.matmul(df_trinuc_feats[common_substitute_feats].values,
								 mut_df.loc[common_substitute_feats].values_oi.values)
		mut_sig_based_df[key] = mut_sig_vals
	return mut_sig_based_df

def merge_dfs(df_list: pd.DataFrame, ids_common: List[str]) -> pd.DataFrame:
	"""Merges a list of dataframes using common sample IDs.
	
	Args:
		df_list: List of dataframes to merge.
		ids_common: Common sample IDs.
	Returns:
		df_merged_to_return: Merged dataframe.
	"""
	for idx, df in enumerate(df_list):
		if idx == 0:
			df_merged_to_return = df.loc[ids_common]
		else:
			df_merged_to_return = pd.merge(df_merged_to_return, df.loc[ids_common], how = 'left', left_index = True, right_index = True)
	return df_merged_to_return

def pre_process_features_genie(df_mutations: pd.DataFrame,
							   df_cna: pd.DataFrame,
							   df_mutation_signatures: pd.DataFrame,
							   df_patients: pd.DataFrame,
							   cancer_types: List[str], 
							   id_column: str = 'SAMPLE_ID') -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Pre-process genetics data to create feature df for GENIE.
	
	Args:
		df_mutations: DataFrame containing mutation data.
		df_cna: DataFrame containing CNA data.
		df_mutation_signatures: DataFrame containing mutation signature data.
		df_patients: DataFrame containing patient data.
		cancer_types: List of cancer types.
		id_column: Column name for sample IDs.
	Returns:
		df_features_merged_final: DataFrame containing merged features.
		df_labels_final: DataFrame containing labels.
	"""
	# Get mutation features.
	sample_ids = df_patients[id_column].values
	df_mutations_chosen = df_mutations.loc[df_mutations.Tumor_Sample_Barcode.isin(sample_ids)]
	# add 'mut' at the end of gene mutation feature
	mutation_gene_names = np.unique(df_mutations_chosen.Hugo_Symbol.values)
	mutation_gene_names = [gene + '_mut' for gene in mutation_gene_names]
	df_mutation_feature = pd.DataFrame(0, columns=mutation_gene_names, index=sample_ids)
	for entry in df_mutations_chosen[['Tumor_Sample_Barcode', 'Hugo_Symbol']].values:
		sample_id = entry[0]; gene = entry[1];
		if gene in df_mutation_feature.columns:
			# Count the number of mutation frequency in each gene.
			df_mutation_feature.at[sample_id, gene] = df_mutation_feature.at[sample_id, gene] + 1
	df_mutation_feature.columns = mutation_gene_names
		
	# Get CNA features.
	df_cna_chosen = df_cna.loc[set(df_cna.index) & set(sample_ids)]
	df_cna_chosen.fillna(0, inplace = True)

	# Obtain age/sex features.
	df_patients_id_indexed = df_patients.set_index('SAMPLE_ID')
	df_sex_age_feature = df_patients_id_indexed[['SEX', 'AGE_AT_SEQ_REPORT']]
	sex_list = []
	age_nan_indices = []
	age_list = []
	for sample_id, (sex, age) in zip(df_sex_age_feature.index, df_sex_age_feature.values):
		if sex == 'Male':
			sex_list.append(1)
		elif sex == 'Female':
			sex_list.append(-1)
		else: # Not reported
			sex_list.append(0)
		if pd.isnull(age):
			age_nan_indices.append(sample_id)
			age_list.append(0) # append zero for now
		elif '>89' in age:
			age_list.append(int(age[1:]))
		elif '<18' in age:
			age_list.append(int(age[1:]))
		else:
			age_list.append(int(age))

	df_sex_age_feature['SEX'] = sex_list
	df_sex_age_feature['AGE_AT_SEQ_REPORT'] = age_list
	df_sex_age_feature.columns = ['Sex', 'Age']
	# Merge all feature dfs using common sample IDs.
	common_ids = (set(df_mutation_feature.index) & set(df_cna_chosen.index) & set(df_mutation_signatures.index)
				  & set(df_sex_age_feature.index))
	dfs_list = [df_mutation_feature, df_cna_chosen, df_mutation_signatures, df_sex_age_feature]
	df_features_merged = merge_dfs(dfs_list, common_ids)
	# Exclude age NaN indices.
	df_features_merged_final = df_features_merged.loc[set(df_features_merged.index) - set(age_nan_indices)]
	df_labels = df_patients_id_indexed.loc[df_features_merged_final.index]['CANCER_TYPE']
	df_labels_final = pd.DataFrame([cancer_types.index(val) for val in df_labels.values],
								columns = ['cancer_label'], index = df_labels.index)
	df_labels_final['cancer_type'] = df_labels.values
	return df_features_merged_final, df_labels_final	

def pre_process_features_dfci(df_mutations,
							  df_cna,
							  df_mutation_signatures,
							  df_patients,
							  cancer_types, 
							  cup_samples_ids: Any = None,
							  id_column: str = 'UNIQUE_SAMPLE_ID'
							  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Pre-process genetics data to create feature df for DFCI data.
	
	Args:
		df_mutations: DataFrame containing mutation data.
		df_cna: DataFrame containing CNA data.
		df_mutation_signatures: DataFrame containing mutation signature data.
		df_patients: DataFrame containing patient data.
		cancer_types: List of cancer types.
		cup_samples_ids: List of CUP sample IDs.
		id_column: Column name for sample IDs.
	Returns:
		df_features_merged_ckps: DataFrame containing merged features for CKP samples.
		df_labels_final: DataFrame containing labels for CKP samples.
		df_features_merged_cups: DataFrame containing merged features for CUP samples.
	"""
	# Get mutation features.
	sample_ids = df_patients[id_column].values
	df_mutations_chosen = df_mutations.loc[df_mutations[id_column].isin(sample_ids)]
	# add 'mut' at the end of gene mutation feature
	mutation_gene_names = np.unique(df_mutations_chosen.CANONICAL_GENE.values)
	mutation_gene_names = [gene + '_mut' for gene in mutation_gene_names]
	df_mutation_feature = pd.DataFrame(0, columns=mutation_gene_names, index=sample_ids)
	for entry in df_mutations_chosen[['UNIQUE_SAMPLE_ID', 'CANONICAL_GENE']].values:
		sample_id = entry[0]; gene = entry[1];
		if gene in df_mutation_feature.columns:
			# Count the number of mutation frequency in each gene.
			df_mutation_feature.at[sample_id, gene] = df_mutation_feature.at[sample_id, gene] + 1
	df_mutation_feature.columns = mutation_gene_names
		
	# Get CNA features.
	df_cna_chosen = df_cna.loc[set(df_cna.index) & set(sample_ids)]
	df_cna_chosen.fillna(0, inplace = True)

	# Obtain age/sex features.
	df_patients_id_indexed = df_patients.set_index(id_column)
	df_sex_age_feature = df_patients_id_indexed[['Sex', 'Age']]
	sex_list = []
	age_nan_indices = []
	age_list = []
	for sample_id, (sex, age) in zip(df_sex_age_feature.index, df_sex_age_feature.values):
		if sex == 'MALE':
			sex_list.append(1)
		elif sex == 'FEMALE':
			sex_list.append(-1)
		else: # Not reported
			sex_list.append(0)
	df_sex_age_feature['Sex'] = sex_list
	df_sex_age_feature.columns = ['Sex', 'Age']
	# Merge all features using the same sample IDs.
	common_ids = (set(df_mutation_feature.index) & set(df_cna_chosen.index) &
				  set(df_mutation_signatures.index) & set(df_sex_age_feature.index))
	dfs_list = [df_mutation_feature, df_cna_chosen, df_mutation_signatures, df_sex_age_feature]
	df_features_merged = merge_dfs(dfs_list, common_ids)
	# Make gene names consistent across GENIE and PROFILE.
	profile_old_new_gene_mapping = {}
	profile_old_new_gene_mapping['C17ORF70'] = 'FAAP100' 
	profile_old_new_gene_mapping['C17orf70_mut'] = 'FAAP100_mut'
	profile_old_new_gene_mapping['C19ORF40'] = 'FAAP24' 
	profile_old_new_gene_mapping['C19orf40_mut'] = 'FAAP24_mut'
	profile_old_new_gene_mapping['LOC96610_mut'] = 'BMS1P20_mut'
	profile_old_new_gene_mapping['LOC729991-MEF2B_mut'] = 'MEF2BNB-MEF2B_mut'
	profile_old_new_gene_mapping['C1orf86_mut'] = 'FAAP20_mut'
	profile_old_new_gene_mapping['C1ORF86'] = 'FAAP20'
	profile_old_new_gene_mapping['GNB2L1_mut'] = 'RACK1_mut'
	profile_old_new_gene_mapping['GNB2L1'] = 'RACK1'
	df_features_merged.rename(columns=profile_old_new_gene_mapping,
								   inplace=True)
	df_features_merged_ckps = (df_features_merged
							   .loc[set(df_features_merged.index) - set(cup_samples_ids)])
	df_features_merged_cups = (df_features_merged
							   .loc[set(df_features_merged.index) & set(cup_samples_ids)])
	df_labels = df_patients_id_indexed.loc[df_features_merged_ckps.index]['CANCER_TYPE']
	df_labels_final = pd.DataFrame([cancer_types.index(val) for val in df_labels.values],
								   columns=['cancer_label'], index=df_labels.index)
	df_labels_final['cancer_type'] = df_labels.values
	return df_features_merged_ckps, df_labels_final, df_features_merged_cups

def main(argv):
	# Loading GENIE data.
	df_patients_genie = pd.read_csv(os.path.join(_FILEPATH, "genie/data_clinical_patient_5.0-public.txt"), sep = '\t', comment='#')
	df_samples_genie = pd.read_csv(os.path.join(_FILEPATH, "genie/data_clinical_sample_5.0-public.txt"), sep = '\t', comment='#')

	# Choose panels from GENIE data.
	panels = ['MSK-IMPACT468', 'MSK-IMPACT410', 'MSK-IMPACT341', 'VICC-01-T7', 'VICC-01-T5A']
	df_samples_genie = df_samples_genie.loc[df_samples_genie.SEQ_ASSAY_ID.isin(panels)]
	df_genie_patients = pd.merge(df_patients_genie, df_samples_genie, how = 'right', on = 'PATIENT_ID')

	# Choose centers of interest : MSK, VICC
	df_genie_patients = df_genie_patients.loc[df_genie_patients.CENTER.isin(['MSK', 'VICC'])]
	# Load GENIE mutation data.
	df_mutations_genie = pd.read_csv(os.path.join(_FILEPATH, "genie/data_mutations_extended_5.0-public.txt"), sep = '\t', comment='#')
	# Load CNA data.
	df_cna_genie = pd.read_csv(os.path.join(_FILEPATH, "genie/data_CNA_5.0-public.txt"), sep = '\t', comment='#')
	df_cna_genie = df_cna_genie.set_index('Hugo_Symbol').T
	df_cna_genie.rename(columns = {'Hugo_Symbol' : 'Tumor_Sample_Barcode'}, inplace = True)
	# Load data for GENIE, contraining 96 single base substitutions in a trinucleotide context;
	# Used deconstructSigs (see deconstructSigs_trinucs_data.R)
	df_trinuc_feats_genie = pd.read_csv(os.path.join(_FILEPATH, "genie/trinucs_genie_msk_vicc.csv"), sep = ',')
	df_trinuc_feats_genie.set_index('Unnamed: 0', inplace = True)
	# Load Profile DFCI patients and samples info.
	df_samples_dfci = pd.read_csv(os.path.join(_FILEPATH, "dfci/profile_samples_info"), sep = '\t', comment='#')

	# Load Profile DFCI mutation and CNA data.
	df_mutations_dfci = pd.read_csv(os.path.join(_FILEPATH, "dfci/profile_mutation_dfci"), sep = '\t', comment='#')
	# Drop NaN canonical genes.
	df_mutations_dfci.dropna(subset=['CANONICAL_GENE'], inplace=True)
	df_mutations_dfci['UNIQUE_SAMPLE_ID'] = [float(idx[:-3]) for idx in df_mutations_dfci.UNIQUE_SAMPLE_ID.values]
	df_cna_dfci = pd.read_csv(os.path.join(_FILEPATH, "dfci/profile_cnv"), sep = '\t', comment='#')
	df_cna_dfci.set_index('Unnamed: 0', inplace = True)

	# Load data for Profile DFCI, contraining 96 single base substitutions in a trinucleotide context
	# Used deconstructSigs (see deconstructSigs_trinucs_data.R)
	df_trinuc_feats_dfci = pd.read_csv(os.path.join(_FILEPATH, "dfci/trinucs_dfci_profile.csv"), comment='#')
	df_trinuc_feats_dfci.set_index('Unnamed: 0', inplace = True)
	df_trinuc_feats_dfci.index = [float(idx[:-3]) for idx in df_trinuc_feats_dfci.index]		

	# Get mutation signature features based on tri-nucleotide data.
	df_mut_sigs_dfci = obtain_mutation_signatures(df_trinuc_feats_dfci)
	df_mut_sigs_genie = obtain_mutation_signatures(df_trinuc_feats_genie)

	# Define CUP subtypes.
	cup_subtypes = ['Undifferentiated Malignant Neoplasm',
					 'Poorly Differentiated Carcinoma, NOS',
					 'Acinar Cell Carcinoma, NOS',
					 'Adenocarcinoma, NOS',
					 'Cancer of Unknown Primary, NOS',
					 'Small Cell Carcinoma of Unknown Primary',
					 'Neuroendocrine Tumor, NOS',
					 'Squamous Cell Carcinoma, NOS',
					 'Cancer of Unknown Primary',
					 'Mixed Cancer Types',
					 'Neuroendocrine Carcinoma, NOS']
	df_cup_samples_dfci = df_samples_dfci.loc[df_samples_dfci.PRIMARY_CANCER_DIAGNOSIS.isin(cup_subtypes)]
	df_cup_samples_dfci.CANCER_TYPE = 'Cancer of Unknown Primary'
		
	# Get OncoTree defined cancer types to include.
	with open('../data/cancer_type_to_oncotree_subtypes_dict.pkl', 'rb') as handle:
		cancers_to_include_dic = pickle.load(handle)

	# Get total detailed cancer types under consideration in GENIE project.
	total_detailed_cancers_genie = []
	for _, detailed_cancers in cancers_to_include_dic.items():
		total_detailed_cancers_genie = total_detailed_cancers_genie + list(detailed_cancers)

	# Data pre-processing for AACR GENIE.	
	df_genie_patients_chosen = df_genie_patients.loc[df_genie_patients.CANCER_TYPE_DETAILED.isin(total_detailed_cancers_genie)]
	cancer_count = 0
	for cancer_key, detailed_cancers in cancers_to_include_dic.items():
		if cancer_count == 0:
			df_genie_patients_partial = df_genie_patients_chosen.loc[df_genie_patients_chosen.CANCER_TYPE_DETAILED.isin(detailed_cancers)]
			if len(detailed_cancers) > 1:
				df_genie_patients_partial['CANCER_TYPE'] = cancer_key
			else:  # If cancer type consists of only one detailed cancer type, just include the detailed cacner type as main cancer type
				df_genie_patients_partial['CANCER_TYPE'] = detailed_cancers[0]
			df_genie_patients_final = df_genie_patients_partial
		else:
			df_genie_patients_partial = df_genie_patients_chosen.loc[df_genie_patients_chosen.CANCER_TYPE_DETAILED.isin(detailed_cancers)]
			if len(detailed_cancers) > 1:
				df_genie_patients_partial['CANCER_TYPE'] = cancer_key
			else:
				df_genie_patients_partial['CANCER_TYPE'] = detailed_cancers[0]
			df_genie_patients_final = pd.concat([df_genie_patients_final, df_genie_patients_partial])
		cancer_count = cancer_count + 1
		
	# Get final cancer types of interest.
	cancer_types_final = list(np.unique(df_genie_patients_final.CANCER_TYPE.values))

	# Data pre-processing for DFCI cancers.
	# Choose samples based on GENIE-derived cancer types.
	df_samples_chosen_dfci = df_samples_dfci.loc[df_samples_dfci.PRIMARY_CANCER_DIAGNOSIS.isin(total_detailed_cancers_genie)]

	cancer_count = 0
	for cancer_key, detailed_cancers in cancers_to_include_dic.items():
		if cancer_count == 0:
			df_dfci_samples_partial = df_samples_chosen_dfci.loc[df_samples_chosen_dfci.PRIMARY_CANCER_DIAGNOSIS.isin(detailed_cancers)]
			if len(detailed_cancers) > 1:
				df_dfci_samples_partial['CANCER_TYPE'] = cancer_key
			else:
				df_dfci_samples_partial['CANCER_TYPE'] = detailed_cancers[0]
			df_dfci_samples_final = df_dfci_samples_partial
		else:
			df_dfci_samples_partial = df_samples_chosen_dfci.loc[df_samples_chosen_dfci.PRIMARY_CANCER_DIAGNOSIS.isin(detailed_cancers)]
			if len(detailed_cancers) > 1:
				df_dfci_samples_partial['CANCER_TYPE'] = cancer_key
			else:
				df_dfci_samples_partial['CANCER_TYPE'] = detailed_cancers[0]
			df_dfci_samples_final = pd.concat([df_dfci_samples_final, df_dfci_samples_partial])
		cancer_count = cancer_count + 1	
	# Process CUP samples.
	df_dfci_samples_cups = df_samples_dfci.loc[df_samples_dfci.UNIQUE_SAMPLE_ID.isin(df_cup_samples_dfci.UNIQUE_SAMPLE_ID.values)]
	df_dfci_samples_cups['CANCER_TYPE'] = 'Cancer of Unknown Primary'

	# Get final cancer types of interest.
	df_cancer_counts_dfci = pd.value_counts(df_dfci_samples_final.CANCER_TYPE.values, sort = True)
	print('	PROFILE cancer counts')
	print(df_cancer_counts_dfci)

	# Display cancer counts acorss each center and make sure each center has consistent cancer type
	cancer_counts_genie = pd.value_counts(df_genie_patients_final.CANCER_TYPE.values, sort = True)
	print('	GENIE cancer counts')
	print(cancer_counts_genie) 

	# Cancer names should be consistent.
	assert set(cancer_counts_genie.index) == set(df_cancer_counts_dfci.index)

	print('	Feature preparation continues...')
	# GENIE somatic feature processing.
	df_features_genie, df_labels_genie = pre_process_features_genie(df_mutations_genie,
																	df_cna_genie,
																	df_mut_sigs_genie,
																	df_genie_patients_final,
																	cancer_types_final)

	# Make data_patients_new_total consistent with the pipeline.
	df_samples_dfci_total = pd.concat([df_dfci_samples_final, df_cup_samples_dfci])
	# Change column names to be consistent.
	df_samples_dfci_total.rename(columns={"age_at_seq": "Age", "gender": "Sex"}, inplace=True)
	(df_features_dfci,
	 df_labels_dfci,
	 df_features_cup) = pre_process_features_dfci(df_mutations_dfci,
											   df_cna_dfci,
											   df_mut_sigs_dfci,
											   df_samples_dfci_total, 
											   cancer_types_final,
											   cup_samples_ids=df_cup_samples_dfci.UNIQUE_SAMPLE_ID.values)
	
	print('	Exporting features and labels of interest...')
	# Load onconpc_processed_cups_data
	# df_features_cups = pd.read_csv('data/onconpc_processed_cups_data', sep = '\t', index_col = 0)
	common_features = set(df_features_genie.columns) & set(df_features_dfci.columns)
	df_features_combined = pd.concat([df_features_genie[common_features], df_features_dfci[common_features]])
	df_labels_combined = pd.concat([df_labels_genie, df_labels_dfci])
	df_features_cup = df_features_cup[common_features]
	# Export common features
	common_features_df = pd.DataFrame(common_features,
									  index=np.arange(len(common_features)),
									  columns=['common_features'])
	common_features_df.to_csv('../data/common_features_df', sep = '\t', index = False)
	# Export combined (Profile & GENIE) features and labels.
	df_features_combined.to_csv(f'../data/features_combined_{_FILENAME_SUFFIX.value}', sep='\t', index=True)
	df_labels_combined.to_csv(f'../data/labels_combined_{_FILENAME_SUFFIX.value}', sep='\t', index=True)
	df_features_cup.to_csv(f'../data/features_combined_cup_{_FILENAME_SUFFIX.value}', sep='\t', index=True)
	return

if __name__ == '__main__':
  app.run(main)