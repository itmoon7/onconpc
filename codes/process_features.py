"""
Author: Intae Moon
The following script provides features and labels for a supervised training of classifying tumors with known primaries based on samples'
1. Clinical information (e.g. age and sex)
2. Somatic mutation profile (i.e. frequence of mutated genes)
3. Somatic copy number alteration events (i.e. genes associated with CNA : -2 (deep loss), -1 (single-copy loss), 0 (diploid), 1 (low-level gain), 2 (high-level amplification))
4. Somatic mutation signatures.
"""
import os
import pickle
from typing import List, Mapping, Tuple

import numpy as np
import pandas as pd
from absl import app, flags

import utils

FLAGS = flags.FLAGS
_FILEPATH = '../data/seq_panel_data'
flags.DEFINE_string('config', None, 'Cancer centers to incorporate for feature processing (genie, profile_dfci, or both)')
flags.DEFINE_string('filename_suffix', None, 'Filename suffix for processed files.')

def load_genie_data(filepath: str,
					panels: List[str],
					centers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	Load GENIE data.
	Args:
		filepath: Path to GENIE data.
		panels: List of panels to include.
		centers: List of centers to include.
	Returns:
		df_patients: DataFrame containing patient data.
		df_mutations: DataFrame containing mutation data.
		df_cna: DataFrame containing CNA data.
	"""
	df_patients = pd.read_csv(os.path.join(filepath, "genie/data_clinical_patient_5.0-public.txt"), sep='\t', comment='#')
	df_samples = pd.read_csv(os.path.join(filepath, "genie/data_clinical_sample_5.0-public.txt"), sep='\t', comment='#')
	df_samples = df_samples[df_samples.SEQ_ASSAY_ID.isin(panels)]
	df_patients = pd.merge(df_patients, df_samples, how='right', on='PATIENT_ID')
	df_patients = df_patients[df_patients.CENTER.isin(centers)]

	df_mutations = pd.read_csv(os.path.join(filepath, "genie/data_mutations_extended_5.0-public.txt"), sep='\t',
							comment='#', low_memory=False)
	df_mutations = df_mutations[df_mutations.Center.isin(centers)]
	df_cna = pd.read_csv(os.path.join(filepath, "genie/data_CNA_5.0-public.txt"), sep='\t', comment='#')
	df_cna = df_cna.set_index('Hugo_Symbol').T
	df_cna.index.name = 'Tumor_Sample_Barcode'
	return df_patients, df_mutations, df_cna

def load_dfci_data(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	Load Profile DFCI data.
	Args:
		filepath: Path to Profile DFCI data.
	Returns:
		df_samples: DataFrame containing sample data.
		df_mutations: DataFrame containing mutation data.
		df_cna: DataFrame containing CNA data.
	"""
	df_samples = pd.read_csv(os.path.join(filepath, "dfci/profile_samples_info"), sep='\t', comment='#')
	df_mutations = pd.read_csv(os.path.join(filepath, "dfci/profile_mutation_dfci"), sep='\t',
							comment='#', low_memory=False)
	df_mutations.dropna(subset=['CANONICAL_GENE'], inplace=True)
	df_mutations['UNIQUE_SAMPLE_ID'] = [float(idx[:-3]) for idx in df_mutations.UNIQUE_SAMPLE_ID.values]

	df_cna = pd.read_csv(os.path.join(filepath, "dfci/profile_cnv"), sep='\t', comment='#').set_index('Unnamed: 0')
	return df_samples, df_mutations, df_cna

def process_patient_data(df: pd.DataFrame,
						 cancers_to_include_dic: Mapping[str, List[str]],
						 cancer_type_column: str,
						 new_cancer_type_column: str='CANCER_TYPE') -> pd.DataFrame:
	"""
	Get relevant patient info DataFrame based on cancer types to include.
	Args:
		df: DataFrame containing patient data.
		cancers_to_include_dic: Dictionary containing cancer types to include.
		cancer_type_column: Column name for cancer type.
		new_cancer_type_column: New column name for cancer type.
	Returns:
		df_final: DataFrame containing processed patient data.
	"""
	df_final = pd.DataFrame()
	for cancer_key, detailed_cancers in cancers_to_include_dic.items():
		df_partial = df[df[cancer_type_column].isin(detailed_cancers)].copy()
		# If cancer type consists of only one detailed cancer type, just include the detailed cacner type as main cancer type
		df_partial[new_cancer_type_column] = cancer_key if len(detailed_cancers) > 1 else detailed_cancers[0]
		df_final = pd.concat([df_final, df_partial])
	return df_final

def combine_feature_dfs(df_list: List[pd.DataFrame],
						feature_list: List[str]) -> pd.DataFrame:
	"""
	Combine feature dfs.
	Args:
		df_list: List of feature dfs.
		feature_list: List of features to include.
	Returns:
		df: DataFrame containing combined features.
	"""
	df = pd.DataFrame()
	for df_partial in df_list:
		df_partial = utils.zero_pad_missing_features(df_partial, feature_list)
		df = pd.concat([df, df_partial])
	return df

def main(argv):
	# Get flags.
	FLAGS = flags.FLAGS
	config = FLAGS.config
	filename_suffix = FLAGS.filename_suffix
	print('Loading data...')
	if config == 'genie':
		panels = ['DFCI-ONCOPANEL-1', 'DFCI-ONCOPANEL-2', 'DFCI-ONCOPANEL-3',
			'MSK-IMPACT468', 'MSK-IMPACT410', 'MSK-IMPACT341', 'VICC-01-T7', 'VICC-01-T5A']
		centers = ['DFCI', 'MSK', 'VICC']
		df_genie_patients, df_mutations_genie, df_cna_genie = load_genie_data(_FILEPATH, panels, centers)
		print('Obtaining mutation signatures via R function...')
		df_trinuc_feats_genie = utils.get_snv_in_trinuc_context(df_mutations_genie,
														 sample_id_col='Tumor_Sample_Barcode',
														 chromosome_col='Chromosome',
														 start_pos_col='Start_Position',
														 ref_allele_col='Reference_Allele',
														 alt_allele_col='Tumor_Seq_Allele2',
														 config=config)
		df_mut_sigs_genie = utils.obtain_mutation_signatures(df_trinuc_feats_genie)
	elif config == 'profile_dfci':
		df_samples_dfci, df_mutations_dfci, df_cna_dfci = load_dfci_data(_FILEPATH)
		print('Obtaining mutation signatures via R function...')
		df_trinuc_feats_dfci = utils.get_snv_in_trinuc_context(df_mutations_dfci,
														 sample_id_col='UNIQUE_SAMPLE_ID',
														 chromosome_col='CHROMOSOME',
														 start_pos_col='POSITION',
														 ref_allele_col='REF_ALLELE',
														 alt_allele_col='ALT_ALLELE',
														 config=config)
		df_mut_sigs_dfci = utils.obtain_mutation_signatures(df_trinuc_feats_dfci)
	elif config == 'both':
		# In this config, remove DFCI samples in GENIE, since we are directly using Profile DFCI data.
		panels = ['MSK-IMPACT468', 'MSK-IMPACT410', 'MSK-IMPACT341', 'VICC-01-T7', 'VICC-01-T5A']
		centers = ['MSK', 'VICC']
		df_genie_patients, df_mutations_genie, df_cna_genie = load_genie_data(_FILEPATH, panels, centers)
		df_samples_dfci, df_mutations_dfci, df_cna_dfci = load_dfci_data(_FILEPATH)
		print('Obtaining mutation signatures via R function...')
		df_trinuc_feats_genie = utils.get_snv_in_trinuc_context(df_mutations_genie,
														 sample_id_col='Tumor_Sample_Barcode',
														 chromosome_col='Chromosome',
														 start_pos_col='Start_Position',
														 ref_allele_col='Reference_Allele',
														 alt_allele_col='Tumor_Seq_Allele2',
														 config='genie')
		df_trinuc_feats_dfci = utils.get_snv_in_trinuc_context(df_mutations_dfci,
														 sample_id_col='UNIQUE_SAMPLE_ID',
														 chromosome_col='CHROMOSOME',
														 start_pos_col='POSITION',
														 ref_allele_col='REF_ALLELE',
														 alt_allele_col='ALT_ALLELE',
														 config='profile_dfci')
		df_mut_sigs_dfci = utils.obtain_mutation_signatures(df_trinuc_feats_dfci)
		df_mut_sigs_genie = utils.obtain_mutation_signatures(df_trinuc_feats_genie)
	else:
		raise ValueError('Invalid config.')
	print('Processing data...')
	if config in ['profile_dfci', 'both']:
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
		df_cup_samples_dfci = df_samples_dfci.loc[df_samples_dfci.PRIMARY_CANCER_DIAGNOSIS.isin(cup_subtypes)].copy()
		df_cup_samples_dfci.CANCER_TYPE = 'Cancer of Unknown Primary'
	# Get OncoTree-based cancer types to include.
	with open('../data/cancer_type_to_oncotree_subtypes_dict.pkl', 'rb') as handle:
		cancers_to_include_dic = pickle.load(handle)
	if config in ['genie', 'both']:
		# Process patient data based on cancer types to include.
		df_genie_patients_final = process_patient_data(df_genie_patients, cancers_to_include_dic, cancer_type_column='CANCER_TYPE_DETAILED')
		# Get final cancer types.
		cancer_types_final = list(np.unique(df_genie_patients_final.CANCER_TYPE.values))
		cancer_counts_genie = df_genie_patients_final.CANCER_TYPE.value_counts(sort=True)
		print('GENIE cancer counts')
		print(cancer_counts_genie) 
		# GENIE somatic feature processing.
		df_features_genie, df_labels_genie = utils.pre_process_features_genie(df_mutations_genie,
																		df_cna_genie,
																		df_mut_sigs_genie,
																		df_genie_patients_final,
																		cancer_types=cancer_types_final)
	if config in ['profile_dfci', 'both']:
		# Process patient data based on cancer types to include.	
		df_dfci_samples_final = process_patient_data(df_samples_dfci, cancers_to_include_dic, cancer_type_column='PRIMARY_CANCER_DIAGNOSIS')
		if config == 'profile_dfci':
			# Get final cancer types
			cancer_types_final = list(np.unique(df_dfci_samples_final.CANCER_TYPE.values))
		cancer_counts_dfci = df_dfci_samples_final.CANCER_TYPE.value_counts(sort=True)
		print('PROFILE DFCI cancer counts')
		print(cancer_counts_dfci)
		# DFCI somatic feature processing.
		df_samples_dfci_total = pd.concat([df_dfci_samples_final, df_cup_samples_dfci])
		# Change column names to match GENIE data.
		df_samples_dfci_total.rename(columns={"age_at_seq": "Age", "gender": "Sex"}, inplace=True)
		(df_features_dfci,
		df_labels_dfci,
		df_features_cup) = utils.pre_process_features_dfci(df_mutations_dfci,
													 df_cna_dfci,
													 df_mut_sigs_dfci,
													 df_samples_dfci_total, 
													 cancer_types_final,
													 cup_samples_ids=df_cup_samples_dfci.UNIQUE_SAMPLE_ID.values)
	print('Exporting features and labels of interest...')
	# Load features onconpc pickle file
	with open('../data/features_onconpc.pkl', "rb") as fp:
		features_onconpc = pickle.load(fp)
	if config == 'both':
		# Combine features and labels across centers.
		df_features_combined = combine_feature_dfs([df_features_genie, df_features_dfci], features_onconpc)
		df_features_combined.columns = utils.standardize_feat_names(df_features_combined.columns)
		df_labels_combined = pd.concat([df_labels_genie, df_labels_dfci])
		df_features_cup = df_features_cup[features_onconpc]
		df_features_cup.columns = utils.standardize_feat_names(df_features_cup.columns)
		# Export combined (Profile & GENIE) features and labels.
		df_features_combined.to_csv(f'../data/features_combined_{filename_suffix}', sep='\t', index=True)
		df_labels_combined.to_csv(f'../data/labels_combined_{filename_suffix}', sep='\t', index=True)
		df_features_cup.to_csv(f'../data/features_combined_cup_{filename_suffix}', sep='\t', index=True)
	elif config == 'genie':
		df_features_genie = utils.zero_pad_missing_features(df_features_genie, features_onconpc)
		df_features_genie.columns = utils.standardize_feat_names(df_features_genie.columns)
		df_features_genie.to_csv(f'../data/features_genie_{filename_suffix}', sep='\t', index=True)
		df_labels_genie.to_csv(f'../data/labels_genie_{filename_suffix}', sep='\t', index=True)
	elif config == 'profile_dfci':
		df_features_dfci.columns = utils.standardize_feat_names(df_features_dfci.columns)
		df_features_cup.columns = utils.standardize_feat_names(df_features_cup.columns)
		df_features_dfci.to_csv(f'../data/features_dfci_{filename_suffix}', sep='\t', index=True)
		df_labels_dfci.to_csv(f'../data/labels_dfci_{filename_suffix}', sep='\t', index=True)
		df_features_cup.to_csv(f'../data/features_dfci_cup_{filename_suffix}', sep='\t', index=True)
	return

if __name__ == '__main__':
  	# Define flags that are required
	flags.mark_flags_as_required(['config',
								'filename_suffix'])
	app.run(main)