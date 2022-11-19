"""
Author: Intae Moon

Date : July 6th, 2020
Description : 
The following script provides features and labels for a supervised training of classifying tumors with known primaries based on samples'
1. clinical information (e.g. age and gender)
2. somatic mutation profile (i.e. frequence of mutated genes)
3. copy number alteration events (i.e. genes associated with CNA : -2 (deep loss), -1 (single-copy loss), 0 (diploid), 1 (low-level gain), 2 (high-level amplification))

As outputs, an user has an option of obtaining processed features and labels (for non-CUP only):
1. GENIE data
2. PROFILE data (DFCI)
3. PROFILE CUP (Cancer of Unknown Primaries)s
4. PROFILE CUP (Cancer of Unknown Primaries)s compatible with features of GENIE data

Modification :
1. (July 6th, 2020) Ensures that CUP feature dataframe contains no NaN entries
2. (Oct 24th, 2020) Featurize linear mapping of single nucleotide changes in a tri-nucleotide context into mutation signatures

Change to be made :
1. Makes sure column names are alphabetically ordered...
"""

from absl import app
from absl import flags

import glob
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

# import warnings
# warnings.filterwarnings("ignore", message="Numerical issues were encountered ")
FLAGS = flags.FLAGS
flags.DEFINE_string('name', 'Jane Random', 'Your name.')
flags.DEFINE_integer('age', None, 'Your age in years.', lower_bound=0)
flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
flags.DEFINE_enum('job', 'running', ['running', 'stopped'], 'Job status.')


def get_cancer_type_to_detailed_cancer_types_dic():
	# build the dictionary based on pre-specified cancer types and detailed cancer types
	# this dic is based on AACR values
	# cancers_to_include = pd.read_csv('cancer_types_July_16th_treatment_clf', sep ='\t')
	cancers_to_include = pd.read_csv('cancer_types_to_include_Oct_24th_2020', sep ='\t')
	cancers_to_include.dropna(how = 'all', inplace = True)
	cancers_to_include.set_index('cancer', inplace = True)
	cancers_to_include_dic = {}
	for cancer, i in zip(cancers_to_include.index, np.arange(len(cancers_to_include))):
		cancer_list_oi = cancers_to_include.iloc[i].values
		cancer_list_oi = [elem for elem in cancer_list_oi if str(elem) != 'nan']
		cancers_to_include_dic.update({cancer:cancer_list_oi})

	# create 
		
	cancer_types_oi = list(cancers_to_include_dic.keys())


def get_sv_feats(df_profile_sv):
	# need to work on that more 
	sv_vals = []; sv_vals_set_list = []
	df_sv_feats = pd.DataFrame([])#, index = set(df_profile_sv.UNIQUE_SAMPLE_ID.values))#, columns = chosen_sv_feats)
	df_sv_feats['sample_id'] = list(set(df_profile_sv.UNIQUE_SAMPLE_ID.values))
	df_sv_feats.set_index('sample_id', inplace = True)
	for sample_id, l_gene, r_gene, sv_type in zip(df_profile_sv.UNIQUE_SAMPLE_ID.values, df_profile_sv.L_GENE.values, df_profile_sv.R_GENE.values, df_profile_sv.SV_TYPE.values):
		if not pd.isnull(l_gene) and not pd.isnull(r_gene):
			sv_vals.append(l_gene + '_' + r_gene)
			sv_vals_set = set([l_gene, r_gene])
			if str(sv_vals_set) + '_sv' not in df_sv_feats.columns:
				df_sv_feats[str(sv_vals_set) + '_sv'] = 0
				df_sv_feats.at[sample_id, str(sv_vals_set) + '_sv'] = 1
			else:
				df_sv_feats.at[sample_id, str(sv_vals_set) + '_sv'] = df_sv_feats.at[sample_id, str(sv_vals_set) + '_sv'] + 1

			if sv_type == 'Rearrangement_inversion':
				sv_vals_set_type = str(sv_vals_set) + '_Rearrangement' + '_sv'
			else:
				sv_vals_set_type = str(sv_vals_set) + '_' + sv_type + '_sv'

			if sv_vals_set_type not in df_sv_feats.columns:
				df_sv_feats[sv_vals_set_type] = 0
				df_sv_feats.at[sample_id, sv_vals_set_type] = 1
			else:
				df_sv_feats.at[sample_id, sv_vals_set_type] = df_sv_feats.at[sample_id, sv_vals_set_type] + 1
	# sv_counts_stats = pd.value_counts(sv_vals, sort = True)
	sv_set_counts_stats = pd.value_counts(sv_vals_set_list, sort = True)
	return df_sv_feats
	# breakpoint()
	# chosen_sv_feats = sv_counts_stats.loc[sv_counts_stats.values > 0].index

	# # create sv features df and fill out the df
	# df_sv_feats = pd.DataFrame(0.0, index = set(df_profile_sv.UNIQUE_SAMPLE_ID.values), columns = chosen_sv_feats)
	# for sample_id, l_gene, r_gene in zip(df_profile_sv.UNIQUE_SAMPLE_ID.values, df_profile_sv.L_GENE.values, df_profile_sv.R_GENE.values):
	# 	if not pd.isnull(l_gene) and not pd.isnull(r_gene):
	# 		feat_oi = l_gene + '_' + r_gene
	# 		if feat_oi in chosen_sv_feats:
	# 			df_sv_feats.at[sample_id, feat_oi] = df_sv_feats.at[sample_id, feat_oi] + 1
	# return df_sv_feats

def obtain_mutation_signatures(tri_features):
	"""Transforms tri-nucleotide features into mutation signature based features.
	"""
	# Load Mutational Signatures COSMIC data.
	file_path = '../analysis/features_and_labels/mutation_signatures/'
	file_names = glob.glob(file_path + 'sigProfiler*.csv')
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
	mut_sig_based_df = pd.DataFrame([], index = tri_features.index)
	for key, mut_df in sbs_conversion_df_dict.items():
		common_substitute_feats = set(mut_df.index) & set(tri_features.columns)
		if len(common_substitute_feats) != 96:
			raise ValueError('Number of substitution features does not sum up to 96.')
		# Get mutation signature values
		mut_sig_vals = np.matmul(tri_features[common_substitute_feats].values, mut_df.loc[common_substitute_feats].values_oi.values)
		mut_sig_based_df[key] = mut_sig_vals
	return mut_sig_based_df

def merge_dfs(df_list, ids_common):
	merge_count = 0
	for df in df_list:
		if merge_count == 0:
			df_merged_to_return = df.loc[ids_common]
		else:
			df_merged_to_return = pd.merge(df_merged_to_return, df.loc[ids_common], how = 'left', left_index = True, right_index = True)
		merge_count = merge_count + 1
	return df_merged_to_return

def process_genetics_features_genie(data_mutations, data_cna_t, data_trinuc, data_patients, cancer_types, 
									cup_samples_ids = None, AACR_only = False, profile_only = False, cup_only = False):
	# obtain mutation features
	if AACR_only:
		sample_ids_oi = data_patients.SAMPLE_ID.values
	elif profile_only or cup_only:
		sample_ids_oi = data_patients.Tumor_Sample_Barcode.values
	data_mutations_oi = data_mutations.loc[data_mutations.Tumor_Sample_Barcode.isin(sample_ids_oi)]
	# add 'mut' at the end of gene mutation feature
	col_names = np.unique(data_mutations_oi.Hugo_Symbol.values)
	col_names_fixed = [val + '_mut' for val in col_names]
	mutation_feature_df = pd.DataFrame(0, columns = col_names, index = sample_ids_oi)
	for entry in data_mutations_oi[['Tumor_Sample_Barcode', 'Hugo_Symbol']].values:
		sample_id = entry[0]; gene = entry[1];
		if gene in mutation_feature_df.columns:
			mutation_feature_df.at[sample_id, gene] = mutation_feature_df.at[sample_id, gene] + 1
	mutation_feature_df.columns = col_names_fixed
		
	# obatin cna features
	cna_feature = data_cna_t.loc[set(data_cna_t.index) & set(sample_ids_oi)]
	# breakpoint()
	cna_feature.fillna(0, inplace = True)

	# obtain tri-nucleotide features
	trinuc_feature = data_trinuc.copy()

	# obtain age/gender features
	if AACR_only:
		data_patients_indexed = data_patients.set_index('SAMPLE_ID')
		age_gender_feature = data_patients_indexed[['SEX', 'AGE_AT_SEQ_REPORT']]
		gender_list = []
		for entry in age_gender_feature.values:
			gender = entry[0];
			if gender == 'Male':
				gender_list.append(1)
			elif gender == 'Female':
				gender_list.append(-1)
			else:
				gender_list.append(0)
		age_gender_feature['SEX'] = gender_list
		age_val_list = age_gender_feature['AGE_AT_SEQ_REPORT'].values
		age_nan_ids = []; age_int_converted = [] 
		for idx, val in zip(age_gender_feature.index, age_val_list):
			if pd.isnull(val):
				age_nan_ids.append(idx)
				age_int_converted.append(0) # append zero for now
			elif '>89' in val:
				age_int_converted.append(int(val[1:]))
			elif '<18' in val:
				age_int_converted.append(int(val[1:]))
			else:
				age_int_converted.append(int(val))
		age_val_array = np.asarray(age_int_converted)
		age_gender_feature['AGE_AT_SEQ_REPORT'] = (age_val_array - np.mean(age_val_array))/np.std(age_val_array)
		age_gender_feature.columns = ['GENDER', 'AGE']
	elif profile_only or cup_only: # profile only
		data_patients_indexed = data_patients.set_index('Tumor_Sample_Barcode')
		age_gender_feature = data_patients_indexed[['GENDER', 'AGE']]
		gender_list = []
		for entry in age_gender_feature.values:
			gender = entry[0];
			if gender == 'Male':
				gender_list.append(1)
			elif gender == 'Female':
				gender_list.append(-1)
			else:
				gender_list.append(0)
		age_gender_feature['GENDER'] = gender_list
		age_val_list = age_gender_feature['AGE'].values
		age_nan_ids = []; age_int_converted = [] 
		for idx, val in zip(age_gender_feature.index, age_val_list):
			if pd.isnull(val):
				age_nan_ids.append(idx)
				age_int_converted.append(0) # append zero for now
			else:
				age_int_converted.append(int(val))
		age_val_array = np.asarray(age_int_converted)
		age_gender_feature['AGE'] = (age_val_array - np.mean(age_val_array))/np.std(age_val_array)
	# merge all features using Sample IDs
	common_ids = set(mutation_feature_df.index) & set(cna_feature.index) & set(trinuc_feature.index) & set(age_gender_feature.index)
	# merge four different features 
	df_list_features = [mutation_feature_df, cna_feature, trinuc_feature, age_gender_feature]
	merged_features = merge_dfs(df_list_features, common_ids)
	# print('total length of merged_features : ', len(merged_features))

	# exclude age NaN indices
	if profile_only or cup_only:
		merged_features_final = merged_features.loc[set(merged_features.index) - set(age_nan_ids) - set(cup_samples_ids)]
		print('total length of merged_features_final after removing CUPs : ', len(merged_features_final))
	else:
		merged_features_final = merged_features.loc[set(merged_features.index) - set(age_nan_ids)]

	if cup_only:
		return merged_features.loc[set(cup_samples_ids)]
	else:
		# prepare labels : 
		labels_raw = data_patients_indexed.loc[merged_features_final.index]['CANCER_TYPE']
		labels_final = pd.DataFrame([cancer_types.index(val) for val in labels_raw.values], columns = ['cancer_label'], index = labels_raw.index)
		labels_final['cancer_type'] = labels_raw.values
		return merged_features_final, labels_final	

def process_genetics_features_profile(data_mutations, data_cna_t, data_trinuc, data_patients, cancer_types, 
									  cup_samples_ids = None, return_cup = False):
	# obtain mutation features
	# if AACR_only:
	#     sample_ids_oi = data_patients.SAMPLE_ID.values
	# elif profile_only or cup_only:
	sample_ids_oi = data_patients.UNIQUE_SAMPLE_ID.values
	data_mutations_oi = data_mutations.loc[data_mutations.UNIQUE_SAMPLE_ID.isin(sample_ids_oi)]
	# add 'mut' at the end of gene mutation feature
	col_names = np.unique(data_mutations_oi.CANONICAL_GENE.values)
	col_names_fixed = [val + '_mut' for val in col_names]
	mutation_feature_df = pd.DataFrame(0, columns = col_names, index = sample_ids_oi)
	for entry in data_mutations_oi[['UNIQUE_SAMPLE_ID', 'CANONICAL_GENE']].values:
		sample_id = entry[0]; gene = entry[1];
		if gene in mutation_feature_df.columns:
			mutation_feature_df.at[sample_id, gene] = mutation_feature_df.at[sample_id, gene] + 1
	mutation_feature_df.columns = col_names_fixed
	# obatin cna features
	cna_feature = data_cna_t.loc[set(data_cna_t.index) & set(sample_ids_oi)]
	cna_feature.fillna(0, inplace = True)

	# obtain tri-nucleotide features
	trinuc_feature = data_trinuc.copy()

	# if profile_only or cup_only: # profile only
	data_patients_indexed = data_patients.set_index('UNIQUE_SAMPLE_ID')
	age_gender_feature = data_patients_indexed[['GENDER', 'AGE']]
	gender_list = []
	for entry in age_gender_feature.values:
		gender = entry[0];
		if pd.isnull(gender):
			gender_list.append(0)
		elif gender == 'MALE':
			gender_list.append(1)
		elif gender == 'FEMALE':
			gender_list.append(-1)
	age_gender_feature['GENDER'] = gender_list
	age_val_list = age_gender_feature['AGE'].values
	age_mean = age_gender_feature['AGE'].dropna().values
	age_nan_ids = []; age_int_converted = [] 
	for idx, val in zip(age_gender_feature.index, age_val_list):
	# print(val)
		if pd.isnull(val):
			age_nan_ids.append(idx)
			print('nan age : ', idx)
			age_int_converted.append(age_mean) # append zero for now
		else:
			age_int_converted.append(int(val))
	age_val_array = np.asarray(age_int_converted)
	age_gender_feature['AGE'] = (age_val_array - np.mean(age_val_array))/np.std(age_val_array)
	# merge all features using Sample IDs
	common_ids = set(mutation_feature_df.index) & set(cna_feature.index) & set(trinuc_feature.index) & set(age_gender_feature.index)
	# print('total length of common indices : ', len(common_ids))
	# merge four different features 
	df_list_features = [mutation_feature_df, cna_feature, trinuc_feature, age_gender_feature]
	merged_features = merge_dfs(df_list_features, common_ids)
	# exclude age NaN indices
	# if cup_only:
	merged_features_final = merged_features.loc[set(merged_features.index) - set(cup_samples_ids)]
	# else:
	#     merged_features_final = merged_features.loc[set(merged_features.index) - set(age_nan_ids)]

	## Make some of the gene names consistent across old and new data
	profile_old_new_gene_mapping = {}
	profile_old_new_gene_mapping['C17ORF70'] = 'FAAP100' 
	profile_old_new_gene_mapping['C17orf70_mut'] = 'FAAP100_mut'
	profile_old_new_gene_mapping['C19ORF40'] = 'FAAP24' 
	profile_old_new_gene_mapping['C19orf40_mut'] = 'FAAP24_mut'
	# profile_old_new_gene_mapping['C19ORF40'] = 'FAAP20' 
	# profile_old_new_gene_mapping['C19ORF40_mut'] = 'FAAP20_mut'
	profile_old_new_gene_mapping['LOC96610_mut'] = 'BMS1P20_mut'
	profile_old_new_gene_mapping['LOC729991-MEF2B_mut'] = 'MEF2BNB-MEF2B_mut'
	profile_old_new_gene_mapping['C1orf86_mut'] = 'FAAP20_mut'
	profile_old_new_gene_mapping['C1ORF86'] = 'FAAP20'

	profile_old_new_gene_mapping['GNB2L1_mut'] = 'RACK1_mut'
	profile_old_new_gene_mapping['GNB2L1'] = 'RACK1'


	new_columns = []
	for col in merged_features_final.columns:
		if col in profile_old_new_gene_mapping.keys():
			new_columns.append(profile_old_new_gene_mapping[col])
		else:
			new_columns.append(col)

	merged_features_final.columns = new_columns
	labels_raw = data_patients_indexed.loc[merged_features_final.index]['CANCER_TYPE']
	labels_final = pd.DataFrame([cancer_types.index(val) for val in labels_raw.values], columns = ['cancer_label'], index = labels_raw.index)
	labels_final['cancer_type'] = labels_raw.values
	if return_cup:
		merged_features.columns = new_columns
		return merged_features_final, labels_final, merged_features.loc[set(merged_features.index) & set(cup_samples_ids)]
	else:
		return merged_features_final, labels_final  

def main(argv):
	# Loading Profile DFCI data.
	df_patients = pd.read_csv("../analysis/data/profile/data_patients.txt", sep = '\t', comment='#')
	df_samples = pd.read_csv("../analysis/data/profile/data_samples_editted.txt", sep = '\t', comment='#')
	df_profile_patients = pd.merge(data_patients, data_samples, how = 'left', on = 'PATIENT_ID')
	df_profile_patients.rename(columns = {'SAMPLE_ID' : 'Tumor_Sample_Barcode'}, inplace = True)

	# Loading GENIE data.
	df_patients_genie = pd.read_csv("../analysis/data/genie/data_clinical_patient_5.0-public.txt", sep = '\t', comment='#')
	df_samples_genie = pd.read_csv("../analysis/data/genie/data_clinical_sample_5.0-public.txt", sep = '\t', comment='#')

	# Choose panels to choose from MSK and VICC from GENIE.
	panels = ['MSK-IMPACT468', 'MSK-IMPACT410', 'MSK-IMPACT341', 'VICC-01-T7', 'VICC-01-T5A']
	df_samples_genie = data_samples_genie.loc[data_samples_genie.SEQ_ASSAY_ID.isin(panels)]
	df_genie_patients = pd.merge(data_patients_genie, data_samples_genie, how = 'right', on = 'PATIENT_ID')

	# Choose centers of interest : MSK, VICC, WAKE, UCSF for now
	df_genie_patients = df_genie_patients.loc[df_genie_patients.CENTER.isin(['MSK', 'VICC'])]
	# Load GENIE mutation data.
	df_mutations_genie = pd.read_csv("../analysis/data/genie/data_mutations_extended_5.0-public.txt", sep = '\t', comment='#')
	# Load CNA data.
	df_cna_genie = pd.read_csv("../analysis/data/genie/data_CNA_5.0-public.txt", sep = '\t', comment='#')
	df_cna_genie = df_cna_genie.set_index('Hugo_Symbol').T
	df_cna_genie.rename(columns = {'Hugo_Symbol' : 'Tumor_Sample_Barcode'}, inplace = True)
	# Load data for GENIE, contraining 96 single base substitutions in a trinucleotide context; use deconstructSigs.
	df_trinuc_genie = pd.read_csv('data/genie/genie_MSK_UCSF_VICC_trinuc_6_3_21.csv', sep = ',')
	df_trinuc_genie.set_index('Unnamed: 0', inplace = True)
	# Load Profile DFCI patients and samples info.
	df_patients_dfci = pd.read_csv("../analysis/data/Profile_v2/profile_patient_info_3_2_21", sep = '\t', comment='#')
	df_samples_dfci = pd.read_csv("../analysis/data/Profile_v2/profile_samples_info_3_2_21", sep = '\t', comment='#')

	# Load Profile DFCI mutation and CNA data.
	df_mutations_dfci = pd.read_csv("../analysis/data/Profile_v2/profile_mutation_3_2_21", sep = '\t', comment='#')
	df_cna_dfci = pd.read_csv("../analysis/data/Profile_v2/profile_cnv_3_2_21", sep = '\t', comment='#')
	df_cna_dfci.set_index('Unnamed: 0', inplace = True)

	# Load data for Profile DFCI, contraining 96 single base substitutions in a trinucleotide context; use deconstructSigs.
	df_trinuc_feats_dfci = pd.read_csv("../analysis/data/Profile_v2/profile_trinuc_3_2_21.csv", comment='#')
	df_trinuc_feats_dfci.set_index('Unnamed: 0', inplace = True)
	df_trinuc_feats_dfci.index = [int(idx[:-3]) for idx in df_trinuc_feats_dfci.index]		

	# Get mutation signature features based on tri-nucleotide data.
	data_trinuc_feats_new = obtain_mutation_signatures(data_trinuc_feats_new)
	data_trinuc_genie = obtain_mutation_signatures(data_trinuc_genie)

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
	data_profile_patients_cup_new = data_samples_new.loc[data_samples_new.PRIMARY_CANCER_DIAGNOSIS.isin(cup_subtypes)]
	data_profile_patients_cup_new.CANCER_TYPE = 'Cancer of Unknown Primary'
		
	# Get OncoTree defined cancer types to include.
	with open('features_and_labels/cancer_type_to_oncotree_subtypes_dic.pkl', 'rb') as handle:
		cancers_to_include_dic = pickle.load(handle)
	cancer_types_oi = list(cancers_to_include_dic.keys())

	# Get total detailed cancer types under consideration in GENIE project.
	total_detailed_cancers_genie = []
	for cancer, detailed_cancers in cancers_to_include_dic.items():
		total_detailed_cancers_genie = total_detailed_cancers_genie + list(detailed_cancers)

	"""
	Pre-processing samples based on cancer types <br>
	This following scripts chooses cancer types where detailed cancer types underneath each chosen cancer types 
	have enough samples (e.g. selects detailed cancer types whose # of samples are > 100) in order to regulate the noises within the data.
	In addition, the script names of cancer classes consistent across GENIE and PROFILE
	"""

	## AACR GENIE
	# breakpoint()/
	if data_to_obtain == 'all':
		genie_patients_chosen = data_genie_patients.loc[data_genie_patients.CANCER_TYPE_DETAILED.isin(total_detailed_cancers_genie)]

		cancer_count = 0
		for cancer_key, cancer_vals in cancers_to_include_dic.items():
			if cancer_count == 0:
				data_genie_patients_partial = data_genie_patients_filtered_oi.loc[data_genie_patients_filtered_oi.CANCER_TYPE_DETAILED.isin(cancer_vals)]
				if len(cancer_vals) > 1:
					data_genie_patients_partial['CANCER_TYPE'] = cancer_key
				else: # if cancer type consists of only one detailed cancer type, just include 
					data_genie_patients_partial['CANCER_TYPE'] = cancer_vals[0]
				data_genie_patients_final = data_genie_patients_partial
			else:
				data_genie_patients_partial = data_genie_patients_filtered_oi.loc[data_genie_patients_filtered_oi.CANCER_TYPE_DETAILED.isin(cancer_vals)]
				if len(cancer_vals) > 1:
					data_genie_patients_partial['CANCER_TYPE'] = cancer_key
				else:
					data_genie_patients_partial['CANCER_TYPE'] = cancer_vals[0]
				data_genie_patients_final = pd.concat([data_genie_patients_final, data_genie_patients_partial])
			cancer_count = cancer_count + 1
			
		# get final cancer types of interest
		cancer_types_final = list(np.unique(data_genie_patients_final.CANCER_TYPE.values))

	## Profile new
	data_samples_new_filtered_oi = data_samples_new.loc[data_samples_new.PRIMARY_CANCER_DIAGNOSIS.isin(total_detailed_cancers_oi_genie)]

	cancer_count = 0
	for cancer_key, cancer_vals in cancers_to_include_dic.items():
		if cancer_count == 0:
			data_samples_new_partial = data_samples_new_filtered_oi.loc[data_samples_new_filtered_oi.PRIMARY_CANCER_DIAGNOSIS.isin(cancer_vals)]
			if len(cancer_vals) > 1:
				data_samples_new_partial['CANCER_TYPE'] = cancer_key
			else:
				data_samples_new_partial['CANCER_TYPE'] = cancer_vals[0]
			data_samples_new_final = data_samples_new_partial
		else:
			data_samples_new_partial = data_samples_new_filtered_oi.loc[data_samples_new_filtered_oi.PRIMARY_CANCER_DIAGNOSIS.isin(cancer_vals)]
			if len(cancer_vals) > 1:
				data_samples_new_partial['CANCER_TYPE'] = cancer_key
			else:
				data_samples_new_partial['CANCER_TYPE'] = cancer_vals[0]
			data_samples_new_final = pd.concat([data_samples_new_final, data_samples_new_partial])
		cancer_count = cancer_count + 1	
		# if 3458570.0 in data_samples_new_final.UNIQUE_SAMPLE_ID.values:
		# 	breakpoint()
	# Process CUP samples
	data_samples_cups = data_samples_new.loc[data_samples_new.UNIQUE_SAMPLE_ID.isin(data_profile_patients_cup_new.UNIQUE_SAMPLE_ID.values)]
	data_samples_cups['CANCER_TYPE'] = 'Cancer of Unknown Primary'
	data_samples_all = pd.concat([data_samples_new_final, data_samples_cups])

	# get final cancer types of interest
	cancer_types_final_profile_new = list(np.unique(data_samples_new_final.CANCER_TYPE.values))
	count_profile_patients_new = pd.value_counts(data_samples_new_final.CANCER_TYPE.values, sort = True)
	print('	PROFILE cancer counts')
	print(count_profile_patients_new)
	# breakpoint()

	if data_to_obtain == 'all':
		# Display cancer counts acorss each center and make sure each center has consistent cancer type
		counts_genie_patients = pd.value_counts(data_genie_patients_final.CANCER_TYPE.values, sort = True)
		print('	GENIE cancer counts')
		print(counts_genie_patients) 

		print('	cancer names consistent ?')
		print(set(counts_genie_patients.index) == set(count_profile_patients_new.index))

		print('	Feature preparation continues...')
		## genie
		features_genie, labels_genie = process_genetics_features_genie(data_mutations_genie, data_cna_genie_t, data_trinuc_genie, data_genie_patients_final, cancer_types_final, AACR_only = True)

		# Exclude some cancer types from GENIE which hurts overall performance of the classifier
		# cancer_types_to_exclude = {'Esophagogastric Cancer', 'Gastrointestinal Neuroendocrine Tumor', 
		# 					   'Head and Neck Carcinoma', 'Leukemia', 'Melanoma', 'Ovarian Cancer', 'CNS Cancer'}
		# cancer_types_to_exclude = set(['Prostate Adenocarcinoma'])
		cancer_types_to_exclude = set([])
		# cancer_types_to_include = {'Non-Small Cell Lung Cancer'}
		# print('cancer_types_to_include from GENIE : ', cancer_types_to_include)
		labels_genie_filtered = labels_genie.loc[labels_genie.cancer_type.isin(set(cancer_types_final) - cancer_types_to_exclude)]
		# labels_genie_filtered = labels_genie.loc[labels_genie.cancer_type.isin(cancer_types_to_include)]
		features_genie_filtered = features_genie.loc[labels_genie_filtered.index]

	## profile new
	# make sure you don't have any nan values in mutation feature
	non_nan_idx = []
	for idx, gene in zip(np.arange(len(data_mutations_new.CANONICAL_GENE.values)), data_mutations_new.CANONICAL_GENE.values):
		if pd.isnull(gene):
			continue
		else:
			non_nan_idx.append(idx)
	data_mutations_new_filtered = data_mutations_new.iloc[non_nan_idx]

	# make data_patients_new_total consistent with the pipeline
	data_patients_new_total = pd.concat([data_samples_new_final, data_profile_patients_cup_new])
	# modify column names associated with age or gender
	new_col_names = []
	for col in data_patients_new_total.columns:
		if 'age_at_seq' == col.lower():
			new_col_names.append('AGE')
		elif 'gender' in col.lower():
			new_col_names.append('GENDER')
		else:
			new_col_names.append(col)
	# data_patients_new_total.columns = ['PATIENT_ID', 'UNIQUE_SAMPLE_ID', 'PRIMARY_CANCER_DIAGNOSIS',
	# 								   'BIOPSY_SITE', 'BIOPSY_SITE_TYPE', 'ORIGINAL_PATH_DIAGNOSIS',
	# 								   'TUMOR_PURITY', 'TEST_ORDER_DT', 'D_TEST_ORDER_DT',
	# 								   'AGE90_TEST_ORDER_DT', 'REPORT_DT', 'D_REPORT_DT', 'AGE90_REPORT_DT',
	# 								   'CANCER_TYPE', 'MUTATIONAL_BURDEN', 'TOBACCO_SIGNATURE',
	# 								   'TEMOZOLOMIDE_SIGNATURE', 'POLE_SIGNATURE', 'APOBEC_SIGNATURE',
	# 								   'UVA_SIGNATURE', 'AGE', 'GENDER']
	data_patients_new_total.columns = new_col_names

	# breakpoint()
	# if data_to_obtain == 'profile' or data_to_obtain == 'all':
	if data_to_obtain == 'profile_cup_only' or data_to_obtain == 'all':
		features_profile_new, labels_profile_new, features_profile_cup = process_genetics_features_profile(data_mutations_new_filtered, data_cna_new, data_trinuc_feats_new, data_patients_new_total, 
																										   cancer_types_final_profile_new, cup_samples_ids = data_profile_patients_cup_new.UNIQUE_SAMPLE_ID.values, 
																										   return_cup = True)
	else:
		features_profile_new, labels_profile_new, features_profile_cup = process_genetics_features_profile(data_mutations_new_filtered, data_cna_new, data_trinuc_feats_new, data_patients_new_total, 
																											cancer_types_final_profile_new, cup_samples_ids = data_profile_patients_cup_new.UNIQUE_SAMPLE_ID.values, 
																											return_cup = True)
		if obtain_sv_feats:
			df_sv_counts_per_sample = df_sv_feats.sum(axis = 1)
			# CKP
			samples_oi = set(df_sv_counts_per_sample.loc[df_sv_counts_per_sample.values >= 1].index) & (set(features_profile_new.index) | set(features_profile_cup.index))
			df_sv_counts_per_sample_non_zero = df_sv_feats.loc[samples_oi]
			df_sv_sum = df_sv_counts_per_sample_non_zero.sum()
			sv_feats_oi = df_sv_sum.loc[df_sv_sum.values >= sv_feat_thresh].index
			df_sv_counts_per_sample_non_zero_filtered = df_sv_counts_per_sample_non_zero[sv_feats_oi]
			samples_oi_ckp = samples_oi & set(features_profile_new.index)
			for col in df_sv_counts_per_sample_non_zero_filtered.columns:
				features_profile_new[col] = 0.0
				features_profile_new.loc[samples_oi_ckp, col] = df_sv_counts_per_sample_non_zero_filtered.loc[samples_oi_ckp][col].values
			# CUP
			# samples_oi = set(df_sv_counts_per_sample.loc[df_sv_counts_per_sample.values >= 1].index) & set(features_profile_cup.index)
			# df_sv_counts_per_sample_non_zero = df_sv_feats.loc[samples_oi]
			samples_oi_cup = samples_oi & set(features_profile_cup.index)
			for col in df_sv_counts_per_sample_non_zero_filtered.columns:
				features_profile_cup[col] = 0.0
				features_profile_cup.loc[samples_oi_cup, col] = df_sv_counts_per_sample_non_zero_filtered.loc[samples_oi_cup][col].values
			# breakpoint()
			# pass
	# Obtain profile CUP features
	# features_profile_cup = process_genetics_features_v2(data_mutations_new_filtered, data_cna_new, data_trinuc_feats_new, data_patients_new_total, 
	# 													cancer_types_final_profile_new, cup_samples_ids = data_profile_patients_cup_new.UNIQUE_SAMPLE_ID.values, cup_only = True)
	print('	Exporting features and labels of interest...')
	path = 'features_and_labels' #input("Enter the directory you would like to export the files to : ")
	suffix = input("Enter the suffix you would like to add to the exported file names : ")
	# breakpoint()
	# breakpoint()
	if data_to_obtain == 'all':
		# Obtain features
		common_features = set(features_genie.columns) & set(features_profile_new.columns)
		features_combined_filtered = pd.concat([features_genie_filtered[common_features], features_profile_new[common_features]])
		labels_combined_filtered = pd.concat([labels_genie_filtered, labels_profile_new])
		features_combined_cup = features_profile_cup[common_features]

		if obtain_sv_feats:
			# fix this part!!
			df_sv_counts_per_sample = df_sv_feats_total.sum(axis = 1)
			# CKP
			# samples_oi = set(df_sv_counts_per_sample.loc[df_sv_counts_per_sample.values >= 1].index) & set(features_combined_filtered.index)
			samples_oi = set(df_sv_counts_per_sample.loc[df_sv_counts_per_sample.values >= 1].index) & (set(features_combined_filtered.index) | set(features_combined_cup.index))
			df_sv_counts_per_sample_non_zero = df_sv_feats_total.loc[samples_oi]
			df_sv_sum = df_sv_counts_per_sample_non_zero.sum()
			sv_feats_oi = df_sv_sum.loc[df_sv_sum.values >= sv_feat_thresh].index
			df_sv_counts_per_sample_non_zero_filtered = df_sv_counts_per_sample_non_zero[sv_feats_oi]
			samples_oi_ckp = samples_oi & set(features_combined_filtered.index)
			for col in df_sv_counts_per_sample_non_zero_filtered.columns:
				features_combined_filtered[col] = 0.0
				features_combined_filtered.loc[samples_oi_ckp, col] = df_sv_counts_per_sample_non_zero_filtered.loc[samples_oi_ckp][col].values
			# CUP
			# samples_oi = set(df_sv_counts_per_sample.loc[df_sv_counts_per_sample.values >= 1].index) & set(features_profile_cup.index)
			# df_sv_counts_per_sample_non_zero = df_sv_feats.loc[samples_oi]
			samples_oi_cup = samples_oi & set(features_combined_cup.index)
			for col in df_sv_counts_per_sample_non_zero_filtered.columns:
				features_combined_cup[col] = 0.0
				features_combined_cup.loc[samples_oi_cup, col] = df_sv_counts_per_sample_non_zero_filtered.loc[samples_oi_cup][col].values
			# breakpoint()

		# Export common_features for future use
		common_features_df = pd.DataFrame(common_features, index = np.arange(len(common_features)), columns = ['common_features'])
		common_features_df.to_csv('common_features_df', sep = '\t', index = False)

		# Export combined (Profile & GENIE) features and labels
		features_combined_filtered.to_csv(path + '/features_combined_' + suffix, sep = '\t', index = True)
		labels_combined_filtered.to_csv(path + '/labels_combined_' + suffix, sep = '\t', index = True)
		features_combined_cup.to_csv(path + '/features_combined_cup_' + suffix, sep = '\t', index = True)

		# get stats
		print('\n')
		print('\n')
		print('Printing stats...')
		genie_sample_ids = set([idx for idx in features_combined_filtered.index if not type(idx) == float])
		profile_sample_ids = set(features_combined_filtered.index) - genie_sample_ids
		# genie_cancer_types = data_samples_genie.loc[data_samples_genie.SAMPLE_ID.isin(genie_sample_ids)].CANCER_TYPE.values
		genie_cancer_types = labels_combined_filtered.loc[genie_sample_ids].cancer_type.values
		genie_centers = data_genie_patients.loc[data_genie_patients.SAMPLE_ID.isin(genie_sample_ids)].CENTER.values
		print('GENIE')
		print(pd.value_counts(genie_cancer_types, sort = True))
		print(pd.value_counts(genie_centers, sort = True))
		print('Profile')
		profile_cancer_types = labels_combined_filtered.loc[profile_sample_ids].cancer_type.values
		print(pd.value_counts(profile_cancer_types, sort = True))
		# breakpoint()
	if data_to_obtain == 'profile' or data_to_obtain == 'all':
		# Export Profile features
		# breakpoint()
		features_profile_new.to_csv(path + '/features_profile_' + suffix, sep = '\t', index = True)
		labels_profile_new.to_csv(path + '/labels_profile_' + suffix, sep = '\t', index = True)
		features_profile_cup.to_csv(path + '/features_profile_cup_' + suffix, sep = '\t', index = True)
		print('Profile only')
		print(pd.value_counts(labels_profile_new.cancer_type.values, sort = True))

	# Export CUP features
	# if data_to_obtain != 'all':
	# 	common_features = list(pd.read_csv('common_features_df', sep = '\t').common_features.values)
	# features_profile_cup.to_csv(path + '/features_profile_cup_' + suffix, sep = '\t', index = True)
	# features_combined_cup = features_profile_cup[common_features]
	# features_combined_cup.to_csv(path + '/features_combined_cup_' + suffix, sep = '\t', index = True)
	return

if __name__ == '__main__':
  app.run(main)
# if __name__ == "__main__":
# 	explain = """
# Description : 

# The following script provides features and labels for a supervised training of classifying tumors with known primaries based on samples'
# 	1. clinical information (e.g. age and gender)
# 	2. somatic mutation profile (i.e. frequency of mutated genes)
# 	3. copy number alteration events 
# 	(i.e. genes associated with CNA : -2 (deep loss), -1 (single-copy loss), 0 (diploid), 1 (low-level gain), 2 (high-level amplification))

# As outputs, an user has an option of obtaining processed features and labels (for non-CUP only):
# 	1. GENIE data
# 	2. PROFILE data (DFCI)
# 	3. PROFILE CUP (Cancer of Unknown Primaries)s
# 	4. PROFILE CUP (Cancer of Unknown Primaries)s compatible with features of GENIE data
# 	"""
# 	print("\n")
# 	print(explain)
# 	print("\n")

# 	data_to_obtain = input("Enter the features/labels you would like to get (all, profile, profile_cup_only) ")
# 	print("\n")
# 	include_mut_sig = input("Use mutation signature features instead? (Y/N) : ") == 'Y'
# 	if data_to_obtain not in ['all', 'profile', 'profile_cup_only']:
# 		raise KeyError('Please choose among all, profile, and profile_cup_only')
# 	# second_number = int(input("Enter second number "))
# 	main(data_to_obtain)

