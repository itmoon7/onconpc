import glob
from typing import List, Mapping, Optional, Any, Tuple
import collections
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import xgboost as xgb
from adjustText import adjust_text

"""
Author: Intae Moon
This script contains utility functions for predictions and interpretations.
"""

def get_xgboost_cancer_type_preds(xgb_model: xgb.sklearn.XGBClassifier,
								  features_test_df: pd.DataFrame,
								  cancer_types: List[str]) -> pd.DataFrame:
	"""Returns cancer type predictions for test set using XGBoost model.
	
	Args:
		xgb_model: XGBoost model.
		features_test_df: Test set features.
		cancer_types: List of cancer types.
	Returns:
		pd.DataFrame containing cancer type predictions and prediction probabilities.
	"""
	ckp_test_pred_probs = xgb_model.predict_proba(features_test_df.values)
	ckp_test_preds = ckp_test_pred_probs.argmax(axis=1)
	max_posteriors = [pred_dist[max_idx] for pred_dist, max_idx in
					  zip(ckp_test_pred_probs, ckp_test_preds)]
	ckp_test_preds_df = pd.DataFrame(ckp_test_pred_probs,
									 index=features_test_df.index,
									 columns=cancer_types)
	ckp_test_preds_df['max_posterior'] = max_posteriors
	ckp_test_preds_df['cancer_type'] = [cancer_types[max_idx] for max_idx in ckp_test_preds]
	return ckp_test_preds_df

def get_xgboost_latest_cancer_type_preds(xgb_model: xgb.core.Booster,
										 features_test_df: pd.DataFrame,
										 cancer_types: List[str]) -> pd.DataFrame:
	"""Returns cancer type predictions for test set using XGBoost model.
	
	Args:
		xgb_model: XGBoost model.
		features_test_df: Test set features.
		cancer_types: List of cancer types.
	Returns:
		pd.DataFrame containing cancer type predictions and prediction probabilities.
	"""
	dtest = xgb.DMatrix(features_test_df.values)
	ckp_test_pred_probs = xgb_model.predict(dtest, output_margin=True)

	ckp_test_pred_probs = np.exp(ckp_test_pred_probs)
	ckp_test_pred_probs /= ckp_test_pred_probs.sum(axis=1, keepdims=True)
	ckp_test_preds = ckp_test_pred_probs.argmax(axis=1)
	max_posteriors = [pred_dist[max_idx] for pred_dist, max_idx in
					  zip(ckp_test_pred_probs, ckp_test_preds)]
	ckp_test_preds_df = pd.DataFrame(ckp_test_pred_probs,
									 index=features_test_df.index,
									 columns=cancer_types)
	ckp_test_preds_df['max_posterior'] = max_posteriors
	ckp_test_preds_df['cancer_type'] = [cancer_types[max_idx] for max_idx in ckp_test_preds]
	return ckp_test_preds_df

def obtain_shap_values(model: xgb.sklearn.XGBClassifier,
					   data: pd.DataFrame) -> np.ndarray:
	"""Returns SHAP values for predictions based on data.
	
	Args:
		model: XGBoost model.
		data: Data to obtain SHAP values for.
	Returns:
		Numpy array containing SHAP values.
	"""
	# Get SHAP values using the model in byte array.
	mybooster = model.get_booster()
	model_bytearray = mybooster.save_raw()[4:]
	def in_bytearray(self=None):
		return model_bytearray
	mybooster.save_raw = in_bytearray
	shap_ex = shap.TreeExplainer(mybooster)
	return shap_ex.shap_values(data)

def obtain_shap_values_with_latest_xgboost(model: xgb.core.Booster,
										   data: pd.DataFrame) -> np.ndarray:
	"""Returns SHAP values for predictions based on data.
	
	Args:
		model: XGBoost model.
		data: Data to obtain SHAP values for.
	Returns:
		Numpy array containing SHAP values.
	"""
	# Directly use the Booster with the TreeExplainer
	shap_ex = shap.TreeExplainer(model)
	return shap_ex.shap_values(data)

def partition_feature_names_by_group(fature_names: List[str]):
	"""Partitions feature names into groups.
	
	Args:
		feature_names: List of feature names.
	Returns:
		Dictionary mapping feature groups to feature names.
	"""
	feature_group_to_features_dict = collections.defaultdict(list)
	for feat in fature_names:
		if 'SBS' in feat:
			feature_group_to_features_dict['signature'].append(feat)
		elif feat in ['Age', 'Sex']:
			feature_group_to_features_dict['clinical'].append(feat)
		elif 'CNA' in feat:
			feature_group_to_features_dict['cna'].append(feat)
		else:
			feature_group_to_features_dict['mutation'].append(feat)
	return feature_group_to_features_dict

def get_individual_pred_interpretation(shap_pred_sample_df: pd.DataFrame,
									   feature_sample_df: pd.DataFrame,
									   feature_group_to_features_dict: Mapping[str, List[str]],
									   sample_info: Optional[str]=None,
									   filename: Optional[str]=None,
									   filepath: str='others_prediction_explanation',
									   top_feature_num: int=10,
									   top_n_predictions: Optional[Mapping[str, float]]=None,
									   save_plot: bool=False):
	"""
	Get individual prediction interpretation for a given tumor sample.

	Args:
		shap_pred_sample_df: DataFrame containing SHAP values for a given tumor sample.
		feature_sample_df: DataFrame containing feature values for a given tumor sample.
		feature_group_to_features_dict: Dictionary mapping feature groups to feature names.
		sample_info: Sample information to be displayed.
		filename: Filename to save the figure.
		top_feature_num: Number of top features to display.
		top_n_predictions_dict: Dictionary containing top N predictions.
		save_plot: Whether to save the plot.
	"""
	# set font size and font family
	fig, ax = plt.subplots()
	plt.rcParams.update({'font.size': 15})
	plt.rcParams["font.family"] = "Arial"
	plt.close()

	# Create subplots (side-by-side)
	fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

	# Get group-specific absolute SHAP values.
	group_names=['Somatic Mut.', 'CNA events', 'Mutation Sig.', 'Age/Sex']
	shaps_in_feature_groups = []; subgroup_names = []
	for key, feat_names in feature_group_to_features_dict.items():
		if key == 'mutation':
			shaps_mutations = list(shap_pred_sample_df.loc[feat_names].abs().values)
		elif key == 'clinical':
			shaps_clinical = list(shap_pred_sample_df.loc[feat_names].abs().values)
		elif key == 'signature':
			shaps_mut_sigs = list(shap_pred_sample_df.loc[feat_names].abs().values)
		else:
			shaps_cna = list(shap_pred_sample_df.loc[feat_names].abs().values)
	shaps_in_feature_groups = ([shap_pred_sample_df.loc[feature_group_to_features_dict['mutation']].abs().values.sum()] +
							   [shap_pred_sample_df.loc[feature_group_to_features_dict['cna']].abs().values.sum()] +
							   [shap_pred_sample_df.loc[feature_group_to_features_dict['signature']].abs().values.sum()] +
							   [shap_pred_sample_df.loc[feature_group_to_features_dict['clinical']].abs().values.sum()])
	subgroup_size = shaps_mutations + shaps_cna + shaps_mut_sigs + shaps_clinical
	cna_names = [name[:-4] for name in feature_group_to_features_dict['cna']] # get rid of CNA at the end
	subgroup_names_orig = feature_group_to_features_dict['mutation'] + feature_group_to_features_dict['cna'] + feature_group_to_features_dict['signature'] + feature_group_to_features_dict['clinical'] 
	subgroup_names = feature_group_to_features_dict['mutation'] + cna_names + feature_group_to_features_dict['signature'] + feature_group_to_features_dict['clinical'] 

	# Create colors for the pie chart
	a, b, c, d =[plt.cm.Reds, plt.cm.Greens, plt.cm.Blues, plt.cm.Greys]
	 
	# Configure the outer ring (outside)
	ax[0].axis('equal')
	first_ring_width = 0.2
	mypie, _ = ax[0].pie(shaps_in_feature_groups, radius=1.3, labels=group_names, colors=[a(0.6), b(0.6), c(0.6), d(0.6)] )
	plt.setp( mypie, width=first_ring_width, edgecolor='white')
	
	if top_n_predictions:
		# plot top n predictions as text in ax[0]
		top_n_preds_str = '\n'.join([f'{pred}: {prob:.3f}' for pred, prob in top_n_predictions.items()])
		# place them bottom left of the plot
		ax[0].text(-1.5, -1.875, 'Top 3 predictions', fontsize=12, ha='center', va='center')
		ax[0].text(-1.5, -2.25, top_n_preds_str, fontsize=12, ha='center', va='center')
	 
	# Configure the inner ring
	subcolors = []; subgroup_portion = []
	total_score = sum(shaps_in_feature_groups)
	for idx, (group_val, subgroup_val, color_sel) in enumerate(zip(shaps_in_feature_groups,
																   [shaps_mutations,
																   shaps_cna,
																   shaps_mut_sigs,
																   shaps_clinical],
																   [a,b,c,d])):
		for score in subgroup_val:
			subgroup_portion.append(score/total_score)
			if idx == 3:
				subcolors.append(color_sel(score/group_val*0.6))
			else:
				subcolors.append(color_sel(score/group_val))
	subgroup_names_chosen = []
	top_features = []
	top_features_orig = []
	top_feature_indices = np.argsort(subgroup_portion)[-top_feature_num:]
	for name, orig_name, idx in zip(subgroup_names, subgroup_names_orig, np.arange(len(subgroup_names))):
		if idx in top_feature_indices:
			if name == 'Gender':
				subgroup_names_chosen.append('Sex')
			else:
				subgroup_names_chosen.append(name)
			top_features_orig.append(orig_name)
			top_features.append(name)
		else:
			subgroup_names_chosen.append('')

	# If clinical features are not in the top feats. Then use lighter shade
	if 'Age' not in subgroup_names_chosen and 'Gender' not in subgroup_names_chosen:
		subcolors[-1] = d(0.2)
		subcolors[-2] = d(0.2)
	mypie2, _ = ax[0].pie(subgroup_size, radius=1.3-first_ring_width, labels=subgroup_names_chosen, labeldistance=0.8, 
					   colors=subcolors)
	plt.setp(mypie2, width=0.4, edgecolor='white')
	plt.margins(0,0)
	ax[0].set_title(sample_info)

	top_feats_df = pd.DataFrame([])
	top_feats_df['feat_name'] = top_features
	top_feats_df['feat_val'] = feature_sample_df.loc[top_features_orig].values
	top_feats_df['SHAP_val'] = shap_pred_sample_df.loc[top_features_orig].values

	top_features_color = []
	for feat in top_features_orig:
		if feat in feature_group_to_features_dict['mutation']:
			top_features_color.append('red')
		elif feat in feature_group_to_features_dict['cna']:
			top_features_color.append('green')
		elif feat in feature_group_to_features_dict['signature']:
			top_features_color.append('blue')
		else:
			top_features_color.append('gray')

	top_feats_df['colors'] = top_features_color
	top_feats_df.sort_values('feat_val', inplace = True)
	dot_size_list = []
	for val in top_feats_df.SHAP_val.values:
		dot_size_list.append(abs(val)*150)
	ax[1].scatter(top_feats_df.feat_val.values, top_feats_df.SHAP_val.values, color = top_feats_df.colors, s=dot_size_list, alpha=0.6)
	# Label markers
	texts = [plt.text(x_cor, y_cor, feat_name, ha = 'center', va = 'center', color = color) for x_cor, y_cor, color, feat_name in zip(top_feats_df.feat_val.values, top_feats_df.SHAP_val.values, top_feats_df.colors.values, top_feats_df.feat_name.values)]
	adjust_text(texts, expand_text = (1.6, 1.8), arrowprops=dict(arrowstyle='-', color='black', linestyle = ':'))
	ax[1].set_xlabel('Feature value')
	ax[1].set_ylabel('SHAP value')
	ax[1].set_xlim(min(top_feats_df.feat_val.values) - 0.25, max(top_feats_df.feat_val.values) + 0.25)
	ax[1].set_ylim(min(top_feats_df.SHAP_val.values) - 0.25, max(top_feats_df.SHAP_val.values) + 0.25)

	ax[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
	ax[1].axvline(x=0, color='k', linestyle='--', alpha=0.5)

	ax[1].spines['right'].set_visible(False)
	ax[1].spines['top'].set_visible(False)

	fig.tight_layout()
	# Store image as a pdf
	if save_plot:
		full_filename = os.path.join(filepath, filename + '.pdf')
		plt.savefig(full_filename,
			  bbox_inches='tight')
		plt.show()
		print(f'Explanation plot saved at {full_filename}')
		return full_filename
	plt.show()

def get_onconpc_prediction_explanations(query_ids: List[str], 
										preds_df: pd.DataFrame, 
										shaps: np.array, 
										df_features_genie: pd.DataFrame, 
										cancer_types_to_consider: List[str],
										filepath: str='others_prediction_explanation',
										save_plot: bool=False,
										top_three_preds: bool=False,
										) -> List[Mapping[str, Any]]:
	"""
	Get OncoNPC predictions and generate SHAP-based explanation plots for multiple query IDs.

	Args:
		query_ids: List of IDs of the tumor samples to query.
		preds_df: DataFrame containing predictions.
		shaps: Array of SHAP values.
		df_features_genie: DataFrame containing features.
		cancer_types_to_consider: List of cancer types considered in the prediction.
		filepath: Path to save the explanation plots.
		save_plot: Whether to save the explanation plots.
		top_three_preds: Whether to visualize top three predictions.
	Returns:
		List of dictionaries containing prediction details and explanation plots for each query ID.
	"""
	if top_three_preds:
		top_n_predictions_dict = get_top_n_pred(preds_df, cancer_types_to_consider, n=3)
	results_dict = {}
	for query_id in query_ids:
		# Get OncoNPC prediction
		pred_prob = preds_df.at[query_id, 'max_posterior']
		pred_cancer = preds_df.at[query_id, 'cancer_type']
		pred_cancer_idx = cancer_types_to_consider.index(pred_cancer)

		# Get SHAP-based explanation for the prediction
		feature_sample_df = df_features_genie.loc[query_id]
		shap_pred_cancer_df = pd.DataFrame(shaps[pred_cancer_idx],
										   index=df_features_genie.index,
										   columns=df_features_genie.columns)
		shap_pred_sample_df = shap_pred_cancer_df.loc[query_id]

		# Information and plot generation
		sample_info = f'SAMPLE_ID: {query_id}\nPrediction: {pred_cancer}\nPrediction probability: {pred_prob:.3f}'
		feature_group_to_features_dict = partition_feature_names_by_group(df_features_genie.columns)
		top_n_predictions = None if not top_three_preds else top_n_predictions_dict[query_id]
		full_filename = get_individual_pred_interpretation(shap_pred_sample_df,
													 feature_sample_df,
													 feature_group_to_features_dict,
													 sample_info=sample_info,
													 filename=str(query_id),
													 filepath=filepath,
													 top_n_predictions=top_n_predictions,
													 save_plot=save_plot)
		# Store the results
		results_dict[query_id] = {
			'pred_prob': pred_prob,
			'pred_cancer': pred_cancer,
			'explanation_plot': full_filename
		}
	return results_dict

def get_onconpc_features_from_raw_data(df_patients_chosen: pd.DataFrame, 
									   df_samples_chosen: pd.DataFrame, 
									   df_mutations_chosen: pd.DataFrame, 
									   df_cna_chosen: pd.DataFrame, 
									   features_onconpc_path: str='data/features_onconpc.pkl',
									   combined_cohort_age_stats_path: str='data/combined_cohort_age_stats.pkl',
									   mut_sig_weights_filepath: str='data/mutation_signatures/sigProfiler*.csv'
									   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""
	Process raw GENIE data for OncoNPC prediction inference.

	Args:
		df_patients_chosen: DataFrame containing chosen patient data.
		df_samples_chosen: DataFrame containing chosen sample data.
		df_mutations_chosen: DataFrame containing chosen mutation data.
		df_cna_chosen: DataFrame containing chosen copy number alteration data.
		features_onconpc_path: Path to OncoNPC features file.
		combined_cohort_age_stats_path: Path to combined cohort age statistics file.
		weights_filepath: Path to mutation signatures weights file.

	Returns:
		Processed DataFrame ready for OncoNPC prediction inference.
	"""
	# Re-shape df_cna_chosen
	df_cna_reshaped = df_cna_chosen.set_index(['Hugo_Symbol']).T.copy()
	# Merge patient and sample data
	df_patients_merged = pd.merge(df_patients_chosen, df_samples_chosen, how='right', on='PATIENT_ID')

	# Process mutation data
	df_trinuc_feats_genie = get_snv_in_trinuc_context(df_mutations_chosen,
												   sample_id_col='Tumor_Sample_Barcode',
												   chromosome_col='Chromosome',
												   start_pos_col='Start_Position',
												   ref_allele_col='Reference_Allele',
												   alt_allele_col='Tumor_Seq_Allele2',
												   config='genie')

	# Obtain mutation signatures
	df_mut_sigs_genie = obtain_mutation_signatures(df_trinuc_feats_genie, weights_filepath=mut_sig_weights_filepath)
	# Preprocess features and labels
	df_features_genie, df_labels_genie = pre_process_features_genie(df_mutations_chosen, 
																 df_cna_reshaped, 
																 df_mut_sigs_genie, 
																 df_patients_merged)
	# Zero-pad missing features
	with open(features_onconpc_path, "rb") as fp:
		features_onconpc = pickle.load(fp)
	df_features_genie_final = zero_pad_missing_features(df_features_genie, features_onconpc)
	# Standardize feature names
	df_features_genie_final.columns = standardize_feat_names(df_features_genie_final.columns)
	# Standardize Age feature
	with open(combined_cohort_age_stats_path, "rb") as fp:
		combined_cohort_age_stats = pickle.load(fp)
	df_features_genie_final['Age'] = (df_features_genie_final['Age'] - combined_cohort_age_stats['Age_mean']) / combined_cohort_age_stats['Std_mean']
	return df_features_genie_final, df_labels_genie

def get_snv_in_trinuc_context(df_mutations: pd.DataFrame,
							  sample_id_col: str,
							  chromosome_col: str,
							  start_pos_col: str,
							  ref_allele_col: str,
							  alt_allele_col: str,
							  config: Optional[str]='genie') -> pd.DataFrame:
	"""
	Processes the given mutations DataFrame to get SNV in trinucleotide context.

	Args:
		df_mutations: DataFrame containing mutation data.
		sample_id_col: Column name for sample IDs.
		chromosome_col: Column name for chromosome.
		start_pos_col: Column name for start position.
		ref_allele_col: Column name for reference allele.
		alt_allele_col: Column name for alternative allele.
	Returns:
		DataFrame with SNV in trinucleotide context.
	"""
	# Specify the columns to be chosen
	columns_chosen = [sample_id_col, chromosome_col, start_pos_col, ref_allele_col, alt_allele_col]

	# Process the DataFrame
	df_mutations_chosen_trinuc = df_mutations[columns_chosen].copy()
	if config == 'genie':
		df_mutations_chosen_trinuc[chromosome_col] = 'chr' + df_mutations_chosen_trinuc[chromosome_col].astype(str)
	df_mutations_chosen_trinuc = df_mutations_chosen_trinuc[~df_mutations_chosen_trinuc[chromosome_col].str.contains('GL|chrMT')]
	# Change the data types for R processing
	df_mutations_chosen_trinuc[sample_id_col] = df_mutations_chosen_trinuc[sample_id_col].astype('str')
	# R function code
	r_code = '''
	function(mutationData, sample_id, chr, pos, ref, alt) {
		library(deconstructSigs)
		sigs.input <- mut.to.sigs.input(mut.ref = mutationData, 
										sample.id = sample_id,
										chr = chr, 
										pos = pos, 
										ref = ref, 
										alt = alt)

		# Filter samples with low mutations
		sigs.input <- sigs.input[rowSums(sigs.input) >= 1,]
		return(sigs.input)
	}
	'''
	# Convert the DataFrame for R processing
	with localconverter(robjects.default_converter + pandas2ri.converter):
		r_df_mutations = robjects.conversion.py2rpy(df_mutations_chosen_trinuc)

	# Load and call the R function
	r_function = robjects.r(r_code)
	with localconverter(robjects.default_converter + pandas2ri.converter):
		df_trinuc_feats = robjects.conversion.rpy2py(r_function(r_df_mutations, 
															   sample_id_col, 
															   chromosome_col, 
															   start_pos_col, 
															   ref_allele_col, 
															   alt_allele_col))
	if config=='profile_dfci':
		df_trinuc_feats.index = df_trinuc_feats.index.astype('float')
	return df_trinuc_feats

def obtain_mutation_signatures(df_trinuc_feats: pd.DataFrame,
							   weights_filepath: str='../data/mutation_signatures/sigProfiler*.csv'
							   ) -> pd.DataFrame:
	"""Transforms tri-nucleotide features into mutation signature based features.

	Args:
		df_trinuc_feats: DataFrame containing tri-nucleotide features.
	Returns:
		mut_sig_based_df: DataFrame containing mutation signature based features.
	"""
	# Load Mutational Signatures COSMIC data.
	file_names = glob.glob(weights_filepath)
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
		common_substitute_feats = list(set(mut_df.index) & set(df_trinuc_feats.columns))
		if len(common_substitute_feats) != 96:
			raise ValueError('Number of substitution features does not sum up to 96.')
		# Get mutation signature values
		mut_sig_vals = np.matmul(df_trinuc_feats[common_substitute_feats].values,
								 mut_df.loc[common_substitute_feats].values_oi.values)
		mut_sig_based_df[key] = mut_sig_vals
	return mut_sig_based_df

def pre_process_features_genie(df_mutations: pd.DataFrame,
							   df_cna: pd.DataFrame,
							   df_mutation_signatures: pd.DataFrame,
							   df_patients: pd.DataFrame,
							   cancer_types: Optional[List[str]]=None, 
							   id_column: str = 'SAMPLE_ID',
							   cancer_type_column: str='CANCER_TYPE') -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Pre-process genetics data to create feature df for GENIE.
	
	Args:
		df_mutations: DataFrame containing mutation data.
		df_cna: DataFrame containing CNA data.
		df_mutation_signatures: DataFrame containing mutation signature data.
		df_patients: DataFrame containing patient data.
		cancer_types: List of cancer types.
		id_column: Column name for sample IDs.
		cancer_type_column: Column name for cancer types.
	Returns:
		df_features_merged_final: DataFrame containing merged features.
		df_labels_final: DataFrame containing labels.
	"""
	# Get mutation features.
	sample_ids = df_patients[id_column].values
	df_mutations_chosen = df_mutations.loc[df_mutations.Tumor_Sample_Barcode.isin(sample_ids)]
	mutation_gene_names = np.unique(df_mutations_chosen.Hugo_Symbol.values)
	df_mutation_feature = pd.DataFrame(0, columns=mutation_gene_names, index=sample_ids)
	for entry in df_mutations_chosen[['Tumor_Sample_Barcode', 'Hugo_Symbol']].values:
		sample_id = entry[0]
		gene = entry[1]
		if gene in df_mutation_feature.columns:
			# Count the number of mutation frequency in each gene.
			df_mutation_feature.at[sample_id, gene] = df_mutation_feature.at[sample_id, gene] + 1
	# Add 'mut' at the end of each mutation feature
	df_mutation_feature.columns = [gene + '_mut' for gene in mutation_gene_names]
	
	# Get CNA features.
	df_cna_chosen = df_cna.loc[list(set(df_cna.index) & set(sample_ids))]
	df_cna_chosen.fillna(0, inplace = True)

	# Obtain age/sex features.
	df_patients_id_indexed = df_patients.set_index('SAMPLE_ID')
	df_sex_age_feature = df_patients_id_indexed[['SEX', 'AGE_AT_SEQ_REPORT']].copy()
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
		elif '>89' in str(age):
			age_list.append(int(age[1:]))
		elif '<18' in str(age):
			age_list.append(int(age[1:]))
		else:
			age_list.append(int(age))
	df_sex_age_feature['SEX'] = sex_list
	df_sex_age_feature['AGE_AT_SEQ_REPORT'] = age_list
	df_sex_age_feature.columns = ['Sex', 'Age']
	# Merge all feature dfs using common sample IDs.
	common_ids = list(set(df_mutation_feature.index) & set(df_cna_chosen.index) & set(df_mutation_signatures.index)
				  & set(df_sex_age_feature.index))
	dfs_list = [df_mutation_feature, df_cna_chosen, df_mutation_signatures, df_sex_age_feature]
	df_features_merged = merge_dfs(dfs_list, common_ids)
	# Exclude age NaN indices.
	df_features_merged_final = df_features_merged.loc[list(set(df_features_merged.index) - set(age_nan_indices))]
	df_labels = df_patients_id_indexed.loc[df_features_merged_final.index][cancer_type_column]
	if cancer_types is not None:
		df_labels_final = pd.DataFrame([cancer_types.index(val) for val in df_labels.values],
									columns = ['cancer_label'], index = df_labels.index)
		df_labels_final['cancer_type'] = df_labels.values	
		return df_features_merged_final, df_labels_final
	else:
		return df_features_merged_final, df_labels

def pre_process_features_dfci(df_mutations: pd.DataFrame,
							  df_cna: pd.DataFrame,
							  df_mutation_signatures: pd.DataFrame,
							  df_patients: pd.DataFrame,
							  cancer_types: List[str], 
							  cup_samples_ids: Optional[List[str]]=None,
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
	mutation_gene_names = np.unique(df_mutations_chosen.CANONICAL_GENE.values)
	df_mutation_feature = pd.DataFrame(0, columns=mutation_gene_names, index=sample_ids)
	for entry in df_mutations_chosen[['UNIQUE_SAMPLE_ID', 'CANONICAL_GENE']].values:
		sample_id = entry[0]; gene = entry[1];
		if gene in df_mutation_feature.columns:
			# Count the number of mutation frequency in each gene.
			df_mutation_feature.at[sample_id, gene] = df_mutation_feature.at[sample_id, gene] + 1
	# Add 'mut' at the end of each mutation feature
	df_mutation_feature.columns = [gene + '_mut' for gene in mutation_gene_names]
	# Get CNA features.
	df_cna_chosen = df_cna.loc[list(set(df_cna.index) & set(sample_ids))]
	df_cna_chosen.fillna(0, inplace = True)

	# Obtain age/sex features.
	df_patients_id_indexed = df_patients.set_index(id_column)
	df_sex_age_feature = df_patients_id_indexed[['Sex', 'Age']].copy()
	sex_list = []
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
	common_ids = list(set(df_mutation_feature.index) & set(df_cna_chosen.index) &
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
							   .loc[list(set(df_features_merged.index) - set(cup_samples_ids))])
	df_features_merged_cups = (df_features_merged
							   .loc[list(set(df_features_merged.index) & set(cup_samples_ids))])
	df_labels = df_patients_id_indexed.loc[df_features_merged_ckps.index]['CANCER_TYPE']
	df_labels_final = pd.DataFrame([cancer_types.index(val) for val in df_labels.values],
								   columns=['cancer_label'], index=df_labels.index)
	df_labels_final['cancer_type'] = df_labels.values
	return df_features_merged_ckps, df_labels_final, df_features_merged_cups

def zero_pad_missing_features(df: pd.DataFrame,
							  feature_list: List[str]) -> pd.DataFrame:
	"""
	Zero pad missing features.
	Args:
		df: DataFrame containing features.
		feature_list: List of features to include.
	Returns:
		df: DataFrame containing zero-padded features.
	"""
	# Create a dictionary for new columns to add
	new_columns = {feature: [0] * len(df) for feature in feature_list if feature not in df.columns}

	# Concatenate new columns with the original DataFrame
	df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
	return df[feature_list]

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

def standardize_feat_names(curr_feat_names: List[str]) -> List[str]:
	"""
	Standardize feature names.
	Args:
		curr_feat_names: list of feature names
	Returns:
		new_feat_names: list of standardized feature names
	"""
	new_feat_names = []
	for feat in curr_feat_names:
		if '_mut' in feat:
			new_feat_names.append(feat.replace('_mut', ''))
		elif 'AGE' in feat or 'Age' in feat:
			new_feat_names.append('Age')
		elif 'GENDER' in feat or 'Sex' in feat:
			new_feat_names.append('Sex')
		elif 'SBS' in feat:
			new_feat_names.append(feat)
		else:   
			new_feat_names.append(feat + ' CNA')
	return new_feat_names

def get_top_n_pred(df: pd.DataFrame,				
				   cancer_types: List[str],
				   n: int=3) -> Mapping[str, Mapping[str, float]]:
	"""
	Get top n predictions for each sample in the given DataFrame.
	Args:
		df: DataFrame containing predictions.
		n: Number of top predictions to return.
		cancer_types: List of cancer types.
	Returns:
		Dictionary mapping each sample ID to top n predictions and their corresponding probability values.
	"""
	# Drop columns other than cancer types
	columns_to_drop = [c for c in df.columns if c not in cancer_types]
	df_chosen = df.drop(columns=columns_to_drop)
	# Creating a dictionary to map each sample ID to top 3 predictions and their corresponding probability values
	top_n_predictions_dict = {}
	for index, row in df_chosen.iterrows():
		# Getting the top n predictions sorted by their probabilities
		sorted_top_3 = row.nlargest(n).sort_values(ascending=False)
		top_n_predictions_dict[index] = sorted_top_3.to_dict()
	return top_n_predictions_dict