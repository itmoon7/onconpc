import pandas as pd
from typing import Optional, List, Tuple, Mapping, Union
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import os
import random

def get_new_indices(curr_indices: List[Union[str, float]],
					mapping_dict: Mapping[float, Union[float,str]],
					non_profile_prefix: Optional[str]='GENIE') -> List[str]:
	"""
	Get new indices based on the mapping dictionary.
	Args:
		curr_indices: list of indices to be mapped
		mapping_dict: dictionary of mapping
		non_profile_prefix: prefix for non-profile data
	Returns:
		new_indices: list of new indices
	"""
	new_indices = []
	for idx in curr_indices:
		if non_profile_prefix in str(idx):
			new_indices.append(idx)
		else:
			new_indices.append(str(mapping_dict[float(idx)]))
	return new_indices

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
			new_feat_names.append(feat.replace('_mut', '_MUT'))
		elif 'AGE' in feat or 'Age' in feat:
			new_feat_names.append('Age')
		elif 'GENDER' in feat or 'Sex' in feat:
			new_feat_names.append('Sex')
		elif 'SBS' in feat:
			new_feat_names.append(feat)
		else:   
			new_feat_names.append(feat + '_CNA')
	return new_feat_names

def filter_by_threshold(df: pd.DataFrame,
						threshold_per_sample: int,
						threshold_per_feature: int) -> Tuple[List[str], set]:
	"""
	Filter out features and samples based on threshold.
	"""
	binary_df = df != 0
	feature_sum = binary_df.sum()
	sample_sum = binary_df.sum(axis=1)
	features_to_exclude = feature_sum.index[feature_sum < threshold_per_feature]
	samples_to_exclude = sample_sum.index[sample_sum < threshold_per_sample]
	return features_to_exclude, set(samples_to_exclude)

def categorize_samples_by_center(samples_to_exclude, labels_ckp):
	"""
	Categorize samples by cancer ceneter.
	"""
	profile_excluded = []
	msk_excluded = []
	vicc_excluded = []
	for sample_id in samples_to_exclude:
		sample_str = str(sample_id)
		if 'MSK' in sample_str:
			msk_excluded.append(sample_id)
		elif 'VICC' in sample_str:
			vicc_excluded.append(sample_id)
		else:
			profile_excluded.append(sample_id)
	return labels_ckp.loc[profile_excluded], labels_ckp.loc[vicc_excluded], labels_ckp.loc[msk_excluded]

def filter_out_low_freq_feats_and_samples(data_ckp: pd.DataFrame,
										  labels_ckp: pd.DataFrame,
										  data_cup: pd.DataFrame,
										  feature_group_to_features_dict: Mapping[str, List[str]],
										  threshold_per_sample: int=3,
										  threshold_per_feature: int=50) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	Filter out low frequency features and samples.

	Args:
		data_ckp: CKP data
		labels_ckp: CKP labels
		data_cup: CUP-CKP data
		feature_group_to_features_dict: dictionary of feature groups to features
		threshold_per_sample: threshold per sample
		threshold_per_feature: threshold per feature
	Returns:
		data_ckp_filtered: filtered CKP data
		labels_ckp_filtered: filtered CKP labels
		data_cup_fitlered: filtered CUP-CKP data
	"""
	mutation_features_to_exclude, mutation_samples_to_exclude = filter_by_threshold(
		data_ckp[feature_group_to_features_dict['mutation']], threshold_per_sample, threshold_per_feature
	)
	
	cna_features_to_exclude, cna_samples_to_exclude = filter_by_threshold(
		data_ckp[feature_group_to_features_dict['cna']], threshold_per_sample, threshold_per_feature
	)
	
	samples_to_exclude = mutation_samples_to_exclude & cna_samples_to_exclude
	
	profile_labels, vicc_labels, msk_labels = categorize_samples_by_center(samples_to_exclude, labels_ckp)
	
	total_excluded_samples = len(profile_labels) + len(vicc_labels) + len(msk_labels)
	print('\n')
	print('Filtering out low frequency features and samples...\n')
	print(f'Total excluded number of patients : {total_excluded_samples}\n')
	print('Profile:\n', pd.value_counts(profile_labels.cancer_type.values, sort=True))
	print('VICC:\n', pd.value_counts(vicc_labels.cancer_type.values, sort=True))
	print('MSK:\n', pd.value_counts(msk_labels.cancer_type.values, sort=True))
	
	features_to_drop = set(mutation_features_to_exclude) | set(cna_features_to_exclude)
	
	print('Dropping features...\n', features_to_drop)
	data_ckp_filtered = data_ckp.drop(columns=features_to_drop)
	data_cup_fitlered = data_cup.drop(columns=features_to_drop)
	
	print('Dropping samples...\n')
	data_ckp_filtered = data_ckp.drop(index=samples_to_exclude)
	labels_ckp_filtered = labels_ckp.drop(index=samples_to_exclude)
	
	print('Samples labels and features matching :')
	print('CKP labels:', all(labels_ckp.index == data_ckp.index))
	print('CUP-CKP features:', all(data_ckp.columns == data_cup.columns))
	return data_ckp_filtered, labels_ckp_filtered, data_cup_fitlered

def get_cancer_to_num_val_samples(labels: pd.DataFrame,
								  cancer_types: List[str],
								  k_fold: int=10) -> Mapping[str, int]:
	"""
	Get number of validation samples per cancer type.
	Args:
		labels: labels dataframe with cancer types
		cancer_types: list of cancer types to consider
		k_fold: number of folds
	Returns:
		cancer_to_num_val_samples_dict: dictionary of cancer type to number of validation samples
	"""
	test_frac = 1/k_fold
	cancer_to_num_val_samples_dict = {}
	for cancer in cancer_types:
		labels_cancer = labels.loc[labels['cancer_type'] == cancer]
		cancer_to_num_val_samples_dict[cancer] = int(np.floor(len(labels_cancer) * test_frac))
	return cancer_to_num_val_samples_dict

def get_sample_indices_and_labels_based_on_cut_off(pred_probs: np.ndarray,
												   p_max_cut_off: float) -> Tuple[List[int], List[int]]:
	"""
	Get sample indices and labels based on p max cut-off.
	Args:
		pred_probs: predicted probabilities
		p_max_cut_off: cut-off
	Returns:
		indices: list of indices
		max_prob_labels: list of labels
	"""
	indices, max_prob_labels = [], []
	for num_idx, max_prob_idx in enumerate(np.argmax(pred_probs, axis=1)):
		if pred_probs[num_idx][max_prob_idx] > p_max_cut_off:
			indices.append(num_idx)
			max_prob_labels.append(max_prob_idx)
	return indices, max_prob_labels

def fit_and_evaluate_model(X_train: np.ndarray,
						   y_train: np.ndarray,
						   X_test: np.ndarray,
						   y_test: np.ndarray,
						   params_xgb: Mapping[str, Union[str, int, float]],
						   cancer_types: List[str]) -> Tuple[XGBClassifier, np.ndarray, Mapping[str, Union[str, int, float]]]:
	"""
	Fit and evaluate XGBoost model.
	
	Args:
		X_train: training data
		y_train: training labels
		X_test: test data
		y_test: test labels
		params_xgb: XGBoost parameters
		cancer_types: list of cancer types
	Returns:
		xg_clf: XGBoost model
		pred_probs_on_test: Predicted probabilities on test data
		performance_report: classification report
	"""
	xg_clf = XGBClassifier(
		tree_method='hist', 
		n_estimators=int(params_xgb['n_estimators']),
		max_depth=int(params_xgb['max_depth']),
		scale_pos_weight=int(params_xgb['scale_pos_weight']),
		learning_rate=float(params_xgb['learning_rate']),
		verbosity=0
	)
	xg_clf.fit(X_train, y_train, verbose=False)
	pred_probs_on_test = xg_clf.predict_proba(X_test)
	performance_report = classification_report(y_test, xg_clf.predict(X_test), target_names=cancer_types, output_dict=True)
	return xg_clf, pred_probs_on_test, performance_report

def perform_k_fold(data: pd.DataFrame,
				   labels: pd.DataFrame,
				   cancer_types: List[str],
				   params_xgb: Mapping[str, Union[str, int, float]],
				   k_fold: int=10,
				   save_model_name: Optional[str]=None,
				   p_max_cut_offs: Optional[List[float]]=[0.0, 0.5, 0.7, 0.9]
				   ) -> Tuple[Mapping[int, pd.DataFrame], pd.DataFrame]:	
	"""
	Performs k-fold cross validation.

	Args:
		data: training and validation data
		labels: training and validation labels
		cancer_types: list of cancer types
		params_xgb: XGBoost parameters
		k_fold: number of folds
		save_model_name: name of the model to save
		p_max_cut_offs: list of max prediction probability cut-offs
	Returns:
		k_fold_to_performance_report_dict: dictionary of k-fold to performance report
		pred_probs_on_val_total_df: dataframe of predicted probabilities on validation data
	"""
	print('\n')
	print('Chosen parameters for XGBoost:\n', params_xgb)
	print('\n')
	cancer_to_num_val_samples_dict = get_cancer_to_num_val_samples(labels, cancer_types, k_fold)
	k_fold_to_performance_report_dict = {}
	total_val_sampled_so_far = []
	for k in range(k_fold):
		print('\n')
		print(f'k = {k}')
		if k == k_fold - 1:
			# At the last fold, use the remaining samples as validation data.
			val_labels_sampled = labels.loc[list(set(labels.index) - set(total_val_sampled_so_far))]
		else:
			val_labels_sampled = pd.DataFrame()
			# Update the validation labels to sample from.
			val_labels_to_sample_from = labels.loc[list(set(labels.index) - set(total_val_sampled_so_far))]
			for cancer in cancer_types:
				# Sample validation data from each cancer type.
				# this ensures that the validation data is balanced wrt cancer type.
				# Set random seed
				np.random.seed(k)
				random.seed(k)
				val_labels_sampled_curr = val_labels_to_sample_from.loc[val_labels_to_sample_from['cancer_type'] == cancer].sample(
					n=cancer_to_num_val_samples_dict[cancer], replace=False)
				val_labels_sampled = pd.concat([val_labels_sampled, val_labels_sampled_curr])
			# Update the total validation samples sampled so far.
			total_val_sampled_so_far.extend(list(val_labels_sampled.index))

		y_val = val_labels_sampled['cancer_label']
		X_val = data.loc[val_labels_sampled.index]
		X_train = data.loc[list(set(data.index) - set(X_val.index))]
		y_train = labels.loc[X_train.index]['cancer_label']
		# Standardize Age based on train data
		age_mean = X_train['Age'].mean()
		age_std = X_train['Age'].std()
		X_train['Age'] = (X_train['Age'] - age_mean) / age_std
		X_val['Age'] = (X_val['Age'] - age_mean) / age_std
		xg_clf, pred_probs_on_val, performance_report = fit_and_evaluate_model(X_train.values,
																		 y_train.values,
																		 X_val.values,
																		 y_val.values,
																		 params_xgb,
																		 cancer_types)
		# Evaluate the model performance based on different maximum prediction probability cut-offs.
		p_max_cut_off_to_performance_report_dict = {}
		for p_max_cut_off in p_max_cut_offs:
			indices, max_prob_labels = get_sample_indices_and_labels_based_on_cut_off(pred_probs_on_val, p_max_cut_off)
			unique_labels = list(set(y_val[indices]))
			report_cut_off = classification_report(y_val[indices], max_prob_labels,
										  target_names=cancer_types, labels=unique_labels, output_dict=True)
			print(pd.DataFrame(report_cut_off))
			p_max_cut_off_to_performance_report_dict[p_max_cut_off] = pd.DataFrame(report_cut_off)
		k_fold_to_performance_report_dict[k] = p_max_cut_off_to_performance_report_dict
		if save_model_name is not None:
			if not os.path.exists('../models'):
				os.makedirs('../models/models')
			xg_clf.save_model(f'../models/xgboost_{save_model_name}_{k}.json')

		# Get prediction probabilities on val data.
		pred_probs_on_val_df = pd.DataFrame(pred_probs_on_val, columns=cancer_types, index=X_val.index)
		pred_probs_on_val_df['predicted_cancer'] = [cancer_types[max_idx] for max_idx in pred_probs_on_val.argmax(axis=1)]
		pred_probs_on_val_df['prediction_prob'] = [pred_probs_on_val[num_idx][max_idx]
											 for num_idx, max_idx in enumerate(pred_probs_on_val.argmax(axis=1))]
		if k == 0:
			pred_probs_on_val_total_df = pred_probs_on_val_df
		else:
			pred_probs_on_val_total_df = pd.concat([pred_probs_on_val_total_df, pred_probs_on_val_df])
	return k_fold_to_performance_report_dict, pred_probs_on_val_total_df