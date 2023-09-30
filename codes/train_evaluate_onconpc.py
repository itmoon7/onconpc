from absl import app
from absl import flags
import pandas as pd
import pickle
import utils_training
import utils
import os
import numpy as np

"""
Intae Moon
Sep 10, 2023

In this script, we train and evaluate the performance of the XGBoost model on the OncoTree-based cancer types.
One can choose to use k-fold cross validation or train on the entire dataset and test on the held out set.
"""

flags.DEFINE_integer('k_fold', 10, 'Number of folds in k-fold cross-validation')
flags.DEFINE_boolean('use_held_out_set', False, 'Whether to use a held-out set')
flags.DEFINE_boolean('filter_out_non_informatives', False, 'Whether to filter out non-informative features and samples')
flags.DEFINE_string('save_model_name', None, 'Name of the model to save')
flags.DEFINE_string('index_profile_samples_by', 'SAMPLE_ACCESSION_NBR', 'Index profile samples by DFCI_MRN or SAMPLE_ACCESSION_NBR')
DATA_PATH = '../data/internal_training_data/'


def main(argv=None):
	# Custom check for flags that you still require to be non-default
	FLAGS = flags.FLAGS
	if FLAGS.save_model_name is None:
		raise ValueError("Required flag not provided: --save_model_name. Use --helpfull to see help on flags.")
	# Training configs:
	k_fold = FLAGS.k_fold
	use_held_out_set = FLAGS.use_held_out_set
	save_model_name = FLAGS.save_model_name
	index_profile_samples_by = FLAGS.index_profile_samples_by
	filter_out_non_informatives = FLAGS.filter_out_non_informatives

	# load xgb parameters
	hyperparam_xgb = pd.read_csv(os.path.join(DATA_PATH, 'xgb_hyperparams'), sep = '\t')  #xgb_params_to_run # params_w_results_Mar_1st
	hyperparam_xgb.drop(columns = 'Unnamed: 0', inplace = True)
	# Get the best performing hyperparameters
	params_xgb = hyperparam_xgb.iloc[0:1]

	# ====================================================================================================
	# Loading data and labels, and processing them for training
	# ====================================================================================================
	# Load trainable data and labels
	# Tab separated feature data for CKPs
	feature_data_name = os.path.join(DATA_PATH, 'features_combined_onco_tree_based_dev')  # features_combined_onco_tree_based_dev, features_combined_default
	# Tab separated label data for CKPs
	label_data_name = os.path.join(DATA_PATH, 'labels_combined_onco_tree_based_dev')  # labels_combined_onco_tree_based_dev, labels_combined_default
	# Tab separated feature data for CUPs
	feature_data_name_cup = os.path.join(DATA_PATH, 'features_combined_cup_onco_tree_based_dev')  # features_combined_cup_onco_tree_based_dev, features_combined_cup_default
	
	features_ckp_df = pd.read_csv(feature_data_name, sep = '\t')
	features_ckp_df.set_index(features_ckp_df.columns[0], inplace = True)
	labels_ckp_df = pd.read_csv(label_data_name, sep = '\t')
	labels_ckp_df.set_index(labels_ckp_df.columns[0], inplace = True)
	features_cup_df = pd.read_csv(feature_data_name_cup, sep = '\t')
	features_cup_df.set_index('Unnamed: 0', inplace = True)

	if index_profile_samples_by == 'DFCI_MRN':
		# Load ../data/internal_training_data/onconpc_sample_id_to_dfci_mrn.pkl
		with open(os.path.join(DATA_PATH, 'onconpc_sample_id_to_dfci_mrn.pkl'), "rb") as fp:   # Unpickling
			onconpc_sample_id_to_dfci_mrn = pickle.load(fp)
		# index CKP data by DFCI_MRN
		features_ckp_df.index = utils_training.get_new_indices(features_ckp_df.index, onconpc_sample_id_to_dfci_mrn)
		labels_ckp_df.index = utils_training.get_new_indices(labels_ckp_df.index, onconpc_sample_id_to_dfci_mrn)
		features_cup_df.index = utils_training.get_new_indices(features_cup_df.index, onconpc_sample_id_to_dfci_mrn)
		# Set index name
		features_ckp_df.index.name = 'DFCI_MRN_FOR_PROFILE'
		labels_ckp_df.index.name = 'DFCI_MRN_FOR_PROFILE'
		features_cup_df.index.name = 'DFCI_MRN_FOR_PROFILE'
		if use_held_out_set:
			heldout_ckps_preds_df = pd.read_csv(os.path.join(DATA_PATH, 'heldout_ckps_preds_onco_tree_based_dev'), sep = '\t')
			heldout_ckps_preds_df.set_index(heldout_ckps_preds_df.columns[0], inplace = True)
			held_out_indices = utils_training.get_new_indices([idx[:-3] for idx in heldout_ckps_preds_df.index],
													 onconpc_sample_id_to_dfci_mrn)
	elif index_profile_samples_by == 'SAMPLE_ACCESSION_NBR':
		# Load ../data/internal_training_data/onconpc_sample_id_to_sample_accession_nbr.pkl
		with open(os.path.join(DATA_PATH, 'onconpc_sample_id_to_sample_accession_nbr.pkl'), "rb") as fp:   # Unpickling
			onconpc_sample_id_to_sample_accession_nbr = pickle.load(fp)
		# index CKP data by SAMPLE_ACCESSION_NBR
		features_ckp_df.index = utils_training.get_new_indices(features_ckp_df.index, onconpc_sample_id_to_sample_accession_nbr)
		labels_ckp_df.index = utils_training.get_new_indices(labels_ckp_df.index, onconpc_sample_id_to_sample_accession_nbr)
		features_cup_df.index = utils_training.get_new_indices(features_cup_df.index, onconpc_sample_id_to_sample_accession_nbr)
		# Set index name
		features_ckp_df.index.name = 'SAMPLE_ACCESSION_NBR_FOR_PROFILE'
		labels_ckp_df.index.name = 'SAMPLE_ACCESSION_NBR_FOR_PROFILE'
		features_cup_df.index.name = 'SAMPLE_ACCESSION_NBR_FOR_PROFILE'
		if use_held_out_set:
			heldout_ckps_preds_df = pd.read_csv(os.path.join(DATA_PATH, 'heldout_ckps_preds_onco_tree_based_6_3_21'), sep = '\t')
			heldout_ckps_preds_df.set_index(heldout_ckps_preds_df.columns[0], inplace = True)
			held_out_indices = utils_training.get_new_indices([idx[:-3] for idx in heldout_ckps_preds_df.index],
													 onconpc_sample_id_to_sample_accession_nbr)
	if use_held_out_set:
		# Exclude held out samples from training
		indices_to_choose = set(features_ckp_df.index) - set(held_out_indices)
		features_ckp_df = features_ckp_df.loc[indices_to_choose]
		labels_ckp_df = labels_ckp_df.loc[indices_to_choose]
	
	# Standardize feature names
	new_feat_names = utils_training.standardize_feat_names(list(features_ckp_df.columns))
	features_ckp_df.columns = new_feat_names
	features_cup_df.columns = new_feat_names
	feature_group_to_features_dict = utils.partiton_feature_names_by_group(new_feat_names)

	if filter_out_non_informatives:
		(features_ckp_final_df,
	labels_ckp_final_df,
	features_cup_final_df) = utils_training.filter_out_low_freq_feats_and_samples(features_ckp_df,
																					labels_ckp_df,
																					features_cup_df, 
																					feature_group_to_features_dict)
	else:
		features_ckp_final_df = features_ckp_df
		labels_ckp_final_df = labels_ckp_df
		features_cup_final_df = features_cup_df

	# ====================================================================================================
	# Model Training and Evaluation
	# ====================================================================================================
	# Cancer types to consider
	cancer_types = list(np.unique(labels_ckp_final_df.cancer_type.values))
	# Check duplicate indices
	if len(features_ckp_final_df) == len(set(features_ckp_final_df.index)):
		print('\n')
		print('No duplicate indices')
	else:
		print('\n')
		print('Duplicate indices detected (most likely due to DFCI_MRN indexing)')
	
	(k_fold_to_performance_report_dict,
  pred_probs_on_val_total_df) = utils_training.perform_k_fold(features_ckp_final_df,
															  labels_ckp_final_df,
															  cancer_types,
															  params_xgb,
															  k_fold=k_fold,
															  save_model_name=save_model_name)
	# Store k_fold_to_performance_report_dict and pred_probs_on_val_total_df
	with open(os.path.join(DATA_PATH, f'k_fold_to_performance_report_dict_{save_model_name}.pkl'), 'wb') as f:
		pickle.dump(k_fold_to_performance_report_dict, f)
	pred_probs_on_val_total_df.to_csv(os.path.join(DATA_PATH, f'pred_probs_on_val_total_df_{save_model_name}.csv'))
	return 

if __name__ == '__main__':
	flags.mark_flags_as_required(['k_fold', 'use_held_out_set', 'save_model_name'])
	app.run(main)
