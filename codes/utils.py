from adjustText import adjust_text
import collections
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from typing import List


def get_xgboost_cancer_type_preds(xgb_model: xgb.sklearn.XGBClassifier,
								  features_test_df: pd.DataFrame,
								  cancer_types: List[str]) -> pd.DataFrame:
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

def obtain_shap_values(model, data: pd.DataFrame):
	"""Returns SHAP values for predictions based on data."""
	new_column_names = []
	# Get SHAP values using the model in byte array.
	mybooster = model.get_booster()
	model_bytearray = mybooster.save_raw()[4:]
	def in_bytearray(self=None):
		return model_bytearray
	mybooster.save_raw = in_bytearray
	shap_ex = shap.TreeExplainer(mybooster)
	return shap_ex.shap_values(data)

def partiton_feature_names_by_group(fature_names):
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

def get_individual_pred_interpretation(shap_pred_sample_df, feature_sample_df,
									   feature_group_to_features_dict,
									   sample_info=None, filename=None,
									   top_feature_num=10):

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
	filepath = 'cup_prediction_explanation/'
	plt.savefig(filepath + str(filename) + '.pdf', bbox_inches='tight')
	plt.close()