import sys
import numpy as np
import helper_functions.helper_functions as hf
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC, SVC
import sec5a_ml_model_multiclass_pipeline as ml_pipeline

local_control_panel = {
	'done_switch': False,
}

# Main function
######################################################################
def main(on_switch=False):
	if on_switch:
		save_switch = False
		run_on_subsampled_data = False
		run_on_full_data = True
		run_on_unfeatured_data = False
		run_on_featured_data = True

		# Eg, full list >> [5, 50, 500]
		nk_list = [5] 
		# Eg, ['dummy', 'sex_only', 'name_all', 'name_all_loc_all', 'name_all_loc_all_reduced']
		feature_set_list = ['name_all_loc_all_reduced']
		ml_algo_dict = {	
							'LR':	{'clf': LogisticRegression(
										tol=0.01438449888287663, 
										solver='liblinear',
										penalty='l2',
										multi_class='ovr',
										max_iter=50,
										class_weight=None,
										C=10000.0)
									},
							'SVC':	{'clf': OneVsRestClassifier(LinearSVC(
										tol=0.0001,
										penalty='l2',
										multi_class='ovr',
										max_iter=50,
										loss='squared_hinge',
										class_weight='balanced',
										C=0.08858667904100823)),
									},
							'NB':	{'clf': BernoulliNB(
										fit_prior=True,
										binarize=0.6,
										alpha=0.00026366508987303583)
									},
						}

		if run_on_subsampled_data:
			# Loop through subsampling n set with unfeatured data
			if run_on_unfeatured_data:
				for nk in nk_list:
					for algo_key, algo_val in ml_algo_dict.items():
						obj = ml_pipeline.MachineLearningNameEthnicityProjectMultiClass(control_panel = {
								'save_result_switch': save_switch, # WARNING: Will overwrite existing
								'use_subsampled_df_switch': True,
								'use_subsampled_df_nk': nk,
								'use_featured_df_switch': False,
								'use_feature_set': [],
								'feature_selection_switch': False,
								'cross_validation_switch': False,
								'cross_validation_repeat': None,
								'ml_process_on_test_data_switch': True,
								'ml_process_on_training_data_switch': False,
								'ml_process_on_ext_data_switch': False,
								'ml_algo': [algo_key, algo_val],
								'ml_algo_param_grid': None,
								'eval_score': None,
								'label_varname': 'ETHNICITY_RECAT', 
								'random_state': 888,
								})
						obj.machine_learning_steps()

			# Loop through feature set and subsampling n set
			if run_on_featured_data:
				for feature_set in feature_set_list:
					for nk in nk_list:
						for algo_key, algo_val in ml_algo_dict.items():
							obj = ml_pipeline.MachineLearningNameEthnicityProjectMultiClass(control_panel = {
									'save_result_switch': save_switch, # WARNING: Will overwrite existing
									'use_subsampled_df_switch': True,
									'use_subsampled_df_nk': nk,
									'use_featured_df_switch': True,
									'use_feature_set': feature_set,
									'feature_selection_switch': False,
									'cross_validation_switch': False,
									'cross_validation_repeat': None,
									'ml_process_on_test_data_switch': True,
									'ml_process_on_training_data_switch': False,
									'ml_process_on_ext_data_switch': False,
									'ml_algo': [algo_key, algo_val],
									'ml_algo_param_grid': None,
									'eval_score': None,
									'label_varname': 'ETHNICITY_RECAT',
									'random_state': 888,
									})
							obj.machine_learning_steps()

		if run_on_full_data:
			# Run once using unfeatured, full dataset
			if run_on_unfeatured_data:
				for algo_key, algo_val in ml_algo_dict.items():
					obj = ml_pipeline.MachineLearningNameEthnicityProjectMultiClass(control_panel = {
							'save_result_switch': save_switch, # WARNING: Will overwrite existing
							'use_subsampled_df_switch': False,
							'use_subsampled_df_nk': [],
							'use_featured_df_switch': True,
							'use_feature_set': [],
							'feature_selection_switch': False,
							'cross_validation_switch': False,
							'cross_validation_repeat': None,
							'ml_process_on_test_data_switch': True,
							'ml_process_on_training_data_switch': False,
							'ml_process_on_ext_data_switch': False,
							'ml_algo': [algo_key, algo_val],
							'ml_algo_param_grid': None,
							'eval_score': None,
							'label_varname': 'ETHNICITY_RECAT',
							'random_state': 888,
							})
					obj.machine_learning_steps()	

			# Run once using featured, full dataset
			if run_on_featured_data:
				for feature_set in feature_set_list:
					for algo_key, algo_val in ml_algo_dict.items():
						obj = ml_pipeline.MachineLearningNameEthnicityProjectMultiClass(control_panel = {
								'save_result_switch': save_switch, # WARNING: Will overwrite existing
								'use_subsampled_df_switch': False,
								'use_subsampled_df_nk': [],
								'use_featured_df_switch': True,
								'use_feature_set': feature_set,
								'feature_selection_switch': False,
								'cross_validation_switch': False,
								'cross_validation_repeat': None,
								'ml_process_on_test_data_switch': True,
								'ml_process_on_training_data_switch': False,
								'ml_process_on_ext_data_switch': False,
								'ml_algo': [algo_key, algo_val],
								'ml_algo_param_grid': None,
								'eval_score': None,
								'label_varname': 'ETHNICITY_RECAT',
								'random_state': 888,
								})
						obj.machine_learning_steps()

	if local_control_panel['done_switch']:
		hf.done_alert()

if __name__=='__main__':
	main(on_switch=False)