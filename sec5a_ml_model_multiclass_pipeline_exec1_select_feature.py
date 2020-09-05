import sys
import numpy as np
import helper_functions.helper_functions as hf
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
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

		# Eg, Full nk_list >> [5, 50, 500]
		nk_list = [5] 
		'''Eg, Full feature_set_list >> ['dummy', 'sex_only', 'name_basic_only', 'name_substring_only', 'name_numeric_only', 
							'name_metaphone_only', 'name_all', 'loc_basic_only', 'loc_sep_entity_only', 
							'loc_substring_only', 'loc_all', 'name_all_loc_all', 'name_all_loc_all_reduced']'''
		feature_set_list = ['sex_only', 'name_all_loc_all_reduced']
		# Eg, Full eval_score_list >> ['accuracy', 'macro f1 score', 'macro precision', 'macro recall']
		eval_score_list = ['accuracy', 'macro f1 score']
		ml_algo_param_dict = \
						{	'LR': {	'clf': LogisticRegression(),
									'param': { # empty since we're not doing grid-search here
									}},
						}

		if run_on_subsampled_data:
			# Loop through subsampling n set with unfeatured data
			if run_on_unfeatured_data:
				for nk in nk_list:
					for algo_key, algo_val in ml_algo_param_dict.items():
						for eval_score in eval_score_list:
							obj = ml_pipeline.MachineLearningNameEthnicityProjectMultiClass(control_panel = {
									'save_result_switch': save_switch, # WARNING: Will overwrite existing
									'use_subsampled_df_switch': True,
									'use_subsampled_df_nk': nk,
									'use_featured_df_switch': False,
									'use_feature_set': 'none',
									'feature_selection_switch': True,
									'cross_validation_switch': False,
									'ml_process_on_test_data_switch': False,
									'ml_process_on_ext_data_switch': False,
                                    'ml_process_on_training_data_switch': False,
									'ml_algo': None,
									'ml_algo_param_grid': [algo_key, algo_val],
									'eval_score': eval_score,
									'label_varname': 'ETHNICITY_RECAT',
									'random_state': 888,
									})
							obj.machine_learning_steps()

			# Loop through feature set and subsampling n set
			if run_on_featured_data:
				for feature_set in feature_set_list:
					for nk in nk_list:
						for algo_key, algo_val in ml_algo_param_dict.items():
							for eval_score in eval_score_list:
								obj = ml_pipeline.MachineLearningNameEthnicityProjectMultiClass(control_panel = {
										'save_result_switch': save_switch, # WARNING: Will overwrite existing
										'use_subsampled_df_switch': True,
										'use_subsampled_df_nk': nk,
										'use_featured_df_switch': True,
										'use_feature_set': feature_set,
										'feature_selection_switch': True,
										'cross_validation_switch': False,
										'ml_process_on_test_data_switch': False,
										'ml_process_on_ext_data_switch': False,
                                        'ml_process_on_training_data_switch': False,
										'ml_algo': None,
										'ml_algo_param_grid': [algo_key, algo_val],
										'eval_score': eval_score,
										'label_varname': 'ETHNICITY_RECAT',
										'random_state': 888,
										})
								obj.machine_learning_steps()		

		if run_on_full_data:
			# Run once using unfeatured, full dataset
			if run_on_unfeatured_data:
				for algo_key, algo_val in ml_algo_param_dict.items():
					for eval_score in eval_score_list:
						obj = ml_pipeline.MachineLearningNameEthnicityProjectMultiClass(control_panel = {
								'save_result_switch': save_switch, # WARNING: Will overwrite existing
								'use_subsampled_df_switch': False,
								'use_subsampled_df_nk': 'none',
								'use_featured_df_switch': False,
								'use_feature_set': 'none',
								'feature_selection_switch': True,
								'cross_validation_switch': False,
								'ml_process_on_test_data_switch': False,
								'ml_process_on_ext_data_switch': False,
                                'ml_process_on_training_data_switch': False,
								'ml_algo': None,
								'ml_algo_param_grid': [algo_key, algo_val],
								'eval_score': eval_score,
								'label_varname': 'ETHNICITY_RECAT',
								'random_state': 888,
								})
						obj.machine_learning_steps()	

			# Run once using featured, full dataset
			if run_on_featured_data:
				for feature_set in feature_set_list:
					for algo_key, algo_val in ml_algo_param_dict.items():
						for eval_score in eval_score_list:
							obj = ml_pipeline.MachineLearningNameEthnicityProjectMultiClass(control_panel = {
									'save_result_switch': save_switch, # WARNING: Will overwrite existing
									'use_subsampled_df_switch': False,
									'use_subsampled_df_nk': 'none',
									'use_featured_df_switch': True,
									'use_feature_set': feature_set,
									'feature_selection_switch': True,
									'cross_validation_switch': False,
									'ml_process_on_test_data_switch': False,
									'ml_process_on_ext_data_switch': False,
                                    'ml_process_on_training_data_switch': False,
									'ml_algo': None,
									'ml_algo_param_grid': [algo_key, algo_val],
									'eval_score': eval_score,
									'label_varname': 'ETHNICITY_RECAT',
									'random_state': 888,
									})
							obj.machine_learning_steps()			

	if local_control_panel['done_switch']:
		hf.done_alert()

if __name__=='__main__':
	main(on_switch=False)