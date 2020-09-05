import sys
import numpy as np
import datetime
import helper_functions.helper_functions as hf
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
import sec5b_ml_model_binaryclass_pipeline as ml_pipeline

local_control_panel = {
	'done_switch': False,
}

# Main function
######################################################################
def main(on_switch=False):
	if on_switch:
		save_switch = False # WARNING: Will overwrite existing if True
		run_on_subsampled_data = True
		run_on_full_data = False
		run_on_unfeatured_data = False
		run_on_featured_data = True

		# Eg, full list >> [1, 5, 10, 50, 100, 500]
		nk_list = [5]
		'''Eg, Full list >> ['dummy', 'sex_only', 'name_basic_only', 'name_substring_only', 'name_numeric_only', 
							'name_metaphone_only', 'name_all', 'loc_basic_only', 'loc_sep_entity_only', 
							'loc_substring_only', 'loc_all', 'name_all_loc_all']'''
		feature_set_list = ['name_all_loc_all_reduced']
		ml_algo_dict = {	
							'LR': { 'clf': LogisticRegression(
										tol = 3.359818286283781e-05, 
										solver = 'liblinear', 
										penalty = 'l1', 
										max_iter = 4000, 
										class_weight = None, 
										C = 4.281332398719396
									)
									},
							'SVC_LINEAR': {	'clf': CalibratedClassifierCV(LinearSVC(
												penalty='l2',
												C=0.01)),
											},
							'NB': { 'clf': BernoulliNB(),
									},
						}
		# Eg, Full list >> ['ab', 'fn', 'metis', 'inuit', 'ch', 'ja', 'en', 'fr', 'ir', 'it', 'rus', 'sc', 'others']
		target_set_list = ['ch', 'ab']

		if run_on_subsampled_data:
			# Loop through subsampling n set with unfeatured data
			if run_on_unfeatured_data:
				for nk in nk_list:
					for algo_key, algo_val in ml_algo_dict.items():
						for target_label in target_set_list:
							print('>> Current time:', datetime.datetime.now())
							obj = ml_pipeline.MachineLearningNameEthnicityProjectBinaryClass(control_panel = {
									'save_result_switch': save_switch, # WARNING: Will overwrite existing
									'use_subsampled_df_switch': False, # WARNING: Switch to False in production
									'use_subsampled_df_nk': nk,
									'use_featured_df_switch': False,
									'use_feature_set': [],
									'feature_selection_switch': False,
									'cross_validation_switch': False,
									'ml_process_on_test_data_switch': True,
									'ml_process_on_training_data_switch': False,
									'ml_process_on_ext_data_switch': False,
									'ml_algo': [algo_key, algo_val],
									'ml_algo_param_grid': None,
									'binary_target_label': target_label, 
									'eval_score': None,
									'random_state': 888,
									})
							obj.machine_learning_steps()

			# Loop through feature set and subsampling n set
			if run_on_featured_data:
				for feature_set in feature_set_list:
					for nk in nk_list:
						for algo_key, algo_val in ml_algo_dict.items():
							for target_label in target_set_list:
								print('>> Current time:', datetime.datetime.now())
								obj = ml_pipeline.MachineLearningNameEthnicityProjectBinaryClass(control_panel = {
										'save_result_switch': save_switch, # WARNING: Will overwrite existing
										'use_subsampled_df_switch': False, # WARNING: Switch to False in production
										'use_subsampled_df_nk': nk,
										'use_featured_df_switch': True,
										'use_feature_set': feature_set,
										'feature_selection_switch': False,
										'cross_validation_switch': False,
										'ml_process_on_test_data_switch': True,
										'ml_process_on_training_data_switch': False,
										'ml_process_on_ext_data_switch': False,
										'ml_algo': [algo_key, algo_val],
										'ml_algo_param_grid': None,
										'binary_target_label': target_label, 
										'eval_score': None,
										'random_state': 888,
										})
								obj.machine_learning_steps()		

		if run_on_full_data:
			# Run once using unfeatured, full dataset
			if run_on_unfeatured_data:
				for feature_set in feature_set_list:
					for algo_key, algo_val in ml_algo_dict.items():
						for target_label in target_set_list:
							print('>> Current time:', datetime.datetime.now())
							obj = ml_pipeline.MachineLearningNameEthnicityProjectBinaryClass(control_panel = {
									'save_result_switch': save_switch, # WARNING: Will overwrite existing
									'use_subsampled_df_switch': False, # WARNING: Switch to False in production
									'use_subsampled_df_nk': 'none',
									'use_featured_df_switch': False,
									'use_feature_set': [],
									'feature_selection_switch': False,
									'cross_validation_switch': False,
									'ml_process_on_test_data_switch': True,
									'ml_process_on_training_data_switch': False,
									'ml_process_on_ext_data_switch': False,
									'ml_algo': [algo_key, algo_val],
									'ml_algo_param_grid': None,
									'binary_target_label': target_label, 
									'eval_score': None,
									'random_state': 888,
									})
							obj.machine_learning_steps()	

			# Run once using featured, full dataset
			if run_on_featured_data:
				for feature_set in feature_set_list:
					for algo_key, algo_val in ml_algo_dict.items():
						for target_label in target_set_list:
							print('>> Current time:', datetime.datetime.now())
							obj = ml_pipeline.MachineLearningNameEthnicityProjectBinaryClass(control_panel = {
									'save_result_switch': save_switch, # WARNING: Will overwrite existing
									'use_subsampled_df_switch': False, # WARNING: Switch to False in production
									'use_subsampled_df_nk': [],
									'use_featured_df_switch': True,
									'use_feature_set': feature_set,
									'feature_selection_switch': False,
									'cross_validation_switch': False,
									'ml_process_on_test_data_switch': True,
									'ml_process_on_training_data_switch': False,
									'ml_process_on_ext_data_switch': False,
									'ml_algo': [algo_key, algo_val],
									'ml_algo_param_grid': None,
									'binary_target_label': target_label, 
									'eval_score': None,
									'random_state': 888,
									})
							obj.machine_learning_steps()			

	if local_control_panel['done_switch']:
		hf.done_alert()

if __name__=='__main__':
	main(on_switch=True)