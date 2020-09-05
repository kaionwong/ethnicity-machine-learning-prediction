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
import sec5b_ml_model_binaryclass_pipeline as ml_pipeline

local_control_panel = {
	'done_switch': True,
}

# Main function
######################################################################
def main(on_switch=False):
	if on_switch:
		save_switch = False
		run_on_subsampled_data = True
		run_on_full_data = False
		run_on_unfeatured_data = False
		run_on_featured_data = True

		# Eg, full list >> [1, 5, 10, 50, 100, 500]
		nk_list = [5] 
		cv_repeat = 1
		'''Eg, Full list >> ['dummy', 'sex_only', 'name_basic_only', 'name_substring_only', 'name_numeric_only', 
							'name_metaphone_only', 'name_all', 'loc_basic_only', 'loc_sep_entity_only', 
							'loc_substring_only', 'loc_all', 'name_all_loc_all', 'name_all_loc_all_reduced']'''
		feature_set_list = ['name_all']
		eval_score_list = ['macro f1 score']
		# Eg, Full list >> ['ab', 'fn', 'metis', 'inuit', 'ch', 'ja', 'en', 'fr', 'ir', 'it', 'rus', 'sc', 'others']
		target_label_list = ['fn']
		ml_algo_param_dict = \
						{	
							'LR_V1': {	'clf': LogisticRegression(),
										'param': {
											'logisticregression__solver': ['liblinear'],
											'logisticregression__penalty': ['l1', 'l2'],
											'logisticregression__C': np.logspace(-4, 4, 20),
											'logisticregression__tol': np.logspace(-5, 5, 20),
											'logisticregression__class_weight': [None, 'balanced'],
											'logisticregression__max_iter': [50, 1000, 4000, 20000],
										}},

							'LR_V2': {	'clf': LogisticRegression(),
										'param': {
											'logisticregression__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
											'logisticregression__penalty': ['none', 'l2'],
											'logisticregression__C': np.logspace(-4, 4, 20),
											'logisticregression__tol': np.logspace(-5, 5, 20),
											'logisticregression__class_weight': [None, 'balanced'],
											'logisticregression__max_iter': [50, 1000, 4000, 20000],
										}},

							'SVC_LINEAR': {	'clf': LinearSVC(),
								            'param': {
								            	'linearsvc__penalty': ['l2'],
								            	'linearsvc__loss': ['hinge', 'squared_hinge'],
								                'linearsvc__C': np.logspace(-4, 4, 20),
								                'linearsvc__tol': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
								                'linearsvc__class_weight': [None, 'balanced'],
								                'linearsvc__max_iter': [50, 1000, 4000, 20000],
                                     		}},

							'SVC_NONLINEAR': {	'clf': SVC(),
									            'param': {
										            'svc__kernel': ['poly', 'rbf', 'sigmoid'],
										            'svc__C': np.logspace(-4, 4, 20),
										            'svc__tol': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
										            'svc__class_weight': [None, 'balanced'],
										            'svc__decision_function_shape': ['ovo', 'ovr'],
										            'svc__max_iter': [50, 1000, 4000, 20000],
                                        		}},

							'NB': {	'clf': BernoulliNB(),
									'param': {
										'bernoullinb__alpha': np.logspace(-4, 4, 20),
										'bernoullinb__binarize': [None, 0, .2, .4, .6, .8, 1],
										'bernoullinb__fit_prior': [True, False],
									}},
						}

		if run_on_subsampled_data:
			# Loop through subsampling n set with unfeatured data
			if run_on_unfeatured_data:
				for nk in nk_list:
					for target_label in target_label_list:
						for algo_key, algo_val in ml_algo_param_dict.items():
							for eval_score in eval_score_list:
								for i in range(1, cv_repeat+1):
									print('>> Current time:', datetime.datetime.now())
									obj = ml_pipeline.MachineLearningNameEthnicityProjectBinaryClass(control_panel = {
											'save_result_switch': save_switch, # WARNING: Will overwrite existing
											'use_subsampled_df_switch': False, # WARNING: Switch to False in production
											'use_subsampled_df_nk': nk,
											'use_featured_df_switch': False,
											'use_feature_set': [],
											'feature_selection_switch': False,
											'cross_validation_switch': True,
											'cross_validation_repeat': i,
											'ml_process_on_test_data_switch': False,
											'ml_process_on_training_data_switch': False,
											'ml_process_on_ext_data_switch': False,
											'ml_algo': None,
											'ml_algo_param_grid': [algo_key, algo_val],
											'binary_target_label': target_label, 
											'eval_score': eval_score,
											'random_state': 888,
											})
									obj.machine_learning_steps()

			# Loop through feature set and subsampling n set
			if run_on_featured_data:
				for feature_set in feature_set_list:
					for nk in nk_list:
						for target_label in target_label_list:
							for algo_key, algo_val in ml_algo_param_dict.items():
								for eval_score in eval_score_list:
									for i in range(1, cv_repeat+1):
										print('>> Current time:', datetime.datetime.now())
										obj = ml_pipeline.MachineLearningNameEthnicityProjectBinaryClass(control_panel = {
												'save_result_switch': save_switch, # WARNING: Will overwrite existing
												'use_subsampled_df_switch': False, # WARNING: Switch to False in production
												'use_subsampled_df_nk': nk,
												'use_featured_df_switch': True,
												'use_feature_set': feature_set,
												'feature_selection_switch': False,
												'cross_validation_switch': True,
												'cross_validation_repeat': i,
												'ml_process_on_test_data_switch': False,
												'ml_process_on_training_data_switch': False,
												'ml_process_on_ext_data_switch': False,
												'ml_algo': None,
												'ml_algo_param_grid': [algo_key, algo_val],
												'binary_target_label': target_label, 
												'eval_score': eval_score,
												'random_state': 888,
												})
										obj.machine_learning_steps()		

		if run_on_full_data:
			# Run once using unfeatured, full dataset
			if run_on_unfeatured_data:
				for feature_set in feature_set_list:
					for target_label in target_label_list:
						for algo_key, algo_val in ml_algo_param_dict.items():
							for eval_score in eval_score_list:
								for i in range(1, cv_repeat+1):
									print('>> Current time:', datetime.datetime.now())
									obj = ml_pipeline.MachineLearningNameEthnicityProjectBinaryClass(control_panel = {
											'save_result_switch': save_switch, # WARNING: Will overwrite existing
											'use_subsampled_df_switch': False, # WARNING: Switch to False in production
											'use_subsampled_df_nk': 'none',
											'use_featured_df_switch': False,
											'use_feature_set': [],
											'feature_selection_switch': False,
											'cross_validation_switch': True,
											'cross_validation_repeat': i,
											'ml_process_on_test_data_switch': False,
											'ml_process_on_training_data_switch': False,
											'ml_process_on_ext_data_switch': False,
											'ml_algo': None,
											'ml_algo_param_grid': [algo_key, algo_val],
											'binary_target_label': target_label, 
											'eval_score': eval_score,
											'random_state': 888,
											})
									obj.machine_learning_steps()	

			# Run once using featured, full dataset
			if run_on_featured_data:
				for feature_set in feature_set_list:
					for target_label in target_label_list:
						for algo_key, algo_val in ml_algo_param_dict.items():
							for eval_score in eval_score_list:
								for i in range(1, cv_repeat+1):
									print('>> Current time:', datetime.datetime.now())
									obj = ml_pipeline.MachineLearningNameEthnicityProjectBinaryClass(control_panel = {
											'save_result_switch': save_switch, # WARNING: Will overwrite existing
											'use_subsampled_df_switch': False, # WARNING: Switch to False in production
											'use_subsampled_df_nk': [],
											'use_featured_df_switch': True,
											'use_feature_set': feature_set,
											'feature_selection_switch': False,
											'cross_validation_switch': True,
											'cross_validation_repeat': i,
											'ml_process_on_test_data_switch': False,
											'ml_process_on_training_data_switch': False,
											'ml_process_on_ext_data_switch': False,
											'ml_algo': None,
											'ml_algo_param_grid': [algo_key, algo_val],
											'binary_target_label': target_label, 
											'eval_score': eval_score,
											'random_state': 888,
											})
									obj.machine_learning_steps()			

	if local_control_panel['done_switch']:
		hf.done_alert()

if __name__=='__main__':
	main(on_switch=True)