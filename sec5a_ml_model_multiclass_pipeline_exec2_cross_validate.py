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
		cv_repeat = 1
		# Eg = ['name_all', 'name_all_loc_all', 'name_all_loc_all_reduced']
		feature_set_list = ['name_all_loc_all_reduced']
		# Eg = ['accuracy', 'macro f1 score', 'macro precision', 'macro recall']
		eval_score_list = ['macro f1 score']
		ml_algo_param_dict = \
						{	
							'LR_V1': {	'clf': LogisticRegression(),
										'param': {
											'logisticregression__solver': ['liblinear'],
											'logisticregression__penalty': ['l1', 'l2'],
											'logisticregression__C': np.logspace(-4, 4, 20),
											'logisticregression__tol': np.logspace(-5, 5, 20),
											'logisticregression__class_weight': [None, 'balanced'],
											'logisticregression__multi_class': ['ovr', 'auto'],
											'logisticregression__max_iter': [50, 1000, 4000, 20000],
										}},

							'LR_V2': {	'clf': LogisticRegression(),
										'param': {
											'logisticregression__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
											'logisticregression__penalty': ['none', 'l2'],
											'logisticregression__C': np.logspace(-4, 4, 20),
											'logisticregression__tol': np.logspace(-5, 5, 20),
											'logisticregression__class_weight': [None, 'balanced'],
											'logisticregression__multi_class': ['ovr', 'multinomial', 'auto'],
											'logisticregression__max_iter': [50, 1000, 4000, 20000],
										}},

							'SVC_LINEAR': {	'clf': OneVsRestClassifier(LinearSVC()),
								            'param': {
								            	'onevsrestclassifier__estimator__penalty': ['l2'],
								            	'onevsrestclassifier__estimator__loss': ['hinge', 'squared_hinge'],
								                'onevsrestclassifier__estimator__C': np.logspace(-4, 4, 20),
								                'onevsrestclassifier__estimator__tol': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
								                'onevsrestclassifier__estimator__class_weight': [None, 'balanced'],
								                'onevsrestclassifier__estimator__multi_class': ['ovr', 'crammer_singer'],
								                'onevsrestclassifier__estimator__max_iter': [50, 1000, 4000, 20000],
                                     		}},

							'SVC_NONLINEAR': {	'clf': OneVsRestClassifier(SVC()),
									            'param': {
										            'onevsrestclassifier__estimator__kernel': ['poly', 'rbf', 'sigmoid'],
										            'onevsrestclassifier__estimator__C': np.logspace(-4, 4, 20),
										            'onevsrestclassifier__estimator__tol': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
										            'onevsrestclassifier__estimator__class_weight': [None, 'balanced'],
										            'onevsrestclassifier__estimator__decision_function_shape': ['ovo', 'ovr'],
										            'onevsrestclassifier__estimator__max_iter': [50, 1000, 4000, 20000],
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
					for algo_key, algo_val in ml_algo_param_dict.items():
						for eval_score in eval_score_list:
							for i in range(1, cv_repeat+1):
								obj = ml_pipeline.MachineLearningNameEthnicityProjectMultiClass(control_panel = {
										'save_result_switch': save_switch, # WARNING: Will overwrite existing
										'use_subsampled_df_switch': True,
										'use_subsampled_df_nk': nk,
										'use_featured_df_switch': False,
										'use_feature_set': [],
										'feature_selection_switch': False,
										'cross_validation_switch': True,
										'cross_validation_repeat': i,
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
								for i in range(1, cv_repeat+1):
									obj = ml_pipeline.MachineLearningNameEthnicityProjectMultiClass(control_panel = {
											'save_result_switch': save_switch, # WARNING: Will overwrite existing
											'use_subsampled_df_switch': True,
											'use_subsampled_df_nk': nk,
											'use_featured_df_switch': True,
											'use_feature_set': feature_set,
											'feature_selection_switch': False,
											'cross_validation_switch': True,
											'cross_validation_repeat': i,
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
						for i in range(1, cv_repeat+1):
							obj = ml_pipeline.MachineLearningNameEthnicityProjectMultiClass(control_panel = {
									'save_result_switch': save_switch, # WARNING: Will overwrite existing
									'use_subsampled_df_switch': False,
									'use_subsampled_df_nk': [],
									'use_featured_df_switch': True,
									'use_feature_set': [],
									'feature_selection_switch': False,
									'cross_validation_switch': True,
									'cross_validation_repeat': i,
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
							for i in range(1, cv_repeat+1):
								obj = ml_pipeline.MachineLearningNameEthnicityProjectMultiClass(control_panel = {
										'save_result_switch': save_switch, # WARNING: Will overwrite existing
										'use_subsampled_df_switch': False,
										'use_subsampled_df_nk': [],
										'use_featured_df_switch': True,
										'use_feature_set': feature_set,
										'feature_selection_switch': False,
										'cross_validation_switch': True,
										'cross_validation_repeat': i,
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