import sys
import warnings
import numpy as np
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import sec1_data_preparation as data_prep
import sec2_prepped_data_import as prepped_data_import
import sec3_feature_gen as feature_generate
import sec5_ml_model as ml_model

local_control_panel = {
	'done_switch': False,
}

# Main class
######################################################################
class MachineLearningNameEthnicityProjectMultiClass(ml_model.MachineLearning, 
	data_prep.DataPreparationNameEthnicityProject):
	def __init__(self, control_panel):
		assert control_panel is not None, 'Error: `control_panel` is not declared.'
		assert None not in [control_panel['save_result_switch'], control_panel['use_subsampled_df_switch'],
							control_panel['use_subsampled_df_nk'], control_panel['use_featured_df_switch'],
							control_panel['use_feature_set'], control_panel['feature_selection_switch'], 
							control_panel['random_state'], control_panel['cross_validation_switch'], 
							control_panel['ml_process_on_test_data_switch'], control_panel['ml_process_on_training_data_switch'], 
							control_panel['ml_process_on_ext_data_switch'], control_panel['label_varname']], \
							'Error: `control_panel` is not declared properly.'
		'''Only one True can be in `cross_validation_switch`, `feature_selection_switch`, and 
		`ml_process_on_test_data_switch`/`ml_process_on_training_data_switch`/`ml_process_on_ext_data_switch`. 
		While `ml_process_on_test_data_switch`/`ml_process_on_training_data_switch`/`ml_process_on_ext_data_switch` can be 
		all/any True'''
		true_count_in_ml_process = sum([control_panel['ml_process_on_test_data_switch'], 
			control_panel['ml_process_on_training_data_switch'], control_panel['ml_process_on_ext_data_switch']])
		if true_count_in_ml_process >= 1:
			assert (sum([bool(x) for x in [control_panel['cross_validation_switch'], control_panel['feature_selection_switch'], 
				control_panel['ml_process_on_test_data_switch'], control_panel['ml_process_on_training_data_switch'], 
				control_panel['ml_process_on_ext_data_switch']]]) == true_count_in_ml_process), \
				('Total number of True value is incorrect amongst: `cross_validation_switch`, `feature_selection_switch`, ' 
				 '`ml_process_on_test_data_switch`, `ml_process_on_training_data_switch`, `ml_process_on_ext_data_switch`.')
		else:
			assert (sum([bool(x) for x in [control_panel['cross_validation_switch'], control_panel['feature_selection_switch'], 
				control_panel['ml_process_on_test_data_switch'], control_panel['ml_process_on_training_data_switch'], 
				control_panel['ml_process_on_ext_data_switch']]]) == 1), \
				('Total number of True value is incorrect amongst: `cross_validation_switch`, `feature_selection_switch`, ' 
				 '`ml_process_on_test_data_switch`, `ml_process_on_training_data_switch`, `ml_process_on_ext_data_switch`.')
		assert control_panel['label_varname'] in ['ETHNICITY_RECAT', 'ETHNICITY_RECAT_V2', 'ETHNICITY_RECAT_V3'
			'ETHNICITY_RECAT_V4'], 'Error: `control_panel["label_varname"]` is not declared properly.'
		assert control_panel['eval_score'] in ['accuracy', 'balanced accuracy', 'macro f1 score', 'macro precision', 
			'macro recall', 'macro roc auc', None], 'Error: `control_panel["eval_score"]` is not declared properly.'
		super().__init__()
		super().dir_name()
		self.control_panel = control_panel
		# Conditionally re-assign
		if self.control_panel['use_subsampled_df_switch'] in [False, None]:
			self.control_panel['use_subsampled_df_nk'] = []
		if self.control_panel['use_featured_df_switch'] in [False, None]:
			self.control_panel['use_feature_set'] = []

	def import_processed_main_data(self):
		file_full_data_with_features = 'Prepped_CanadianCensus1901_FeatureGenerated.csv'
		file_full_data_without_features = 'Prepped_CanadianCensus1901_NoFeatureGenerated.csv'
		file_subsampled_data_with_features = 'Prepped_CanadianCensus1901_FeatureGenerated_Seed888_N{}K.csv'.format(
			self.control_panel['use_subsampled_df_nk'])
		file_subsampled_data_without_features = 'Prepped_CanadianCensus1901_NoFeatureGenerated_Seed888_N{}K.csv'.format(
			self.control_panel['use_subsampled_df_nk'])
		t_obj = prepped_data_import.PreppedDataImportNameEthnicityProject()
		t_obj.control_panel['df_subsampling_switch'] = False

		if self.control_panel['use_subsampled_df_switch']:
			if self.control_panel['use_featured_df_switch']:
				t_obj.control_panel['filename'] = file_subsampled_data_with_features
				t_obj.prepped_data_import_steps()
				self.df_census1901_withFeat = t_obj.return_df()

			else:
				t_obj.control_panel['filename'] = file_subsampled_data_without_features
				t_obj.prepped_data_import_steps()
				self.df_census1901_noFeat = t_obj.return_df()		
			
		else:
			if self.control_panel['use_featured_df_switch']:
				t_obj.control_panel['filename'] = file_full_data_with_features
				t_obj.prepped_data_import_steps()
				self.df_census1901_withFeat = t_obj.return_df()

			else:
				t_obj.control_panel['filename'] = file_full_data_without_features
				t_obj.prepped_data_import_steps()
				self.df_census1901_noFeat = t_obj.return_df()	

	def data_prep(self):
		var_tree = self.var_tree()
		# Declare which df to use. Can be `df_census1901_withFeat` or `df_census1901_noFeat`
		if self.control_panel['use_featured_df_switch']:
			self.df = self.df_census1901_withFeat.copy()
		else:
			self.df = self.df_census1901_noFeat.copy()

		# Declare feature set labels
		if self.control_panel['use_featured_df_switch']:
			self.feature_list_num, self.feature_list_cat, self.feature_list_liststring, self.feature_list_liststring1letter \
				= self.feature_use(self.control_panel['use_feature_set'])
		else:
			self.feature_list_num, self.feature_list_cat, self.feature_list_liststring, self.feature_list_liststring1letter \
				= [], var_tree['sex']['prepped'] + var_tree['name']['prepped'] + var_tree['loc']['prepped'], [], []
		self.feature_list_all = self.feature_list_num + self.feature_list_cat + self.feature_list_liststring + \
			self.feature_list_liststring1letter

		# Declare and (further) recategorize predicted outcome label
		label_tree = self.label_tree()
		if self.control_panel['label_varname'] == 'ETHNICITY_RECAT':
			self.label_list = label_tree['main_groups']
			self.pred_outcome = var_tree['ethnic']['final_cat'][0]
		elif self.control_panel['label_varname'] == 'ETHNICITY_RECAT_V2':
			self.label_list = label_tree['ab_groups']
			self.pred_outcome = var_tree['ethnic']['final_cat'][1]
		elif self.control_panel['label_varname'] == 'ETHNICITY_RECAT_V3':
			self.label_list = label_tree['fn_tribal_groups']
			self.pred_outcome = var_tree['ethnic']['final_cat'][2]
		elif self.control_panel['label_varname'] == 'ETHNICITY_RECAT_V4':
			self.label_list = label_tree['fn_language_groups']
			self.pred_outcome = var_tree['ethnic']['final_cat'][3]
		self.pred_outcome_vFinal = self.pred_outcome+'_VFINAL'
		self.df[self.pred_outcome_vFinal] = np.where(self.df[self.pred_outcome].isin(self.label_list), 
			self.df[self.pred_outcome], 'others')

		# Data split
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
			self.df[self.feature_list_all], self.df[self.pred_outcome_vFinal], 
			test_size=0.20, random_state=self.control_panel['random_state'])
		X_temp, self.X_dev, y_temp, self.y_dev = train_test_split(
			self.X_train, self.y_train, test_size=0.125, random_state=self.control_panel['random_state'])
		del X_temp, y_temp

	def feature_prep(self):
		transformer_num = Pipeline(steps=[
			('imputer', SimpleImputer(strategy='median')),
			('scaler', StandardScaler())])

		transformer_cat = Pipeline(steps=[
			('imputer', SimpleImputer(strategy='constant', fill_value='')),
			('onehotencoder', OneHotEncoder(handle_unknown='ignore'))])

		transformer_liststring_2orMoreLetters = Pipeline(steps=[
			('imputer', SimpleImputer(strategy='constant', fill_value='')),
			('ravel', RavelTransformer()),
			('countvectorizer', CountVectorizer(analyzer='word', tokenizer=None, preprocessor=None, 
				stop_words=None, max_features=5000))])

		transformer_liststring_1letter = Pipeline(steps=[
			('imputer', SimpleImputer(strategy='constant', fill_value='')),
			('ravel', RavelTransformer()),
			('countvectorizer', CountVectorizer(analyzer='char', tokenizer=None, preprocessor=None, 
				stop_words=None, max_features=5000))])

		liststring_dict = {
				'NAME_FIRST_2LETTER_SUBSTRINGS': [], 'NAME_FIRST_3LETTER_SUBSTRINGS': [], 'NAME_FIRST_4LETTER_SUBSTRINGS': [], 
				'NAME_FIRST_5LETTER_SUBSTRINGS': [], 'NAME_FIRST_6LETTER_SUBSTRINGS': [], 'NAME_MIDDLE_2LETTER_SUBSTRINGS': [], 
				'NAME_MIDDLE_3LETTER_SUBSTRINGS': [], 'NAME_MIDDLE_4LETTER_SUBSTRINGS': [], 'NAME_MIDDLE_5LETTER_SUBSTRINGS': [], 
				'NAME_MIDDLE_6LETTER_SUBSTRINGS': [], 'NAME_LAST_2LETTER_SUBSTRINGS': [], 'NAME_LAST_3LETTER_SUBSTRINGS': [], 
				'NAME_LAST_4LETTER_SUBSTRINGS': [], 'NAME_LAST_5LETTER_SUBSTRINGS': [], 'NAME_LAST_6LETTER_SUBSTRINGS': [], 
				'NAME_FIRST_1LETTER_SUBSTRINGS': [], 'NAME_MIDDLE_1LETTER_SUBSTRINGS': [], 'NAME_LAST_1LETTER_SUBSTRINGS': [], 
				'NAME_FULL_1LETTER_SUBSTRINGS': [], 'NAME_FULL_METAPHONE': [], 'LOC_ENTITY_LIST': [],
			}

		for key in liststring_dict:
			if key in self.feature_list_liststring:
				liststring_dict[key] = [key]

		self.preprocessor = ColumnTransformer(
			transformers=[
				('num', transformer_num, self.feature_list_num),
				('cat', transformer_cat, self.feature_list_cat),
				('liststring_firstName2Lsubstrings', transformer_liststring_2orMoreLetters, liststring_dict['NAME_FIRST_2LETTER_SUBSTRINGS']),
				('liststring_firstName3Lsubstrings', transformer_liststring_2orMoreLetters, liststring_dict['NAME_FIRST_3LETTER_SUBSTRINGS']),
				('liststring_firstName4Lsubstrings', transformer_liststring_2orMoreLetters, liststring_dict['NAME_FIRST_4LETTER_SUBSTRINGS']),
				('liststring_firstName5Lsubstrings', transformer_liststring_2orMoreLetters, liststring_dict['NAME_FIRST_5LETTER_SUBSTRINGS']),
				('liststring_firstName6Lsubstrings', transformer_liststring_2orMoreLetters, liststring_dict['NAME_FIRST_6LETTER_SUBSTRINGS']),
				('liststring_middleName2Lsubstrings', transformer_liststring_2orMoreLetters, liststring_dict['NAME_MIDDLE_2LETTER_SUBSTRINGS']),
				('liststring_middleName3Lsubstrings', transformer_liststring_2orMoreLetters, liststring_dict['NAME_MIDDLE_3LETTER_SUBSTRINGS']),
				('liststring_middleName4Lsubstrings', transformer_liststring_2orMoreLetters, liststring_dict['NAME_MIDDLE_4LETTER_SUBSTRINGS']),
				('liststring_middleName5Lsubstrings', transformer_liststring_2orMoreLetters, liststring_dict['NAME_MIDDLE_5LETTER_SUBSTRINGS']),
				('liststring_middleName6Lsubstrings', transformer_liststring_2orMoreLetters, liststring_dict['NAME_MIDDLE_6LETTER_SUBSTRINGS']),
				('liststring_lastName2Lsubstrings', transformer_liststring_2orMoreLetters, liststring_dict['NAME_LAST_2LETTER_SUBSTRINGS']),
				('liststring_lastName3Lsubstrings', transformer_liststring_2orMoreLetters, liststring_dict['NAME_LAST_3LETTER_SUBSTRINGS']),
				('liststring_lastName4Lsubstrings', transformer_liststring_2orMoreLetters, liststring_dict['NAME_LAST_4LETTER_SUBSTRINGS']),
				('liststring_lastName5Lsubstrings', transformer_liststring_2orMoreLetters, liststring_dict['NAME_LAST_5LETTER_SUBSTRINGS']),
				('liststring_lastName6Lsubstrings', transformer_liststring_2orMoreLetters, liststring_dict['NAME_LAST_6LETTER_SUBSTRINGS']),
				('liststring_firstName1Lsubstrings', transformer_liststring_1letter, liststring_dict['NAME_FIRST_1LETTER_SUBSTRINGS']),
				('liststring_middleName1Lsubstrings', transformer_liststring_1letter, liststring_dict['NAME_MIDDLE_1LETTER_SUBSTRINGS']),
				('liststring_lastName1Lsubstrings', transformer_liststring_1letter, liststring_dict['NAME_LAST_1LETTER_SUBSTRINGS']),
				('liststring_fullName1Lsubstrings', transformer_liststring_1letter, liststring_dict['NAME_FULL_1LETTER_SUBSTRINGS']),
				('liststring_metaphones', transformer_liststring_2orMoreLetters, liststring_dict['NAME_FULL_METAPHONE']),
				('liststring_locSubstrings', transformer_liststring_2orMoreLetters, liststring_dict['LOC_ENTITY_LIST']),
				])

	def feature_selection(self):
		'''To use dev dataset to help select/confirm the predictability of features/feature sets.'''
		if self.control_panel['feature_selection_switch']:
			self.score = self.eval_score(self.control_panel['eval_score'])
			pipe = make_pipeline(self.preprocessor, self.control_panel['ml_algo_param_grid'][1]['clf'])
			self.grid = GridSearchCV(pipe, self.control_panel['ml_algo_param_grid'][1]['param'], 
				n_jobs=-1, cv=5, scoring=self.score)
			self.grid.fit(self.X_dev, self.y_dev)

	def feature_selection_result(self):
		if self.control_panel['feature_selection_switch']:
			print('>> Setting:\n', self.control_panel)
			print('>> Model parameter:')
			print(self.grid.best_estimator_)
			print('>> Best selected parameter:')
			print(self.grid.best_params_)
			print('>> Model score ({}): %.3f'.format(self.control_panel['eval_score']) % self.grid.best_score_)
			print('//////////////////////\n'*2)

			if self.control_panel['save_result_switch']:
				with open('{}{}{}_{}_SubsampleIs{}_ScoreIs{}.txt'.format(self.result_dir, 'MultiClass_FeatureSelection_', 
					self.control_panel['ml_algo_param_grid'][0], self.control_panel['use_feature_set'],
					str(self.control_panel['use_subsampled_df_switch']), self.control_panel['eval_score'].capitalize().replace(
						' ', '')), 'w') as f:
					print('>> Setting:\n', self.control_panel, file=f)
					print('>> Model parameter:', file=f)
					print(self.grid.best_estimator_, file=f)
					print('>> Best selected parameter:', file=f)
					print(self.grid.best_params_, file=f)
					print('>> Model score ({}): %.3f'.format(self.control_panel['eval_score']) % self.grid.best_score_, file=f)
					print('//////////////////////\n'*2, file=f)

	def cross_validation(self):
		'''To use dev dataset via random search CV to help optimize hyperparameter and select ML algos.'''
		if self.control_panel['cross_validation_switch']:
			self.score = self.eval_score(self.control_panel['eval_score'])
			pipe = make_pipeline(self.preprocessor, self.control_panel['ml_algo_param_grid'][1]['clf'])
			self.grid = RandomizedSearchCV(pipe, param_distributions=self.control_panel['ml_algo_param_grid'][1]['param'], 
				n_jobs=-1, cv=5, scoring=self.score)
			self.grid.fit(self.X_dev, self.y_dev)

	def cross_validation_result(self):
		if self.control_panel['cross_validation_switch']:
			print('>> Setting:\n', self.control_panel)
			print('>> Best parameter:')
			print(self.grid.best_estimator_)
			print('>> Best selected parameter:')
			print(self.grid.best_params_)
			print('>> Model score ({}) in cross validation: %.3f'.format(self.control_panel['eval_score']) % 
				self.grid.best_score_)
			print('>> Model score ({}) in dev set: %.3f'.format(self.control_panel['eval_score']) % 
				self.grid.score(self.X_dev, self.y_dev))
			print('>> Model score ({}) in training set: %.3f'.format(self.control_panel['eval_score']) % 
				self.grid.score(self.X_train, self.y_train))
			print('>> Model score ({}) in test set: %.3f'.format(self.control_panel['eval_score']) % 
				self.grid.score(self.X_test, self.y_test))
			print('//////////////////////\n'*2)

			if self.control_panel['save_result_switch']:
				assert type(self.control_panel['cross_validation_repeat']) == int, 'Error: `control_panel` is not declared properly.'

				with open('{}{}{}_{}_SubsampleIs{}_ScoreIs{}_Take{}.txt'.format(self.result_dir, 'MultiClass_CV_BestParameters_', 
					self.control_panel['ml_algo_param_grid'][0], self.control_panel['use_feature_set'],
					str(self.control_panel['use_subsampled_df_switch']), self.control_panel['eval_score'].capitalize().replace(' ', ''), 
					self.control_panel['cross_validation_repeat']), 'w') as f:
					print('>> Setting:\n', self.control_panel, file=f)
					print('>> Best parameter:', file=f)
					print(self.grid.best_estimator_, file=f)
					print('>> Best selected parameter:', file=f)
					print(self.grid.best_params_, file=f)
					print('>> Model score ({}) in cross validation: %.3f'.format(self.control_panel['eval_score']) % 
						self.grid.best_score_, file=f)
					print('>> Model score ({}) in dev set: %.3f'.format(self.control_panel['eval_score']) % 
						self.grid.score(self.X_dev, self.y_dev), file=f)
					print('>> Model score ({}) in training set: %.3f'.format(self.control_panel['eval_score']) % 
						self.grid.score(self.X_train, self.y_train), file=f)
					print('>> Model score ({}) in test set: %.3f'.format(self.control_panel['eval_score']) % 
						self.grid.score(self.X_test, self.y_test), file=f)
					print('//////////////////////\n'*2, file=f)

	def ml_process(self):
		'''To use training set to train via selected ML algos and feature sets, and evaluate on test set.'''
		if (self.control_panel['ml_process_on_test_data_switch']|self.control_panel['ml_process_on_ext_data_switch']):
			self.pipe = make_pipeline(self.preprocessor, self.control_panel['ml_algo'][1]['clf'])
			self.pipe.fit(self.X_train, self.y_train)

			if self.control_panel['save_result_switch']:
				fname = '{}{}{}_{}_SubsampleIs{}.pickle'.format(self.result_dir, 'MultiClass_', 
					self.control_panel['ml_algo'][0], self.control_panel['use_feature_set'], 
					str(self.control_panel['use_subsampled_df_switch']))
				pickle.dump(self.pipe, open(fname, 'wb'))

	def ml_process_result_on_test_data(self):
		if self.control_panel['ml_process_on_test_data_switch']:
			if self.control_panel['use_subsampled_df_switch']:
				print('WARNING: Full data is not being used currently.\n')

			if self.pred_outcome_vFinal == 'ETHNICITY_RECAT_VFINAL':
				target_labels = ['ab', 'ch', 'en', 'fr', 'ir', 'it', 'ja', 'others', 'rus', 'sc']
			predicted_labels = self.pipe.predict(self.X_test)
			print('>>>> On test data <<<<')
			print('>> Setting:\n', self.control_panel, '\n')
			print('>> Class label count, in test set:\n', self.y_test.value_counts(), '\n')
			print('>> Class label count, predicted:\n', pd.Series(predicted_labels).value_counts(), '\n')
			print('>> Classification report:\n', classification_report(self.y_test, predicted_labels, target_names=target_labels))
			print('>> Confusion matrix:\n', confusion_matrix(self.y_test, predicted_labels, labels=target_labels))
			print('//////////////////////\n'*2)

			if self.control_panel['save_result_switch']:
				with open('{}{}{}_{}_SubsampleIs{}.txt'.format(self.result_dir, 'MultiClass_ML_Process_OnTestData_', 
					self.control_panel['ml_algo'][0], self.control_panel['use_feature_set'],
					str(self.control_panel['use_subsampled_df_switch'])), 'w') as f:
					print('>>>> On test data <<<<', file=f)
					print('>> Setting:\n', self.control_panel, '\n', file=f)
					print('>> Class label count, in test set:\n', self.y_test.value_counts(), '\n', file=f)
					print('>> Class label count, predicted:\n', pd.Series(predicted_labels).value_counts(), '\n', file=f)
					print('>> Classification report:\n', classification_report(self.y_test, predicted_labels, 
						target_names=target_labels), file=f)
					print('>> Confusion matrix:\n', confusion_matrix(self.y_test, predicted_labels, 
						labels=target_labels), file=f)
					print('//////////////////////\n'*2, file=f)

	def ml_process_result_on_training_data(self):
		if self.control_panel['ml_process_on_training_data_switch']:
			if self.control_panel['use_subsampled_df_switch']:
				print('WARNING: Full data is not being used currently.\n')

			if self.pred_outcome_vFinal == 'ETHNICITY_RECAT_VFINAL':
				target_labels = ['ab', 'ch', 'en', 'fr', 'ir', 'it', 'ja', 'others', 'rus', 'sc']
			predicted_labels = self.pipe.predict(self.X_train)
			print('>>>> On training data <<<<')
			print('>> Setting:\n', self.control_panel, '\n')
			print('>> Class label count, in training set:\n', self.y_train.value_counts(), '\n')
			print('>> Class label count, predicted:\n', pd.Series(predicted_labels).value_counts(), '\n')
			print('>> Classification report:\n', classification_report(self.y_train, predicted_labels, target_names=target_labels))
			print('>> Confusion matrix:\n', confusion_matrix(self.y_train, predicted_labels, labels=target_labels))
			print('//////////////////////\n'*2)

			if self.control_panel['save_result_switch']:
				with open('{}{}{}_{}_SubsampleIs{}.txt'.format(self.result_dir, 'MultiClass_ML_Process_OnTrainingData_', 
					self.control_panel['ml_algo'][0], self.control_panel['use_feature_set'],
					str(self.control_panel['use_subsampled_df_switch'])), 'w') as f:
					print('>>>> On training data <<<<', file=f)
					print('>> Setting:\n', self.control_panel, '\n', file=f)
					print('>> Class label count, in training set:\n', self.y_train.value_counts(), '\n', file=f)
					print('>> Class label count, predicted:\n', pd.Series(predicted_labels).value_counts(), '\n', file=f)
					print('>> Classification report:\n', classification_report(self.y_train, predicted_labels, 
						target_names=target_labels), file=f)
					print('>> Confusion matrix:\n', confusion_matrix(self.y_train, predicted_labels, 
						labels=target_labels), file=f)
					print('//////////////////////\n'*2, file=f)

	def ml_process_result_on_external_data(self):
		if self.control_panel['ml_process_on_ext_data_switch']:
			# Transform external data with the same feature generation steps
			feature_tranform_obj = feature_transform_external_data()
			feature_tranform_obj.import_processed_main_data()
			feature_tranform_obj.data_prep()
			feature_tranform_obj.create_features(on_switch=True)
			self.df_external_processed = feature_tranform_obj.df_main
			# Prepare ethnicity outcome variable
			self.df_external_processed[self.pred_outcome_vFinal] = np.where(
				self.df_external_processed[self.pred_outcome].isin(self.label_list), 
				self.df_external_processed[self.pred_outcome], 'others')
			# Data split
			self.X_ext_test = self.df_external_processed[self.feature_list_all]
			self.y_ext_test = self.df_external_processed[self.pred_outcome_vFinal]
			# Apply trained model and predict in external data
			if self.pred_outcome_vFinal == 'ETHNICITY_RECAT_VFINAL':
				target_labels = ['ab', 'ch', 'en', 'fr', 'ir', 'it', 'ja', 'others', 'rus', 'sc']			
			predicted_labels = self.pipe.predict(self.X_ext_test)
			# Print statements
			print('>>>> On external data <<<<')
			print('>> Setting:\n', self.control_panel, '\n')
			print('>> Class label count, in external test set:\n', self.y_ext_test.value_counts(), '\n')
			print('>> Class label count, predicted:\n', pd.Series(predicted_labels).value_counts(), '\n')
			print('>> Classification report:\n', classification_report(self.y_ext_test, predicted_labels, 
				target_names=target_labels))
			print('>> Confusion matrix:\n', confusion_matrix(self.y_ext_test, predicted_labels, 
				labels=target_labels))
			print('//////////////////////\n'*2)

			if self.control_panel['save_result_switch']:
				with open('{}{}{}_{}_SubsampleIs{}.txt'.format(self.result_dir, 'MultiClass_ML_Process_OnExtData_', 
					self.control_panel['ml_algo'][0], self.control_panel['use_feature_set'],
					str(self.control_panel['use_subsampled_df_switch'])), 'w') as f:
					print('>>>> On external data <<<<', file=f)
					print('>> Setting:\n', self.control_panel, '\n', file=f)
					print('>> Class label count, in external test set:\n', self.y_ext_test.value_counts(), '\n', file=f)
					print('>> Class label count, predicted:\n', pd.Series(predicted_labels).value_counts(), '\n', file=f)
					print('>> Classification report:\n', classification_report(self.y_ext_test, predicted_labels, 
						target_names=target_labels), file=f)
					print('>> Confusion matrix:\n', confusion_matrix(self.y_ext_test, predicted_labels, 
						labels=target_labels), file=f)
					print('//////////////////////\n'*2, file=f)

# Helper class
######################################################################
class RavelTransformer(BaseEstimator, TransformerMixin):
	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		return X.ravel()

class feature_transform_external_data(feature_generate.FeatureCreationNameEthnicityProject, ml_model.ExternalData):
	def import_processed_main_data(self):
		mock_external_data = self.create_mock_external_data()
		self.df_main = mock_external_data.copy()