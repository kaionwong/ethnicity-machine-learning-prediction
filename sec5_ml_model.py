import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.metrics import make_scorer, f1_score, accuracy_score, balanced_accuracy_score, \
	recall_score, precision_score, roc_auc_score, auc, roc_curve, confusion_matrix, precision_recall_curve, average_precision_score

# Main class
######################################################################
class MachineLearning(ABC):
	def machine_learning_steps(self):
		self.import_processed_main_data()
		self.data_prep()
		self.feature_prep()
		self.feature_selection()
		self.feature_selection_result()
		self.cross_validation()
		self.cross_validation_result()
		self.ml_process()
		self.ml_process_result_on_test_data()
		self.ml_process_result_on_training_data()
		self.ml_process_result_on_external_data()

	@abstractmethod	
	def import_processed_main_data(self): pass
	@abstractmethod
	def data_prep(self): pass
	@abstractmethod
	def feature_prep(self): pass
	def feature_selection(self): pass
	def feature_selection_result(self): pass
	@abstractmethod
	def cross_validation(self): pass
	@abstractmethod
	def cross_validation_result(self): pass
	@abstractmethod
	def ml_process(self): pass
	@abstractmethod
	def ml_process_result_on_test_data(self): pass
	@abstractmethod
	def ml_process_result_on_training_data(self): pass
	@abstractmethod
	def ml_process_result_on_external_data(self): pass

	# Class' helper functions
	######################################################################
	def feature_use(self, feature_set=None) -> tuple:
		assert (feature_set in ['dummy', 'name_basic_only', 'sex_only', 'name_substring_only', 'name_numeric_only', 
			'name_metaphone_only', 'name_all', 'loc_basic_only', 'loc_sep_entity_only', 'loc_substring_only', 'loc_all', 
			'name_all_loc_all', 'name_all_loc_all_reduced', 'name_all_loc_all_sex_all', 'none']), \
			'Error: `feature_set` not specified correctly.'
		feature_set = feature_set.lower()
		var_tree = self.var_tree()
		if feature_set == 'dummy':
			var_num_list = var_tree['dummy']['final_num']
			var_cat_list = var_tree['dummy']['final_cat']
			var_liststring_list = []
			var_liststring1letter_list = []
		elif feature_set == 'none':
			var_num_list = []
			var_cat_list = []
			var_liststring_list = []
			var_liststring1letter_list = []			
		elif feature_set == 'sex_only':
			var_num_list = []
			var_cat_list = var_tree['sex']['final_cat']
			var_liststring_list = []	
			var_liststring1letter_list = []
		elif feature_set == 'name_basic_only':
			var_num_list = []
			var_cat_list = var_tree['name']['final_cat']['basic']	
			var_liststring_list = []	
			var_liststring1letter_list = []
		elif feature_set == 'name_substring_only':
			var_num_list = []
			var_cat_list = []
			var_liststring_list = var_tree['name']['final_cat']['substring_2_or_more_letters']
			var_liststring1letter_list = var_tree['name']['final_cat']['substring_1_letter']
		elif feature_set == 'name_numeric_only':
			var_num_list = var_tree['name']['final_num']
			var_cat_list = []
			var_liststring_list = []	
			var_liststring1letter_list = []
		elif feature_set == 'name_metaphone_only':
			var_num_list = []
			var_cat_list = []
			var_liststring_list = var_tree['name']['final_cat']['metaphone']
			var_liststring1letter_list = []	
		elif feature_set == 'name_all':
			var_num_list = var_tree['name']['final_num']
			var_cat_list = var_tree['name']['final_cat']['basic']
			var_liststring_list = var_tree['name']['final_cat']['substring_2_or_more_letters'] + var_tree['name']['final_cat']['metaphone']	
			var_liststring1letter_list = var_tree['name']['final_cat']['substring_1_letter']
		elif feature_set == 'loc_basic_only':
			var_num_list = []
			var_cat_list = var_tree['loc']['final_cat']['basic']
			var_liststring_list = []
			var_liststring1letter_list = []
		elif feature_set == 'loc_sep_entity_only':
			var_num_list = []
			var_cat_list = var_tree['loc']['final_cat']['separate_entity']
			var_liststring_list = []
			var_liststring1letter_list = []	
		elif feature_set == 'loc_substring_only':
			var_num_list = []
			var_cat_list = []
			var_liststring_list = var_tree['loc']['final_cat']['substring']
			var_liststring1letter_list = []
		elif feature_set == 'loc_all':
			var_num_list = []
			var_cat_list = var_tree['loc']['final_cat']['basic'] + var_tree['loc']['final_cat']['separate_entity']
			var_liststring_list = var_tree['loc']['final_cat']['substring']
			var_liststring1letter_list = []
		elif feature_set == 'name_all_loc_all':
			var_num_list = var_tree['name']['final_num']
			var_cat_list = var_tree['name']['final_cat']['basic'] + var_tree['loc']['final_cat']['basic'] + \
				var_tree['loc']['final_cat']['separate_entity']
			var_liststring_list = var_tree['name']['final_cat']['substring_2_or_more_letters'] + var_tree['name']['final_cat']['metaphone'] + \
				var_tree['loc']['final_cat']['substring']
			var_liststring1letter_list = var_tree['name']['final_cat']['substring_1_letter']
		elif feature_set == 'name_all_loc_all_reduced':
			var_num_list = []
			var_cat_list = var_tree['name']['final_cat']['basic'] + var_tree['loc']['final_cat']['basic']
			var_liststring_list = var_tree['name']['final_cat']['substring_2_or_more_letters'] + var_tree['name']['final_cat']['metaphone']
			var_liststring1letter_list = var_tree['name']['final_cat']['substring_1_letter']
		elif feature_set == 'name_all_loc_all_sex_all':
			var_num_list = var_tree['name']['final_num']
			var_cat_list = var_tree['name']['final_cat']['basic'] + var_tree['loc']['final_cat']['basic'] + \
				var_tree['loc']['final_cat']['separate_entity'] + var_tree['sex']['final_cat']
			var_liststring_list = var_tree['name']['final_cat']['substring_2_or_more_letters'] + var_tree['name']['final_cat']['metaphone'] + \
				var_tree['loc']['final_cat']['substring']
			var_liststring1letter_list = var_tree['name']['final_cat']['substring_1_letter']
		return var_num_list, var_cat_list, var_liststring_list, var_liststring1letter_list

	def var_tree(self) -> dict:
		var_tree = {
			'ethnic':
				{
					'original': ['ETHNICITY'],
					'final_cat': ['ETHNICITY_RECAT', 'ETHNICITY_RECAT_V2', 'ETHNICITY_RECAT_V3', 'ETHNICITY_RECAT_V4'],
					},
			'name':
				{
					'original': ['NAME'],
					'prepped': ['NAME_V2'],
					'final_cat': 
						{
							'basic': [
								'NAME_FIRST', 'NAME_MIDDLE', 'NAME_LAST', 'NAME_FIRST_FIRSTCHAR', 
								'NAME_FIRST_LASTCHAR', 'NAME_MIDDLE_FIRSTCHAR', 'NAME_MIDDLE_LASTCHAR', 
								'NAME_LAST_FIRSTCHAR', 'NAME_LAST_LASTCHAR'],
							'substring_2_or_more_letters': [
								'NAME_FIRST_2LETTER_SUBSTRINGS', 'NAME_FIRST_3LETTER_SUBSTRINGS', 'NAME_FIRST_4LETTER_SUBSTRINGS', 
								'NAME_FIRST_5LETTER_SUBSTRINGS', 'NAME_FIRST_6LETTER_SUBSTRINGS', 'NAME_MIDDLE_2LETTER_SUBSTRINGS', 
								'NAME_MIDDLE_3LETTER_SUBSTRINGS', 'NAME_MIDDLE_4LETTER_SUBSTRINGS', 'NAME_MIDDLE_5LETTER_SUBSTRINGS', 
								'NAME_MIDDLE_6LETTER_SUBSTRINGS', 'NAME_LAST_2LETTER_SUBSTRINGS', 'NAME_LAST_3LETTER_SUBSTRINGS', 
								'NAME_LAST_4LETTER_SUBSTRINGS', 'NAME_LAST_5LETTER_SUBSTRINGS', 'NAME_LAST_6LETTER_SUBSTRINGS', 
								],
							'substring_1_letter': [
								'NAME_FIRST_1LETTER_SUBSTRINGS', 'NAME_MIDDLE_1LETTER_SUBSTRINGS', 'NAME_LAST_1LETTER_SUBSTRINGS', 
								'NAME_FULL_1LETTER_SUBSTRINGS'],
							'metaphone': ['NAME_FULL_METAPHONE'],
							},
					'final_num': ['NAME_ENTITY_COUNT', 'NAME_TOTAL_LENGTH', 'NAME_AVG_LENGTH', 'NAME_VOWEL_COUNT', 'NAME_VOWEL_RATIO'],
					},
			'loc':
				{
					'original': ['LOC'],
					'prepped': ['LOC_V2'],
					'final_cat': 
						{
							'basic': ['LOC_DISTRICT_FULL'],
							'substring': ['LOC_ENTITY_LIST'],
							'separate_entity': ['LOC_PROVINCE', 'LOC_DISTRICT', 'LOC_DISTRICT_SUB'],
							},
					},
			'sex':
				{
					'original': ['SEX'],
					'prepped': ['SEX'],
					'final_cat': ['SEX'],
					},
			'dummy':
				{
					'final_num': ['DUMMY_INTEGER'],
					'final_cat': ['DUMMY_STRING'],
					},
			}
		return var_tree

	def label_tree(self) -> dict:
		label_tree = {
			'main_groups': ['fr', 'en', 'ir', 'sc', 'ab', 'rus', 'ch', 'it', 'ja', 'others'],
			'ab_groups': ['fn', 'metis', 'inuit'],
			'fn_tribal_groups': ['cree', 'ojibwa', 'blackfoot', 'micmac', 'iroquois', 'mohawk', 'nuu-chah-nulth', 'salish', 
									'algonquin', 'slavey', 'gitxsan', 'sioux', 'odawa', 'montagnais', 'oneida', 'six nations',
									'stoney', 'kootenay', 'eskimo'],
			'fn_language_groups': ['algonquian', 'iroquoian', 'wakashan', 'athapaskan', 'siouan', 'salish', 'tsimshian', 'kootenay'],
			}
		return label_tree

	def eval_score(self, choice=None):
		assert choice in [	'accuracy','balanced accuracy', 'macro f1 score',  'macro precision', 'macro recall',
							'macro roc auc', None], \
			'Error: Improper value for param `choice`'
		if choice == 'accuracy':
			return make_scorer(accuracy_score)
		elif choice == 'balanced accuracy':
			return make_scorer(balanced_accuracy_score)
		elif choice == 'macro f1 score':
			return make_scorer(f1_score, average='macro')
		elif choice == 'macro precision':
			return make_scorer(precision_score, average='macro')
		elif choice == 'macro recall':
			return make_scorer(recall_score, average='macro')
		elif choice == 'macro roc auc':
			return make_scorer(roc_auc_score, average='macro')

	def custom_classification_report(self, y_true, y_pred):
		tp, fn, fp, tn = confusion_matrix(y_true, y_pred).ravel()
		acc = (tp+tn)/(tp+tn+fp+fn)
		sen = (tp)/(tp+fn)
		sp = (tn)/(tn+fp)
		ppv = (tp)/(tp+fp)
		npv = (tn)/(tn+fn)
		f1 = 2*(sen*ppv)/(sen+ppv)
		fpr = (fp)/(fp+tn)
		tpr = (tp)/(tp+fn)
		return (	'2X2 confusion matrix:', ['TP', tp, 'FP', fp, 'FN', fn, 'TN', tn],
					'Accuracy:', round(acc, 3),
					'Sensitivity/Recall:', round(sen, 3),
					'Specificity:', round(sp, 3),
					'PPV/Precision:', round(ppv, 3),
					'NPV:', round(npv, 3),
					'F1-score:', round(f1, 3),
					'False positive rate:', round(fpr, 3),
					'True positive rate:', round(tpr, 3),
				)

	def auc_roc(self, y_true, y_pred_score):
		return ('AUC-ROC:', round(roc_auc_score(y_true, y_pred_score), 3))

	def avg_precision(self, y_true, y_pred_score, target_name):
		return ('Average precision:', round(average_precision_score(y_true, y_pred_score, pos_label=target_name), 3))

class ExternalData():
	def create_mock_external_data(self):
		data = {
					'NAME': 			['Xing Hai Long', 'Lee Ka Sing', 'Ling Ming Chui', 'Hiroyuki Sanada', 'Rich Francis', 
										'Jessica Harper', 'Aurélien Matthieu', 'Eileen Murphy', 'Michela Ricci', 'Murilo Silva', 
										'Andryey Petrov', 'Alban Smith'],
					'ETHNICITY': 		['Chinese', 'Chinese', 'Chinese', 'Japanese', 'Aboriginal', 'English', 'French', 'Irish',
										'Italy', 'Brazil', 'Russia', 'Scotland'],
					'SEX': 				['Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female', 'Female', 'Male', 'Male',
										'Male', 'Male'],
					'LOC': 				[', , ,'] * 12,
					'ETHNICITY_RECAT': 	['ch', 'ch', 'ch', 'ja', 'ab', 'en', 'fr', 'ir', 'it', 'other', 'rus', 'sc'],
					'AB_GROUP': 		[np.nan] * 12,
					'AB_TRIBE': 		[np.nan] * 12,
					'AB_LANG': 			[np.nan] * 12,
				}
		df = pd.DataFrame(data)
		return df