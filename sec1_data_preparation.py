import sys
from abc import ABC, abstractmethod
import pandas as pd
import data_transformer.data_transformer as dt
import helper_functions.helper_functions as hf
import os

# Main class
######################################################################
class DataPreparation(ABC):
	def __init__(self):
		self.control_panel = {
			'save_file_switch': True, # WARNING: Will overwrite existing files if True
			'df_subsampling_switch': False,  # WARNING: Switch to False in production
			'df_subsampling_n': 50000,
			'random_seed': 888,
			'df_remove_dup_switch': True,
			'df_remove_missing_switch': True,
			'result_printout_switch': True,
			'done_switch': False,
			}

	def data_preparation_steps(self):
		self._pandas_output_setting()
		self.dir_name()
		self.file_name()
		self.import_ref_data()
		self.import_main_data()
		self.resample_row()
		self.rename_var()
		self.remove_var()
		self.create_var()
		self.merge_datasets()
		self.remove_row_duplicate()
		self.remove_row_missing()
		self.sort_row()
		self.show_processed_data()
		self.save_processed_data()

	def _pandas_output_setting(self):
		'''Set pandas output display setting'''
		pd.set_option('display.max_rows', 500)
		pd.set_option('display.max_columns', 500)
		pd.set_option('display.width', 160)
		pd.options.mode.chained_assignment = None # default='warn'

	'''Set main and sub-directory'''
	@abstractmethod
	def dir_name(self):	pass
	'''Set key data files'''
	@abstractmethod
	def file_name(self): pass
	'''Import study reference info'''
	@abstractmethod
	def import_ref_data(self): pass
	'''Import needed data'''
	@abstractmethod
	def import_main_data(self):	pass
	'''Random data subsampling'''
	@abstractmethod
	def resample_row(self):	pass
	'''Rename variables'''
	def rename_var(self): pass
	'''Remove uneccesary variables'''
	def remove_var(self): pass
	'''Create new variables'''
	def create_var(self): pass
	'''Merge datasets'''
	def merge_datasets(self): pass
	'''Remove duplicated rows'''
	def remove_row_duplicate(self): pass
	'''Remove rows with missing information'''
	def remove_row_missing(self): pass
	'''Sorting rows by variables'''
	def sort_row(self): pass
	'''Visualize data'''
	def show_processed_data(self): pass
	'''Export processed data'''
	def save_processed_data(self): pass

	# Class helper functions
	######################################################################
	def df_info(self, *dfs, full_info=False):
		'''Print related df info'''
		full_info = self.control_panel['result_printout_switch']
		for df in dfs:
			print('length:', len(df))
			if full_info:
				print(df.info())
				print(df.head())

	def random_sampling(self, *dfs, set_replace=False) -> tuple:
		'''Setting default value of 'sampling with replacement'''
		for df in dfs:
			if (self.control_panel['df_subsampling_n'] > len(df)):
				print ("""\nWarning: Specified subsample size is larger than the total no. of row of some of the dataset. 
					As a result, resampling with replacement will be done to reach specified subsample size.""") 
				set_replace = True
		df_list = []
		for df in dfs:
			dt_obj = dt.Data_Transformer()
			df = dt_obj.random_n(df, n=self.control_panel['df_subsampling_n'], on_switch=self.control_panel['df_subsampling_switch'], 
				random_state=self.control_panel['random_seed'], replace=set_replace)
			df_list.append(df)
		if len(df_list) >= 2:
			return tuple(df_list) # convert list to tuple type
		else:
			return df_list[0]

class DataPreparationNameEthnicityProject(DataPreparation):
	def dir_name(self):
		self.main_dir = os.getcwd() # Setting: Main current directory
		self.raw_data_dir = self.main_dir+r'\data\raw\\'
		self.ref_data_dir = self.main_dir+r'\data\ref\\'
		self.processed_data_dir = self.main_dir+r'\data\processed\\'
		self.result_dir = self.main_dir+r'\result\\'

	def file_name(self):
		self.file_census1901 = 'Mock_Data.csv' # Refers to the name of the entire raw census data file (or mock data for testing)
		self.file_census1901_processed = 'CanadianCensus1901.csv'
		self.file_ethnicityRecatMap = 'Ethnicity_Recategorication_Map.csv'

	def import_ref_data(self):
		self.df_ethnicRecat = pd.read_csv(self.ref_data_dir+self.file_ethnicityRecatMap, encoding='utf-8', low_memory=False)

	def import_main_data(self):
		self.df_census1901 = pd.read_csv(self.raw_data_dir+self.file_census1901, encoding='utf-8', low_memory=False)
		self.df_census1901_len_original = len(self.df_census1901)

	def resample_row(self):
		if self.control_panel['df_subsampling_switch']:
			self.df_census1901 = self.random_sampling(self.df_census1901)
			self.df_census1901_len_resampled = len(self.df_census1901)

	def merge_datasets(self):
		self.df_census1901 = self.df_census1901.merge(self.df_ethnicRecat, on='ETHNICITY', how='left')

	def remove_row_duplicate(self): 
		if self.control_panel['df_remove_dup_switch']:
			self.df_census1901 = hf.df_remove_duplicates(self.df_census1901, ['NAME', 'ETHNICITY', 'SEX', 'LOC'])
			self.df_census1901_len_dup_removed = len(self.df_census1901)

	def remove_row_missing(self):
		if self.control_panel['df_remove_missing_switch']:
			self.df_census1901 = self.df_census1901.dropna(subset=['NAME', 'ETHNICITY', 'LOC'])
			self.df_census1901_len_missing_removed = len(self.df_census1901)

	def show_processed_data(self): 
		if self.control_panel['result_printout_switch']:
			self.df_info(self.df_census1901)

			describe_cat_var('SEX', self.df_census1901, 'SEX')
			describe_cat_var('ETHNICITY', self.df_census1901, 'ETHNICITY')

			print('/// Subject N Flow: ///')
			print('Census 1901 (N), original: {}'.format(self.df_census1901_len_original))
			if self.control_panel['df_subsampling_switch']:
				print('Census 1901 (N), after resampling: {}'.format(self.df_census1901_len_resampled))
			if self.control_panel['df_remove_dup_switch']:
				print('Census 1901 (N), after duplicates removed: {}'.format(self.df_census1901_len_dup_removed))
			if self.control_panel['df_remove_missing_switch']:
				print('Census 1901 (N), after records with missing values removed: {}'.format(self.df_census1901_len_missing_removed))

	def save_processed_data(self): 
		if self.control_panel['save_file_switch']:
			if self.control_panel['df_subsampling_switch']:
				self.df_census1901.to_csv('{}_Prepped_Sm_N={}_{}'.format(self.processed_data_dir, self.control_panel['df_subsampling_n'], 
					self.file_census1901_processed), sep=',', encoding='utf-8', index=False)
			elif self.control_panel['df_subsampling_switch']==False:
				self.df_census1901.to_csv('{}Prepped_{}'.format(self.processed_data_dir, self.file_census1901_processed), sep=',', 
					encoding='utf-8', index=False)

# Helper functions
######################################################################
def result_decor(func):
	def print_result(df_key, df, var_name):
		print('/////////////////////////////////////////')
		print('/////////////////////////////////////////')
		func(df_key, df, var_name)
		print('/////////////////////////////////////////')
		print('/////////////////////////////////////////'+'\n')
	return print_result

@result_decor
def describe_num_var(df_key, df, var_name):
	df_head_n = 5
	print('Dataset name: {}'.format(df_key))
	print('{}, min: {}'.format(var_name, df[var_name].min()))
	print('{}, max: {}'.format(var_name, df[var_name].max()))
	print('{}, 25%tile, 50%tile, 75%tile:\n{}'.format(var_name, str(df[var_name].quantile([.25, .5, .75]))))
	print('{}, top value counts:\n{}'.format(var_name, str(df[var_name].value_counts().head(df_head_n))))

@result_decor
def describe_cat_var(df_key, df, var_name):
	df_head_n = 5
	print('Dataset name: {}'.format(df_key))
	print('{}, top value counts:\n{}'.format(var_name, str(df[var_name].value_counts().head(df_head_n))))

# Main function
######################################################################
def main(on_switch=False):
	if on_switch:
		obj = DataPreparationNameEthnicityProject()
		obj.data_preparation_steps()

		if obj.control_panel['done_switch']:
			hf.done_alert()

if __name__ == '__main__':
	main(on_switch=False)