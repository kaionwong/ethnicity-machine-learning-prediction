from abc import ABC, abstractmethod
import pandas as pd

local_control_panel = {
	'done_switch': False,
}

# Main class
######################################################################
class Visualize(ABC):
	def data_visualization_steps(self):
		self.import_processed_main_data()
		self.import_additional_data()
		self.data_prep()
		self.describe_data()
		self.visualize_data()

	@abstractmethod	
	def import_processed_main_data(self): pass
	@abstractmethod
	def import_additional_data(self): pass
	@abstractmethod
	def data_prep(self): pass
	@abstractmethod
	def describe_data(self): pass
	@abstractmethod
	def visualize_data(self): pass

	# Class' helper functions
	######################################################################
	def basic_df_description(self, df, show_rows=False, show_unique_string_val=False):
		print('Number of rows:', len(df), '\n')
		print('Numeric variables:', df.dtypes[df.dtypes!='object'].index, '\n')
		print('Categorical/string variables:', df.dtypes[df.dtypes=='object'].index, '\n')
		print(df.info(), '\n')
		print(df.describe(), '\n')

		if show_rows:
			print(df.head(), '\n')

		if show_unique_string_val:
			for cat_var in df.dtypes[df.dtypes=='object'].index:
				print('Number of unique values in {}: {}'.format(cat_var, df[cat_var].nunique()))
			print('\n')
