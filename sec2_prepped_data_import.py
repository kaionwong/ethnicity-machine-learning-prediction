import pandas as pd
import sec1_data_preparation as data_prep

# Main class
######################################################################
class PreppedDataImportNameEthnicityProject(data_prep.DataPreparationNameEthnicityProject):
	'''This class imports the data prepared by "sec1_data_preparation.py".'''
	def __init__(self): 
		# Overwriting parent's control_panel
		self.control_panel = {
			'df_subsampling_switch': False,  # WARNING: Switch to False in production
			'df_subsampling_n': 5000,
			'random_seed': 888,
			'filename': 'Prepped_CanadianCensus1901.csv',
			}

	def prepped_data_import_steps(self):
		self._pandas_output_setting()
		self.dir_name()
		self.import_main_data()
		self.resample_row()

	def import_main_data(self): 
		self.df_census1901 = pd.read_csv(self.processed_data_dir+self.control_panel['filename'], encoding='utf-8', low_memory=False)

	def return_df(self):
		return self.df_census1901