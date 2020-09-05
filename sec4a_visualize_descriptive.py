import sys
import warnings
import numpy as np
import pandas as pd
import time
from collections import Counter
import sec1_data_preparation as data_prep
import sec2_prepped_data_import as prepped_data_import
import sec4_visualize as visualize

local_control_panel = {
	'done_switch': False,
}

# Main class
######################################################################
class VisualizeNameEthnicityProjectDescriptive(data_prep.DataPreparationNameEthnicityProject, visualize.Visualize):
	def __init__(self, control_panel):
		super().__init__()
		super().dir_name()
		self.control_panel = control_panel

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
			t_obj.control_panel['filename'] = file_subsampled_data_with_features
			t_obj.prepped_data_import_steps()
			self.df = t_obj.return_df()

		else:
			t_obj.control_panel['filename'] = file_full_data_with_features
			t_obj.prepped_data_import_steps()
			self.df = t_obj.return_df()

	def import_additional_data(self):
		pass

	def data_prep(self):
		def transform_substring_val(x):
			if (x is np.nan) | (x is None) | (x is ''):
				return []
			else:
				return x.split(', ')

		def merge_list_columns(xs):
			return list(xi for x in xs for xi in x)

		# Create an aggregated df by by_varname
		@timeit
		def df_census_aggregation(by_varname):
			aggregations = {
				'ETHNICITY_RECAT_COPY': 'count',
				'NAME_ENTITY_COUNT': [min, max, 'mean', np.std], 
				'NAME_TOTAL_LENGTH': [min, max, 'mean', np.std], 
				'NAME_AVG_LENGTH': [min, max, 'mean', np.std], 
				'NAME_VOWEL_COUNT': [min, max, 'mean', np.std], 
				'NAME_VOWEL_RATIO': [min, max, 'mean', np.std], 
				'NAME_FULL_METAPHONE': 'sum', # 'sum' here means it will append all the list item values with duplicates
				'NAME_FIRST': lambda x: list(x),
				'NAME_MIDDLE': lambda x: list(x),
				'NAME_LAST': lambda x: list(x),
				'LOC_V2': lambda x: list(x),
				}
			for pos in name_pos:
				for i in range(1, 7):
					aggregations['{}_{}LETTER_SUBSTRINGS'.format(pos, str(i))] = 'sum'
			df_agg = df_temp.groupby(by_varname, as_index=False).agg(aggregations)
			df_agg['NAME_ALL_NLETTER_SUBSTRINGS_ALL'] = np.empty((len(df_agg), 0)).tolist()

			substring_col_list = []
			for pos in name_pos:
				for i in range(1, 7):
					substring_col_list.append('{}_{}LETTER_SUBSTRINGS'.format(pos, str(i)))
			firstname_substring_col_list = []
			for i in range(1, 7):
				firstname_substring_col_list.append('NAME_FIRST_{}LETTER_SUBSTRINGS'.format(str(i)))			
			middlename_substring_col_list = []
			for i in range(1, 7):
				middlename_substring_col_list.append('NAME_MIDDLE_{}LETTER_SUBSTRINGS'.format(str(i)))	
			lastname_substring_col_list = []
			for i in range(1, 7):
				lastname_substring_col_list.append('NAME_LAST_{}LETTER_SUBSTRINGS'.format(str(i)))	

			df_agg['NAME_ALL_NLETTER_SUBSTRINGS_ALL'] = df_agg[substring_col_list].apply(
				merge_list_columns, axis=1)
			df_agg['NAME_FIRST_NLETTER_SUBSTRINGS_ALL'] = df_agg[firstname_substring_col_list].apply(
				merge_list_columns, axis=1)
			df_agg['NAME_MIDDLE_NLETTER_SUBSTRINGS_ALL'] = df_agg[middlename_substring_col_list].apply(
				merge_list_columns, axis=1)
			df_agg['NAME_LAST_NLETTER_SUBSTRINGS_ALL'] = df_agg[lastname_substring_col_list].apply(
				merge_list_columns, axis=1)
			return df_agg

		df_temp = self.df.copy()
		df_temp['DUMMY_INDEX'] = 1
		df_temp['ETHNICITY_RECAT_COPY'] = df_temp['ETHNICITY_RECAT']

		# Reformat comma-separated string values
		name_pos = ['NAME_FIRST', 'NAME_MIDDLE', 'NAME_LAST']
		for pos in name_pos:
			for i in range(1, 7):
				df_temp['{}_{}LETTER_SUBSTRINGS'.format(pos, str(i))] = df_temp[
				'{}_{}LETTER_SUBSTRINGS'.format(pos, str(i))].apply(transform_substring_val)
		df_temp['NAME_FULL_METAPHONE'.format(str(i))] = df_temp[
			'NAME_FULL_METAPHONE'.format(str(i))].apply(transform_substring_val)

		self.df_agg_by_noGroup = df_census_aggregation('DUMMY_INDEX')
		self.df_agg_by_mainEthGroup = df_census_aggregation('ETHNICITY_RECAT')
		self.df_agg_by_threeAbGroup = df_census_aggregation('ETHNICITY_RECAT_V2')

	def describe_data(self):
		if self.control_panel['save_result_switch']:
			orig_stdout = sys.stdout
			f = open('{}Descriptive_Results.txt'.format(self.result_dir), 'w')
			sys.stdout = f

		if self.control_panel['describe_df_switch']:
			self.basic_df_description(self.df, show_rows=True, show_unique_string_val=True)
			print('//////////////////////////////')

		@timeit
		def describe_numeric_col(df, group_var=None):
			if group_var:
				print(df[[group_var, 'ETHNICITY_RECAT_COPY', 'NAME_ENTITY_COUNT', 'NAME_TOTAL_LENGTH',
					'NAME_AVG_LENGTH', 'NAME_VOWEL_COUNT', 'NAME_VOWEL_RATIO']], '\n')
			else:
				print(df[['ETHNICITY_RECAT_COPY', 'NAME_ENTITY_COUNT', 'NAME_TOTAL_LENGTH',
					'NAME_AVG_LENGTH', 'NAME_VOWEL_COUNT', 'NAME_VOWEL_RATIO']], '\n')

		@timeit
		def describe_substring_col(df, by_group=None, n_most_common=15):
			if by_group:
				group_list = df[by_group].tolist()
			else:
				group_list = ['all subjects']

			group_counter = 0
			for group in group_list:
				print('>>> Group:', group)
				if by_group:
					df_curr = df[df[by_group]==group].copy()
				else:
					df_curr = df.copy()

				name_pos = ['NAME_ALL', 'NAME_FIRST', 'NAME_MIDDLE', 'NAME_LAST']
				substring_col_list = []
				combined_substring_col_list = []
				for pos in name_pos[1:]:
					for i in range(1, 7):
						substring_col_list.append('{}_{}LETTER_SUBSTRINGS'.format(pos, str(i)))
					substring_col_list.append('NAME_FULL_METAPHONE')

				for pos in name_pos:
					combined_substring_col_list.append('{}_NLETTER_SUBSTRINGS_ALL'.format(pos))
				name_col_list = ['NAME_FIRST', 'NAME_MIDDLE', 'NAME_LAST']
				name_col_list.append('LOC_V2')

				if by_group:
					for i in substring_col_list:
						print('Variable: {}'.format(i))
						list_extract = df_curr[i]['sum'][group_counter]
						print('Value counts (top {}):'.format(n_most_common), 
							Counter(list_extract).most_common(n_most_common))
						print('Unique value counts:', len(set(list_extract)), '\n')

					for i in combined_substring_col_list:
						print('Variable: {}'.format(i))
						list_extract = df_curr[i][group_counter]
						print('Value counts (top {}):'.format(n_most_common), 
							Counter(list_extract).most_common(n_most_common))
						print('Unique value counts:', len(set(list_extract)), '\n')

					for i in name_col_list:
						print('Variable: {}'.format(i))
						list_extract = df_curr[i]['<lambda>'][group_counter]
						print('Value counts (top {}):'.format(n_most_common), 
							Counter(list_extract).most_common(n_most_common))
						print('Unique value counts:', len(set(list_extract)), '\n')

				else:
					for i in substring_col_list:
						print('Variable: {}'.format(i))
						list_extract = df_curr[i]['sum'][0]
						print('Value counts (top {}):'.format(n_most_common), 
							Counter(list_extract).most_common(n_most_common))
						print('Unique value counts:', len(set(list_extract)), '\n')

					for i in combined_substring_col_list:
						print('Variable: {}'.format(i))
						list_extract = df_curr[i][0]
						print('Value counts (top {}):'.format(n_most_common), 
							Counter(list_extract).most_common(n_most_common))
						print('Unique value counts:', len(set(list_extract)), '\n')

					for i in name_col_list:
						print('Variable: {}'.format(i))
						list_extract = df_curr[i]['<lambda>'][0]
						print('Value counts (top {}):'.format(n_most_common), 
							Counter(list_extract).most_common(n_most_common))
						print('Unique value counts:', len(set(list_extract)), '\n')

				group_counter += 1
			print()

		if self.control_panel['describe_col_switch']:
			describe_numeric_col(self.df_agg_by_noGroup)
			describe_numeric_col(self.df_agg_by_mainEthGroup, group_var='ETHNICITY_RECAT')
			describe_numeric_col(self.df_agg_by_threeAbGroup, group_var='ETHNICITY_RECAT_V2')
			describe_substring_col(self.df_agg_by_noGroup, by_group=None, n_most_common=50)
			describe_substring_col(self.df_agg_by_mainEthGroup, by_group='ETHNICITY_RECAT', n_most_common=50)
			describe_substring_col(self.df_agg_by_threeAbGroup, by_group='ETHNICITY_RECAT_V2', n_most_common=50)

		if self.control_panel['save_result_switch']:
			sys.stdout = orig_stdout
			f.close()

	def visualize_data(self):
		pass

# Helper function
######################################################################
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1)
        else:
            print ('Execution time: %r  %2.2f s' % (method.__name__, (te - ts) * 1))
        return result
    return timed

# Main function
######################################################################
def main(on_switch=False):
	if on_switch:
		save_switch = False
		run_on_full_data = True
		run_on_subsampled_data = False
		nk = 5

		if run_on_subsampled_data:
			obj = VisualizeNameEthnicityProjectDescriptive(control_panel = {
				'save_result_switch': save_switch, # WARNING: Will overwrite existing
				'use_subsampled_df_switch': True,
				'describe_df_switch': True,
				'describe_col_switch': True,
				'use_subsampled_df_nk': nk,
				})
			obj.data_visualization_steps()

		if run_on_full_data:
			obj = VisualizeNameEthnicityProjectDescriptive(control_panel = {
				'save_result_switch': save_switch, # WARNING: Will overwrite existing
				'use_subsampled_df_switch': False,
				'describe_df_switch': True,
				'describe_col_switch': True,
				'use_subsampled_df_nk': 'none',
				})
			obj.data_visualization_steps()

if __name__=='__main__':
	main(on_switch=False)