import pandas as pd
import numpy as np

# Data transformer transforms the underlying structure of the df into a new df 

class Data_Transformer:
	def conditional_row_filter(self, df, var, value_check):
		self.df = df
		self.var = var
		self.value_check = value_check

		df_temp = self.df.loc[self.df[self.var]==self.value_check]
		return df_temp

	# Display rows if certain conditions are met
	# No restriction on condition inputs
	# Condition should be a dictionary containing var name and corresponding value for filtering
	# i.e., df_dad_sm = transformer_obj.multi_conditional_row_filter(df_dad, {'Var1':1, 'Var2':1})
	def multi_conditional_row_filter(self, df, condition_dict):
		self.df = df
		self.condition_dict = condition_dict
		temp = self.df 

		for key, value in self.condition_dict.items():
			temp = temp.loc[temp[key]==value]
		return temp

	def sort_values(self, df, var_list):
		self.df = df
		self.var_list = var_list

		df_temp = self.df.sort_values(by=(self.var_list))
		return df_temp

	def pivot_table(self, df, index, agg_values, agg_func):
		self.df = df
		self.index = index
		self.agg_values = agg_values
		self.agg_func = agg_func

		df_temp = pd.pivot_table(self.df, index=self.index , values=self.agg_values, aggfunc=self.agg_func)
		df_temp2 = pd.DataFrame(df_temp.to_records())
		return df_temp2

	def random_n(self, df, n=None, frac=None, on_switch=False, random_state=None, replace=False):
		self.df = df
		self.n = n
		self.frac = frac
		self.on_switch = on_switch
		self.random_state = random_state
		self.replace = replace

		if self.on_switch == True:
			df_temp = self.df.sample(n=self.n, frac=self.frac, replace=self.replace, random_state=self.random_state)
			return df_temp
		else: return self.df

def filter_df_by_id(df, subject_id_var, subject_actual_id):
	return df[df[subject_id_var]==subject_actual_id]


