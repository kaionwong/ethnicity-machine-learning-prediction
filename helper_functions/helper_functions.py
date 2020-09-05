import ctypes

def clean_din_list(din_list):
	din_list = [x for x in din_list if str(x)!='nan'] # remove nan
	new_list = []
	for i in din_list:
		i = i.replace(' ', '') # remove empty space in string
		try:
			list_items = i.split(',')
			new_list = new_list + list_items
		except Exception:
			new_list.append(i)
	return list(set(new_list))

def remove_nonstring_items(my_list):
	return [i for i in my_list if type(i)==str]

def remove_leading_zeros(my_list):
	return [i.lstrip('0') for i in my_list]

def add_leading_zeros(my_list, desired_num_length):
	return [i.zfill(desired_num_length) for i in my_list]

def remove_duplicates(my_list) -> list:
	return list(set(my_list)) # removes duplicated list items

def df_remove_duplicates(df, col_list) -> object:
	'''Remove duplicated records (retaining first) when they have the same values on all the variables on `col_list`'''
	return df.drop_duplicates(subset=col_list, keep='first')

def df_remove_unknown_sex(df, sex_var) -> object:
	'''If values of SEX variable is not 'M' or 'F', the record will be removed'''
	return df.query('{}=="M" or {}=="F"'.format(sex_var, sex_var))

def decimal_place(num_input, decimal_place=2) -> float:
	return float('{:.{}f}'.format(float(num_input), decimal_place))

def p_value_recat(p_value:float) -> str:
	if p_value < 0.0001:
		p_value_cat = '<0.0001'
	elif (p_value >= 0.0001) & (p_value < 0.001):
		p_value_cat = '<0.001'
	elif (p_value >= 0.001) & (p_value < 0.01):
		p_value_cat = '<0.01'	
	elif (p_value >= 0.01) & (p_value < 0.05):
		p_value_cat = '<0.05'		
	else:
		p_value_cat = str(decimal_place(p_value, 2))
	return p_value_cat	

def add_number_to_list(my_list):
	# This func takes any list, and convert list items into int type and append to the original list
	temp_list1 = []
	temp_list2 = []
	for i in my_list:
		try:
			i=i.replace(' ', '')
			temp_list1.append(str(i))
		except: pass
	for i in temp_list1:
		try:
			temp_list2.append(int(i))
		except: pass
	return (list(set(temp_list1+temp_list2)))

def done_alert(): 
	ctypes.windll.user32.MessageBoxA(0, b'Hello there', b'Program done.', 0x1000)