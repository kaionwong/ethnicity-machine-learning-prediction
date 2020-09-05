import sys
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import string
import re
import random
from metaphone import doublemetaphone
import helper_functions.helper_functions as hf
import sec1_data_preparation as data_prep
import sec2_prepped_data_import as prepped_data_import

local_control_panel = {
    'save_df_switch': True, # WARNING: Will overwrite existing if True
    'feature_gen_switch': True,
    'df_subsampling_switch': False,  # WARNING: Switch to False in production
    'df_subsampling_n': 1*1000, # WARNING: Use whole thousand(s)
    'random_seed': 888,
    'done_switch': False,
}

# Main class
######################################################################
class FeatureCreation(ABC):
    def feature_generation_steps(self):
        self.import_processed_main_data()
        self.data_prep()
        self.create_features(on_switch=local_control_panel['feature_gen_switch'])
        self.save_df(on_switch=local_control_panel['save_df_switch'])

    @abstractmethod
    def import_processed_main_data(self): pass
    @abstractmethod
    def data_prep(self): pass
    @abstractmethod
    def create_features(self): pass
    @abstractmethod
    def save_df(self): pass

class FeatureCreationNameEthnicityProject(FeatureCreation, data_prep.DataPreparationNameEthnicityProject):
    def __init__(self):
        super().__init__()
        super().dir_name()

    def import_processed_main_data(self):
        t_obj = prepped_data_import.PreppedDataImportNameEthnicityProject()
        t_obj.control_panel['df_subsampling_switch'] = local_control_panel['df_subsampling_switch']
        t_obj.control_panel['df_subsampling_n'] = local_control_panel['df_subsampling_n']
        t_obj.control_panel['random_seed'] = local_control_panel['random_seed']
        t_obj.prepped_data_import_steps()
        self.df_main = t_obj.return_df()

    def data_prep(self):
        self.format_all_var()
        self.format_ethnic_var()
        self.format_sex_var()
        self.format_name_var()
        self.format_loc_var()
        
    def create_features(self, on_switch=False):
        if on_switch:
            self.create_name_feature()
            self.create_loc_feature()
            self.create_dummy_feature()
        else:
            pass

    def save_df(self, on_switch=False):
        filename = 'Prepped_CanadianCensus1901'
        n_k = ('%.0f' % round(local_control_panel['df_subsampling_n']/1000, 0))

        if on_switch:
            if local_control_panel['feature_gen_switch']:
                if local_control_panel['df_subsampling_switch']:
                    self.df_main.to_csv(
                        '{}{}{}{}{}{}'.format(self.processed_data_dir, filename, '_FeatureGenerated', 
                        '_Seed'+str(local_control_panel['random_seed']), '_N'+n_k+'K', '.csv'), 
                        sep=',', encoding='utf-8', index=False)                    
                else:
                    self.df_main.to_csv('{}{}{}{}'.format(self.processed_data_dir, filename, '_FeatureGenerated', '.csv'), 
                        sep=',', encoding='utf-8', index=False)
            else:
                if local_control_panel['df_subsampling_switch']:
                    self.df_main.to_csv(
                        '{}{}{}{}{}{}'.format(self.processed_data_dir, filename, '_NoFeatureGenerated', 
                        '_Seed'+str(local_control_panel['random_seed']), '_N'+n_k+'K', '.csv'), 
                        sep=',', encoding='utf-8', index=False)            
                else:
                    self.df_main.to_csv('{}{}{}{}'.format(self.processed_data_dir, filename, '_NoFeatureGenerated', '.csv'), 
                        sep=',', encoding='utf-8', index=False)                

    # Class' helper functions
    ######################################################################
    def format_all_var(self):
        self.df_main = self.df_main.apply(lambda x: x.astype(str).str.lower())

    def format_ethnic_var(self):
        self.df_main['ETHNICITY_RECAT_V2'] = self.df_main.apply(map_two_columns, var_base='ETHNICITY_RECAT', 
            var_overwrite='AB_GROUP', axis=1)
        self.df_main['ETHNICITY_RECAT_V3'] = self.df_main.apply(map_two_columns, var_base='ETHNICITY_RECAT', 
            var_overwrite='AB_TRIBE', axis=1)
        self.df_main['ETHNICITY_RECAT_V4'] = self.df_main.apply(map_two_columns, var_base='ETHNICITY_RECAT', 
            var_overwrite='AB_LANG', axis=1)

        self.df_main['ETHNICITY_RECAT'] = self.df_main.apply(remove_sp_output, var='ETHNICITY_RECAT', axis=1)
        self.df_main['ETHNICITY_RECAT_V2'] = self.df_main.apply(remove_sp_output, var='ETHNICITY_RECAT_V2', axis=1)
        self.df_main['ETHNICITY_RECAT_V3'] = self.df_main.apply(remove_sp_output, var='ETHNICITY_RECAT_V3', axis=1)
        self.df_main['ETHNICITY_RECAT_V4'] = self.df_main.apply(remove_sp_output, var='ETHNICITY_RECAT_V4', axis=1)

        # Remove rows with missing values
        self.df_main = self.df_main[self.df_main['ETHNICITY_RECAT'].notnull()]        
        self.df_main = self.df_main[self.df_main['ETHNICITY_RECAT_V2'].notnull()]        

    def format_sex_var(self):
        self.df_main['SEX'] = self.df_main.apply(format_sex_var, axis=1)

    def format_name_var(self):
        self.df_main['NAME_V2'] = self.df_main.apply(remove_single_char, var='NAME', axis=1)
        self.df_main['NAME_V2'] = self.df_main.apply(remove_symbol, var='NAME_V2', axis=1)
        self.df_main['NAME_V2'] = self.df_main.apply(remove_number, var='NAME_V2', axis=1)
        self.df_main['NAME_V2'] = self.df_main.apply(remove_title, var='NAME_V2', axis=1)
        self.df_main['NAME_V2'] = self.df_main.apply(remove_double_space, var='NAME_V2', axis=1)
        self.df_main['NAME_V2'] = self.df_main.apply(remove_leading_space, var='NAME_V2', axis=1)
        self.df_main['NAME_V2'] = self.df_main.apply(remove_trailing_space, var='NAME_V2', axis=1)
        self.df_main = self.df_main[self.df_main['NAME_V2']!='']

    def format_loc_var(self):
        self.df_main['LOC_V2'] = self.df_main.apply(remove_french_words, var='LOC', axis=1)

    def create_name_feature(self):
        self.df_main['NAME_FIRST'] = self.df_main.apply(get_first_name, var='NAME_V2', axis=1)
        self.df_main['NAME_MIDDLE'] = self.df_main.apply(get_middle_name, var='NAME_V2', axis=1)
        self.df_main['NAME_LAST'] = self.df_main.apply(get_last_name, var='NAME_V2', axis=1)

        self.df_main['NAME_ENTITY_COUNT'] = self.df_main.apply(name_entity_count, var='NAME', axis=1)
        self.df_main['NAME_TOTAL_LENGTH'] = self.df_main.apply(get_word_length, var='NAME_V2', axis=1)
        self.df_main['NAME_AVG_LENGTH'] = self.df_main.apply(get_avg_word_length, var='NAME_V2', axis=1)
        self.df_main['NAME_VOWEL_COUNT'] = self.df_main.apply(vowel_count, var='NAME_V2', axis=1) 
        self.df_main['NAME_VOWEL_RATIO'] = self.df_main.apply(get_vowel_ratio, var='NAME_V2', axis=1) 

        self.df_main['NAME_FIRST_FIRSTCHAR'] = self.df_main.apply(get_edge_char, var='NAME_FIRST', edge='first', axis=1)
        self.df_main['NAME_FIRST_LASTCHAR'] = self.df_main.apply(get_edge_char, var='NAME_FIRST', edge='last', axis=1)
        self.df_main['NAME_MIDDLE_FIRSTCHAR'] = self.df_main.apply(get_edge_char, var='NAME_MIDDLE', edge='first', axis=1)
        self.df_main['NAME_MIDDLE_LASTCHAR'] = self.df_main.apply(get_edge_char, var='NAME_MIDDLE', edge='last', axis=1)
        self.df_main['NAME_LAST_FIRSTCHAR'] = self.df_main.apply(get_edge_char, var='NAME_LAST', edge='first', axis=1)
        self.df_main['NAME_LAST_LASTCHAR'] = self.df_main.apply(get_edge_char, var='NAME_LAST', edge='last', axis=1)

        self.df_main['NAME_FIRST_1LETTER_SUBSTRINGS'] = self.df_main.apply(get_substring, var='NAME_FIRST', substring_len=1, axis=1)
        self.df_main['NAME_FIRST_2LETTER_SUBSTRINGS'] = self.df_main.apply(get_substring, var='NAME_FIRST', substring_len=2, axis=1)
        self.df_main['NAME_FIRST_3LETTER_SUBSTRINGS'] = self.df_main.apply(get_substring, var='NAME_FIRST', substring_len=3, axis=1)
        self.df_main['NAME_FIRST_4LETTER_SUBSTRINGS'] = self.df_main.apply(get_substring, var='NAME_FIRST', substring_len=4, axis=1)
        self.df_main['NAME_FIRST_5LETTER_SUBSTRINGS'] = self.df_main.apply(get_substring, var='NAME_FIRST', substring_len=5, axis=1)
        self.df_main['NAME_FIRST_6LETTER_SUBSTRINGS'] = self.df_main.apply(get_substring, var='NAME_FIRST', substring_len=6, axis=1)
        self.df_main['NAME_MIDDLE_1LETTER_SUBSTRINGS'] = self.df_main.apply(get_substring, var='NAME_MIDDLE', substring_len=1, axis=1)
        self.df_main['NAME_MIDDLE_2LETTER_SUBSTRINGS'] = self.df_main.apply(get_substring, var='NAME_MIDDLE', substring_len=2, axis=1)
        self.df_main['NAME_MIDDLE_3LETTER_SUBSTRINGS'] = self.df_main.apply(get_substring, var='NAME_MIDDLE', substring_len=3, axis=1)
        self.df_main['NAME_MIDDLE_4LETTER_SUBSTRINGS'] = self.df_main.apply(get_substring, var='NAME_MIDDLE', substring_len=4, axis=1)
        self.df_main['NAME_MIDDLE_5LETTER_SUBSTRINGS'] = self.df_main.apply(get_substring, var='NAME_MIDDLE', substring_len=5, axis=1)
        self.df_main['NAME_MIDDLE_6LETTER_SUBSTRINGS'] = self.df_main.apply(get_substring, var='NAME_MIDDLE', substring_len=6, axis=1)
        self.df_main['NAME_LAST_1LETTER_SUBSTRINGS'] = self.df_main.apply(get_substring, var='NAME_LAST', substring_len=1, axis=1)
        self.df_main['NAME_LAST_2LETTER_SUBSTRINGS'] = self.df_main.apply(get_substring, var='NAME_LAST', substring_len=2, axis=1)
        self.df_main['NAME_LAST_3LETTER_SUBSTRINGS'] = self.df_main.apply(get_substring, var='NAME_LAST', substring_len=3, axis=1)
        self.df_main['NAME_LAST_4LETTER_SUBSTRINGS'] = self.df_main.apply(get_substring, var='NAME_LAST', substring_len=4, axis=1)
        self.df_main['NAME_LAST_5LETTER_SUBSTRINGS'] = self.df_main.apply(get_substring, var='NAME_LAST', substring_len=5, axis=1)
        self.df_main['NAME_LAST_6LETTER_SUBSTRINGS'] = self.df_main.apply(get_substring, var='NAME_LAST', substring_len=6, axis=1)

        self.df_main['NAME_FULL_1LETTER_SUBSTRINGS'] = self.df_main.apply(get_substring, var='NAME_V2', substring_len=1, axis=1)
        self.df_main['NAME_FULL_METAPHONE'] = self.df_main.apply(get_double_metaphone, var='NAME_V2', axis=1)

    def create_loc_feature(self):
        self.df_main['LOC_COUNTRY'] = self.df_main.apply(get_country, var='LOC_V2', axis=1)
        self.df_main['LOC_PROVINCE'] = self.df_main.apply(get_province, var='LOC_V2', axis=1)
        self.df_main['LOC_DISTRICT'] = self.df_main.apply(get_district, var='LOC_V2', axis=1)
        self.df_main['LOC_DISTRICT_SUB'] = self.df_main.apply(get_subdistrict, var='LOC_V2', axis=1)
        self.df_main['LOC_DISTRICT_FULL'] = self.df_main.apply(get_full_district_description, var='LOC_V2', axis=1)
        self.df_main['LOC_ENTITY_COUNT'] = self.df_main.apply(loc_entity_count, var='LOC_V2', axis=1)
        self.df_main['LOC_ENTITY_LIST'] = self.df_main.apply(loc_entity_simplified_list, var='LOC_V2', axis=1)

    def create_dummy_feature(self):
        self.df_main['DUMMY_STRING'] = self.df_main.apply(get_random_str, axis=1) 
        self.df_main['DUMMY_INTEGER'] = self.df_main.apply(get_random_int, axis=1)

# Local helper functions
######################################################################
# Ethnicity var
def map_two_columns(df, var_base, var_overwrite):
    if (df[var_overwrite] != np.nan) & (df[var_overwrite] != '') & (df[var_overwrite] != 'nan'):
        return df[var_overwrite]
    else:
        return df[var_base]

def remove_sp_output(df, var, sp_val='out'):
    if df[var] == sp_val:
        return np.nan
    else:
        return df[var]

# Sex var
def format_sex_var(df):
    if df['SEX'] == 'male':
        return 'm'
    elif df['SEX'] == 'female':
        return 'f'

# Name var
def remove_single_char(df, var):
    return re.sub(r'\b\w\b', '', df[var])

def remove_symbol(df, var):
    return re.sub(r'[?:;()/\*%^&#@$!`.]', '', df[var])

def remove_number(df, var):
    return re.sub(r'\d+','', df[var])

def remove_title(df, var):
    return ' '.join([i for i in df[var].split() if ((i!='mr') and (i!='mrs') and (i!='md')
        and (i!='ms') and (i!='sir') and (i!='madam') and (i!='dr') and (i!='msc') and (i!='frs') and (i!='phd')
        and (i!='rev') and (i!='pr') and (i!='prof') and (i!='adv') and (i!='mz') and (i!="ma'am") and (i!='st')
        and (i!='esq') and (i!='hon') and (i!='jr') and (i!='messrs') and (i!='mmes') and (i!='msgr') and (i!='rt hon'))])

def remove_double_space(df, var):
    return re.sub(r'  ', ' ', df[var])

def remove_leading_space(df, var):
    return re.sub(r'^[ \t]+', '', df[var])

def remove_trailing_space(df, var):
    return re.sub(r'[ \t]+$', '', df[var])

def get_first_name(df, var):
    name_entity = df[var].split()
    if (len(name_entity) == 0)|(len(name_entity) == 1):
        return ''
    else:
        return name_entity[0]

def get_middle_name(df, var):
    name_entity = df[var].split()
    if len(name_entity) >= 3:
        return ' '.join(name_entity[1:len(name_entity)-1])
    else:
        return ''

def get_last_name(df, var):
    name_entity = df[var].split()
    if len(name_entity) == 0:
        return ''
    else:
        return name_entity[-1]

def name_entity_count(df, var):
    name_entity = [x.strip() for x in df[var].split(' ')]
    return len(name_entity)

def get_word_length(df, var):
    space = 0
    for i in df[var]:
        if i == ' ':
            space += 1
    return len(df[var]) - space

def get_avg_word_length(df, var):
    space = 0
    for i in df[var]:
        if i == ' ':
            space += 1
    length = len(df[var]) - space
    name_entity = float(space + 1)
    avg = ('%.2f' % round(float(length/name_entity), 2))
    return avg

def vowel_count(df, var):
    return len(re.findall('[aeiou]', df[var]))

def get_vowel_ratio(df, var):
    vowel_count = len(re.findall('[aeiou]', df[var]))
    space = 0
    for i in df[var]:
        if i == ' ':
            space += 1
    total_length = len(df[var]) - space
    return ('%.2f' % round(vowel_count/total_length, 2))

def get_edge_char(df, var, edge=None):
    assert (edge=='first')|(edge=='last'), 'Error: Param `edge` is neither `first` nor `last`.'
    if edge == 'first':
        try:
            return df[var][0]
        except:
            return ''
    elif edge == 'last':
        try:
            return df[var][-1]
        except:
            return ''

def get_substring(df, var, substring_len) -> str:
    placeholder = []
    try:
        for i in range(0, len(df[var])):
            substring = df[var][i:i+substring_len]
            substring_cleaned = re.sub(' ', '', substring)
            substring_cleaned = substring_cleaned.strip()

            if len(substring_cleaned) == substring_len:
                placeholder.append(substring_cleaned)

        liststring = ', '.join(map(str, placeholder))
        return liststring
    except: 
        return ''        

def get_double_metaphone(df, var):
    placeholder = []
    name_entity = [x.strip() for x in df[var].split(' ')]
    try:
        for i in name_entity:
            placeholder.append(doublemetaphone(i)[0])

            if len(doublemetaphone(i)[1]) > 0:
                placeholder.append(doublemetaphone(i)[1])

        liststring = ', '.join(map(str, placeholder))
        return liststring

    except: 
        return ''

# Location var
def remove_french_words(df, var):
    return re.sub(r'/.*?(\)|(?: \d+)?,)', r'\1', df[var])

def get_country(df, var):
    loc_entity = [x.strip() for x in df[var].split(',')]
    return loc_entity[-1]

def get_province(df, var):
    loc_entity = [x.strip() for x in df[var].split(',')]
    return loc_entity[-2]

def get_district(df, var):
    loc_entity = [x.strip() for x in df[var].split(',')]
    return loc_entity[-3]

def get_subdistrict(df, var):
    loc_entity = [x.strip() for x in df[var].split(',')]
    return loc_entity[-4]

def get_full_district_description(df, var):
    loc_entity = [x.strip() for x in df[var].split(',') if x != '']
    return ', '.join(loc_entity[0:-1])    

def loc_entity_count(df, var):
    loc_entity = [x.strip() for x in df[var].split(',')]
    return len(loc_entity)

def loc_entity_simplified_list(df, var):
    loc_entity = [x.strip() for x in df[var].split(',')]
    loc_entity_simplified = [re.sub(r' \(.*?\)', '', x) for x in loc_entity if x != '']
    loc_list = loc_entity_simplified[0:-1]
    liststring = ', '.join(map(str, loc_list))
    return liststring

# Generic
def get_random_str(df):
    rands_chars = np.array(list(string.ascii_letters + string.digits), dtype=(np.str_, 1))
    n_chars = np.random.randint(5, 15)
    return ''.join(np.random.choice(rands_chars, n_chars))

def get_random_int(df):
    return np.random.randint(1, 50)

# Main function
######################################################################
def main(on_switch=False):
    if on_switch:
        obj = FeatureCreationNameEthnicityProject()
        obj.feature_generation_steps()

        if local_control_panel['done_switch']:
            hf.done_alert()

if __name__=='__main__':
    main(on_switch=False)