import pandas as pd
import numpy as np
from os.path import exists, join
from glob import glob
from .utils import make_label 

def KristjanssonJohannessonThornton2014(store_path=None): 
	if store_path is not None:
		local_file=join(store_path, 'KristjanssonJohannessonThornton2014.fff.csv')
		if  exists(local_file):
			return(pd.read_csv(local_file))
	df =  pd.read_csv("https://ndownloader.figstatic.com/files/1564229", header=None )
	df[2] = [['Feature', 'Conjunction'][i] for i in df[2]]
	df[3] =  [['R/G', 'B/G', 'rSq/gDisk', 'gSq/rDisk'][i-1] for i in df[3]]

	df.columns = ['M_Participant_ID', 'M_Trial_Index', 'M_Condition_Name',
	              'S_Traget_Symbol', 'C_Selection_Target_Count', 'M_Selection_Type',
				  'M_Selection_X', 'M_Selection_Y', 'M_Selection_Time', 'C_Selection_Inter-target_Time',
				  'C_Selection_Target_Switch', 'S_Selection_Repetition_Count', 
				  'S_Selection_Touch_Distance', 'S_Selection_Run_Length', 'X']
	df.drop(columns=['X'], inplace=True)
	df['M_Selection_Role'] = 'target' #Only targets are saved in the data frames
	df['C_Selection_Inter-target_Time'].replace(0, np.nan, inplace=True)
	df['C_Selection_Target_Switch'] = 1 - df['C_Selection_Target_Switch']
	df.loc[df['C_Selection_Target_Count']==1, 'C_Selection_Target_Switch'] = np.nan
	if store_path is not None:
		df.to_csv(join(store_path, 'KristjanssonEtAl2014.fff.csv'))
	return df

def loadFromFolder(path, prepend='subject', extension='.csv', exclude_errors=True, sort=True):
	"""
    Loads and concatenates CSV files from a specified folder.

    This function reads a set of .csv files (e.g. created by experiment builders)
	and returns a fff-compatible pandas.DataFame with all the data.
	TODO: Make the function check for fff columns!
    
    Args:
        path (str): The root directory path.
        prepend (str, optional): A string to prepend to the filename pattern. 
            Defaults to 'subject'.
        extension (str, optional): The file extension to look for. Defaults to '.csv'.

    Returns:
        fff-compatible pandas.DataFrame.

    Example:
        >>> df = loadFromFolder('data', prepend='subject', extension='.csv')
        >>> print(df.head())
    """
	files = glob(join(path, prepend + '*' + extension))
	df = pd.concat(map(pd.read_csv, files), ignore_index=True)

	if sort:
		df.sort_values(by=['M_Participant_ID', 'M_Trial_Index', 'M_Selection_Index'], inplace=True)

	if exclude_errors:
		df = df.query("M_Selection_Role == 'target'")

	return df

def load(filename):
	return pd.read_csv(filename)
	
    
def describe(dataframe):
	print("Found %d participants in the dataset." % dataframe['M_Participant_ID'].nunique())
	print("Found %d conditions across all participants." % dataframe['M_Condition_Name'].nunique())
	print("Found %d selections across all participants." % dataframe['M_Selection_Index'].nunique())
	print("---------- Details -----------")
	for p in dataframe['M_Participant_ID'].unique():
		pdf = dataframe.query("M_Participant_ID == @p")
		print('------ Participant:' + str(p) + ': -------')
		print('Conditions: ' + str(pdf['M_Condition_Name'].unique()))
		print('Trials: ' + str(len(pdf['M_Trial_Index'].unique())))
		print('Selections: ' + str(len(pdf)))

def export_to_wide_format(df, score_columns, filename, category_column='M_Condition_Name', 
                          strip_fff=True, score_transforms=None):
    """
    Converts a fff-compatible pandas.DataFrame to wide format, where each row represents a
    single participant and each column represents a score-condition combination.
    (e.g., for JASP or SPSS).

    Args:
        df (pd.DataFrame): fff-compatible DataFrame with the data to convert.
        score_columns (list): List of column names representing the scores to export.
        filename (str): Path to save the output CSV file.
        category_column (str, optional): Name of the column that determines the condition to pivot on. Default is 'M_Condition_Name'.
        strip_fff (bool, optional): Whether to strip 'fff' from score names for cleaner labels. Default is True.
        score_transforms (dict, optional): Dictionary mapping score names to transformation functions. Default is None.

    Returns:
        fff-compatible pd.DataFrame: DataFrame in wide format with one row per participant and columns representing score-condition combinations.
    """

    wide_data = []
    for score in score_columns:
        for condition in df[category_column].unique():
            subset = df[df[category_column] == condition]
            
            score_values = subset.groupby(['M_Participant_ID'])[score].mean()

            if score_transforms and score in score_transforms:
                transform_func = score_transforms[score]
                score_values = score_values.apply(transform_func)

            if strip_fff:
                score_condition_column = f'{make_label(score)}_{condition}'
            else:
                score_condition_column = f'{score}_{condition}'

            score_values.name = score_condition_column
            wide_data.append(score_values)

    wide_df = pd.concat(wide_data, axis=1).reset_index()
    wide_df.to_csv(filename, index=False)

    return wide_df
