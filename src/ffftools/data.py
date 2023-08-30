import pandas as pd
from os.path import exists, join

def KristjanssonEtAl2014(store_path=None): 
	if store_path is not None:
		local_file=join(store_path, 'KristjanssonEtAl2014.fff.csv')
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
	df['C_Selection_Inter-target_Time'].replace(0, -1, inplace=True)
	df['C_Selection_Target_Switch'] = 1 - df['C_Selection_Target_Switch']
	df.loc[df['C_Selection_Target_Count']==1, 'C_Selection_Target_Switch'] -1
	if store_path is not None:
		df.to_csv(join(store_path, 'KristjanssonEtAl2014.fff.csv'))
	return df


    
