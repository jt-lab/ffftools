import numpy as np

def compute_trialwise(df, fun, colname):
    for p in df.M_Participant_ID.unique():
        maskp = df.M_Participant_ID == p
        for c in df[maskp].M_Condition_Name.unique():
            maskc = df.M_Condition_Name == c
            for t in df[(maskp & maskc)].M_Trial_Index.unique():
                maskt = df.M_Trial_Index == t
                df.loc[(maskp & maskc & maskt), colname] = \
                    fun(df[(maskp & maskc & maskt)])

def compute_Selection_Target_Count(df):
    def _compute_count(df):
        count = []
        last_t_index = -1
        for _, row in df.iterrows():
            if row['M_Trial_Index'] != last_t_index:
                tcount = 0
            if row['M_Selection_Role'] == 'target':
                tcount += 1
            last_t_index  = row['M_Trial_Index']
            count.append(tcount)
        return count
    compute_trialwise(df, _compute_count , 'C_Selection_Target_Count')
    return df

def compute_Selection_Nth_Last_Target(df):
    df['C_Selection_Nth_Last_Target'] = (df['C_Trial_Target_Count'] 
                                        - df['C_Selection_Target_Count'] + 1).astype(int)
    return df

def compute_Selection_Target_Switch(df):
    dfts = df.loc[df['M_Selection_Role'] == 'target']
    ts = (dfts['M_Selection_Type'].values[1:] != dfts['M_Selection_Type'].values[0:-1]).astype(int)
    df['C_Selection_Target_Switch'] = -1
    df.loc[df['M_Selection_Role'] == 'target', 'C_Selection_Target_Switch'] = [-1] + list(ts)
    df.loc[df['C_Selection_Target_Count'] == 1, 'C_Selection_Target_Switch'] = -1
    return df

def compute_Selection_ITL(df):
    dfts = df.loc[df['M_Selection_Role'] == 'target']
    dx = dfts['M_Selection_X'].values[1:] - dfts['M_Selection_X'].values[0:-1]
    dy = dfts['M_Selection_Y'].values[1:] - dfts['M_Selection_Y'].values[0:-1]
    df.loc[df['M_Selection_Role'] == 'target', 'C_Selection_Inter-target_Length'] = \
                                                    [-1] + list(np.sqrt(dx*dx+dy*dy))
    df.loc[df['M_Selection_Role'] != 'target', 'C_Selection_Inter-target_Length'] = -1
    df.loc[df['C_Selection_Target_Count'] == 1, 'C_Selection_Inter-target_Length'] = -1
    return(df)

def compute_Selection_ITT(df):
    dfts = df.loc[df['M_Selection_Role'] == 'target']
    df['C_Selection_Inter-target_Time'] = -1
    itts = dfts['M_Selection_Time'].values[1:] - dfts['M_Selection_Time'].values[0:-1]
    df.loc[df['M_Selection_Role'] == 'target', 'C_Selection_Inter-target_Time'] = np.array([-1] + list(itts), dtype=float)
    df.loc[df['C_Selection_Target_Count'] == 1, 'C_Selection_Inter-target_Time'] = -1
    return(df)

def compute_Selection_IT_LT_Ratio(df):
    #if 'C_Selection_Inter-target_Length' not in df:
    #    compute_Selection_ITL(df) #TODO: resolve auto computation via decorator!
    df['C_Selection_Inter-target_LT_Ratio'] = df['C_Selection_Inter-target_Length'] / df['M_Selection_Inter-target_Time']
    df.loc[df['M_Selection_Inter-target_Time'] == -1, 'C_Selection_Inter-target_Length'] = -1
    return(df)

def compute_Trial_Target_Count(df):
    compute_trialwise(df, lambda d: (d['M_Selection_Role'] == 'target').sum(), 'C_Trial_Target_Count')
    return df

def compute_Trial_BestR(df):
    def _compute_XR_YR(df, spatial_column):
        target_selection = df[df['M_Selection_Role'] == 'target']
        r = np.corrcoef(target_selection[spatial_column].values,
                        target_selection['C_Selection_Target_Count'].values)
        return(r[0,1])

    compute_trialwise(df, lambda x : _compute_XR_YR(x, 'M_Selection_X'), 'C_Trial_XR')
    compute_trialwise(df, lambda x : _compute_XR_YR(x, 'M_Selection_Y'), 'C_Trial_YR')
    df['C_Trial_BestR'] = df[["C_Trial_XR", "C_Trial_YR"]].abs().max(axis=1)
    df['C_Trial_Collection_Direction'] = 'None' # Will be overwritten virually always!
    df.loc[(df['C_Trial_XR'].abs() > df['C_Trial_YR'].abs()) & (df['C_Trial_XR'] < 0),\
        'C_Trial_Collection_Direction'] = 'left to right'
    df.loc[(df['C_Trial_XR'].abs() > df['C_Trial_YR'].abs()) & (df['C_Trial_XR'] > 0),\
        'C_Trial_Collection_Direction'] = 'right to left'
    df.loc[(df['C_Trial_XR'].abs() < df['C_Trial_YR'].abs()) & (df['C_Trial_XR'] < 0),\
        'C_Trial_Collection_Direction'] = 'top to bottom'
    df.loc[(df['C_Trial_XR'].abs() > df['C_Trial_YR'].abs()) & (df['C_Trial_XR'] > 0),\
        'C_Trial_Collection_Direction'] = 'bottom to top'
    
    return(df)
