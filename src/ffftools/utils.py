import pandas as pd
from os.path import join, dirname

def drop_unavailable(df, column):
    df = df[df[column] != -1]
    return df

def nicify(df):
    def rename_columns(df, columns_data):
        for _, row in columns_data.iterrows():
            if row['Column name'] in df.columns:
                df.rename(columns={row['Column name']: row['Label name']}, inplace=True)

    path = join(dirname(__file__), 'columns')
    mdf = pd.read_csv(join(path, 'mandatory.csv')).reset_index(drop=True)
    cdf = pd.read_csv(join(path, 'computable.csv')).reset_index(drop=True)

    rename_columns(df, mdf)
    rename_columns(df, cdf)
    
    return df

def extract_condition(df, name):
    df = df.query('M_Condition_Name == @name')
    return df

def extract_trial(df, index):
    df = df.query('M_Trial_Index == @index')
    return df

def describe(df):
    print('Found a totoal of %d trials'%len(df))
    conds = df['M_Condition_Name'].unique()
    print('Found %d conditions: %s'%(len(conds), str(conds)))
    ps = df['M_Participant_ID'].unique()
    print('Found %d participants: %s'%(len(ps), str(ps)))
    return df

def has_valid_name(column_name):
    return column_name.startswith(('M_', 'C_', '!C_', 'S_'))

def is_mandatory(column_name):
    return column_name.startswith('M_')
