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

    mdf = pd.read_csv(join(dirname(__file__), 'columns', 'mandatory.csv'))
    cdf = pd.read_csv(join(dirname(__file__), 'columns', 'computable.csv'))
    
    rename_columns(df, mdf)
    rename_columns(df, cdf)

    df = df.reset_index()
    
    return df

def extract_condition(df, condition_name):
    df = df[df['M_Condition_Name'] == condition_name]
    return df


def describe(df):
    print('Found a totoal of %d trials'%len(df))
    conds = df['M_Condition_Name'].unique()
    print('Found %d conditions: %s'%(len(conds), str(conds)))
    ps = df['M_Participant_ID'].unique()
    print('Found %d participants: %s'%(len(ps), str(ps)))
    return df
