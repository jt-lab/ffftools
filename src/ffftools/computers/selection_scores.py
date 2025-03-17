import numpy as np
import pandas as pd
from ..decorators import *

@register_computation('C_Selection_Target_Count')
@requires('M_Trial_Index', 'M_Selection_Role')
@check_protection('C_Selection_Target_Count')
@log_and_time
def compute_Selection_Target_Count(df):
    """Computes the cumulative target selection count within  each trial.

    Args:
        df (pd.DataFrame): fff-compatible DataFrame. 

    Returns:
        df (pd.DataFrame): Updated DataFrame with an additional column:
            - 'C_Selection_Target_Count': Cumulative count of 'target' selections within each trial.
    """
    df['C_Selection_Target_Count'] = (
        df[df['M_Selection_Role'] == 'target']
        .groupby(['M_Participant_ID', 'M_Condition_Name', 'M_Trial_Index'])
        .cumcount() + 1  # Start counting from 1
    )
    df['C_Selection_Target_Count'] = df.groupby(['M_Participant_ID', 'M_Condition_Name', 'M_Trial_Index'])['C_Selection_Target_Count'].transform(lambda x: x.ffill())

    return df

@register_computation('C_Selection_Nth_Last_Target')
@requires('C_Selection_Target_Count', 'C_Trial_Target_Count')
@check_protection('C_Selection_Nth_Last_Target')
@log_and_time
def compute_Selection_Nth_Last_Target(df):
    """Computes the Nth-last target selection index within each trial.

    Args:
        df (pd.DataFrame): fff-compatible DataFrame. 

    Returns:
        df (pd.DataFrame): Updated DataFrame with an additional column:
            - 'C_Selection_Nth_Last_Target': Position of the target selection when counted 
              from the last target selection (1-based index).

    Example:
        If a trial has 5 total target selections, they will be numbered as:
            - First target → 5 (farthest from last)
            - Second target → 4
            - Third target → 3
            - Fourth target → 2
            - Fifth target → 1 (last target)

    """
    df['C_Selection_Nth_Last_Target'] = (df['C_Trial_Target_Count'] 
                                         - df['C_Selection_Target_Count'] + 1).astype(int)
    return df

@register_computation('C_Selection_Target_Switch')
@requires('M_Selection_Type', 'M_Selection_Role', 'C_Selection_Target_Count')
@check_protection('C_Selection_Nth_Last_Target')
@log_and_time
def compute_Selection_Target_Switch(df):
    """Computes target type switches.
    
    Args:
        df (pd.DataFrame): fff-compatible DataFrame. 

    Returns:
        pd.DataFrame: The input DataFrame with an additional column:
            - 'C_Selection_Target_Switch': 
                - 1 if a switch occurred from the previous selection.
                - 0 if no switch occurred.
                - pd.NA for the first selection in a trial or distractors.
    """
    dfts = df.loc[df['M_Selection_Role'] == 'target']
    ts = (dfts['M_Selection_Type'].values[1:] != dfts['M_Selection_Type'].values[0:-1]).astype(int)

    df['C_Selection_Target_Switch'] = pd.NA
    df.loc[df['M_Selection_Role'] == 'target', 'C_Selection_Target_Switch'] = [pd.NA] + list(ts)
    df.loc[df['C_Selection_Target_Count'] == 1, 'C_Selection_Target_Switch'] = pd.NA

    return df


@register_computation('C_Selection_Inter-target_Length')
@requires('M_Selection_Role', 'M_Selection_X', 'M_Selection_Y', 'C_Trial_Target_Count')
@check_protection('C_Selection_Nth_Last_Target')
@log_and_time
def compute_Selection_ITL(df):
    """Computes the inter-target selection length (ITL) for target selections.

    Args:
        df (pd.DataFrame): fff-compatible DataFrame. 

    Returns:
        pd.DataFrame: The modified fff-compatible DataFrame with a new column:
            - 'C_Selection_Inter-target_Length': The Euclidean distance between 
              consecutive target selections. NA for first selections and non-target rows.

    """
    dfts = df.loc[df['M_Selection_Role'] == 'target']
    dx = dfts['M_Selection_X'].values[1:] - dfts['M_Selection_X'].values[0:-1]
    dy = dfts['M_Selection_Y'].values[1:] - dfts['M_Selection_Y'].values[0:-1]
    
    df.loc[df['M_Selection_Role'] == 'target', 'C_Selection_Inter-target_Length'] = \
        [pd.NA] + list(np.sqrt(dx * dx + dy * dy))

    df.loc[df['M_Selection_Role'] != 'target', 'C_Selection_Inter-target_Length'] = pd.NA
    df.loc[df['C_Selection_Target_Count'] == 1, 'C_Selection_Inter-target_Length'] = pd.NA

    return df


@register_computation('C_Selection_Inter-target_Time')
@requires('M_Selection_Role', 'M_Selection_Time', 'C_Selection_Target_Count')
@check_protection('C_Selection_Inter-target_Time')
@log_and_time
def compute_Selection_ITT(df):
    dfts = df.loc[df['M_Selection_Role'] == 'target']
    df['C_Selection_Inter-target_Time'] = pd.NA
    itts = dfts['M_Selection_Time'].values[1:] - dfts['M_Selection_Time'].values[0:-1]
    df.loc[df['M_Selection_Role'] == 'target', 'C_Selection_Inter-target_Time'] = np.array([np.nan] + list(itts), dtype=float)
    df.loc[df['C_Selection_Target_Count'] == 1, 'C_Selection_Inter-target_Time'] = pd.NA
    return(df)

@register_computation('C_Selection_Inter-target_LT_Ratio')
@requires('C_Selection_Inter-target_Length', 'C_Selection_Inter-target_Time')
@check_protection('C_Selection_Inter-target_LT_Ratio')
@log_and_time
def compute_Selection_IT_LT_Ratio(df):
    df['C_Selection_Inter-target_LT_Ratio'] = df['C_Selection_Inter-target_Length'] / df['M_Selection_Inter-target_Time']
    df.loc[df['C_Selection_Inter-target_Time'] == pd.NA, 'C_Selection_Inter-target_LT_Ratio'] = pd.NA
    return(df)