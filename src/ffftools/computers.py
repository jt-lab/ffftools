import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from .algorithms.sequence import sts
from .decorators import *
from tqdm.auto import tqdm

def compute_trialwise(df, fun, colname):
    #with tqdm(total=len(df)) as pbar:
    for p in df.M_Participant_ID.unique():
        maskp = df.M_Participant_ID == p
        for c in df[maskp].M_Condition_Name.unique():
            maskc = df.M_Condition_Name == c
            for t in df[(maskp & maskc)].M_Trial_Index.unique():
                maskt = df.M_Trial_Index == t
                value = fun(df[(maskp & maskc & maskt)])
                if not np.isscalar(value) and (len(value) != len(df.loc[(maskp & maskc & maskt)])):
                    # TODO: Needs fixing!
                    #df[colname] = df[colname].astype(object) # TODO: only when needed
                    # Manually broadcast tuples
                    value = [value] * len(df.loc[(maskp & maskc & maskt)])
                #print(df.loc[(maskp & maskc & maskt)])
                df.loc[(maskp & maskc & maskt), colname] = value
#               pbar.update(1)

@register_computation('C_Selection_Target_Count')
@requires('M_Trial_Index', 'M_Selection_Role')
@check_protection('C_Selection_Target_Count')
@log_and_time
def compute_Selection_Target_Count(df):
    #TODO Spped up!
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

@register_computation('C_Selection_Nth_Last_Target')
@requires('C_Selection_Target_Count', 'C_Trial_Target_Count')
@check_protection('C_Selection_Nth_Last_Target')
@log_and_time
def compute_Selection_Nth_Last_Target(df):
    df['C_Selection_Nth_Last_Target'] = (df['C_Trial_Target_Count'] 
                                        - df['C_Selection_Target_Count'] + 1).astype(int)
    return df

@register_computation('C_Selection_Target_Switch')
@requires('M_Selection_Type', 'M_Selection_Role', 'C_Selection_Target_Count')
@check_protection('C_Selection_Nth_Last_Target')
@log_and_time
def compute_Selection_Target_Switch(df):
    dfts = df.loc[df['M_Selection_Role'] == 'target']
    ts = (dfts['M_Selection_Type'].values[1:] != dfts['M_Selection_Type'].values[0:-1]).astype(int)
    df['C_Selection_Target_Switch'] = pd.NA
    df.loc[df['M_Selection_Role'] == 'target', 'C_Selection_Target_Switch'] = [pd.NA] + list(ts)
    df.loc[df['C_Selection_Target_Count'] == 1, 'C_Selection_Target_Switch'] = pd.NA
    return df

@register_computation('C_Selection_Inter-target_Length')
@requires('M_Selection_Role', 'M_Selection_X', 'M_Selection_X', 'C_Trial_Target_Count')
@check_protection('C_Selection_Nth_Last_Target')
@log_and_time
def compute_Selection_ITL(df):
    dfts = df.loc[df['M_Selection_Role'] == 'target']
    dx = dfts['M_Selection_X'].values[1:] - dfts['M_Selection_X'].values[0:-1]
    dy = dfts['M_Selection_Y'].values[1:] - dfts['M_Selection_Y'].values[0:-1]
    df.loc[df['M_Selection_Role'] == 'target', 'C_Selection_Inter-target_Length'] = \
                                                    [pd.NA] + list(np.sqrt(dx*dx+dy*dy))
    df.loc[df['M_Selection_Role'] != 'target', 'C_Selection_Inter-target_Length'] = pd.NA
    df.loc[df['C_Selection_Target_Count'] == 1, 'C_Selection_Inter-target_Length'] = pd.NA
    return(df)

@register_computation('C_Selection_Inter-target_Time')
@requires('M_Selection_Role', 'M_Selection_Time', 'C_Trial_Target_Count')
@check_protection('C_Selection_Nth_Last_Target')
@log_and_time
def compute_Selection_ITT(df):
    dfts = df.loc[df['M_Selection_Role'] == 'target']
    df['C_Selection_Inter-target_Time'] = pd.NA
    itts = dfts['M_Selection_Time'].values[1:] - dfts['M_Selection_Time'].values[0:-1]
    df.loc[df['M_Selection_Role'] == 'target', 'C_Selection_Inter-target_Time'] = np.array([pd.NA] + list(itts), dtype=float)
    df.loc[df['C_Selection_Target_Count'] == 1, 'C_Selection_Inter-target_Time'] = pd.NA
    return(df)

@register_computation('C_Selection_Inter-target_LT_Ratio')
@requires('C_Selection_Inter-target_Length', 'M_Selection_Inter-target_Time')
@check_protection('C_Selection_Nth_Last_Target')
@log_and_time
def compute_Selection_IT_LT_Ratio(df):
    df['C_Selection_Inter-target_LT_Ratio'] = df['C_Selection_Inter-target_Length'] / df['M_Selection_Inter-target_Time']
    df.loc[df['M_Selection_Inter-target_Time'] == pd.NA, 'C_Selection_Inter-target_LT_Ratio'] = pd.NA
    return(df)

@register_computation('C_Trial_Target_Count')
@requires('M_Selection_Role')
@check_protection('C_Trial_Target_Count')
@log_and_time
def compute_Trial_Target_Count(df):
    compute_trialwise(df, lambda d: (d['M_Selection_Role'] == 'target').sum(), 'C_Trial_Target_Count')
    return df

@register_computation('C_Trial_XR', 'C_Trial_YR', 'C_Trial_BestR', 'C_Trial_Collection_Direction')
@requires('M_Selection_Role', 'M_Selection_X', 'M_Selection_Y','C_Selection_Target_Count')
@check_protection('C_Trial_Target_Count')
@log_and_time
def compute_Trial_BestR(df):
    """Computes the Best-R value for a given trial, as described by Woods et al. (2013).

    The Best-R metric assesses the correlation between selection positions and target selection counts 
    to determine the dominant movement direction. 

    Assumes that the negative Y-axis points upwards.

    Args:
        fff-compatible df (pd.DataFrame): Required columns:
            - 'M_Selection_Role': Role of the selection (e.g., 'target').
            - 'M_Selection_X': X-coordinate of the selection.
            - 'M_Selection_Y': Y-coordinate of the selection.

    Returns:
         fff-compatible df  (pd.DataFrame): Additional computed columns:
            - 'C_Trial_XR': Correlation of selections along the X-axis.
            - 'C_Trial_YR': Correlation of selections along the Y-axis.
            - 'C_Trial_BestR': Maximum absolute correlation value (Best-R).
            - 'C_Trial_Collection_Direction': Estimated movement direction ('left to right', 
              'right to left', 'top to bottom', 'bottom to top').

    Warning:
        Best-R calculation assumes a negative Y-axis pointing upwards.

    References:
        Woods, A. J., Göksun, T., Chatterjee, A., Zelonis, S., Mehta, A., & Smith, S. E. (2013). 
        The development of organized visual search. *Neuropsychologia, 51*(13), 2956–2967. 
        https://doi.org/10.1016/j.neuropsychologia.2013.10.001
    """
    logger.warning("Attention: Best-r calculation assumes negative-y-axis pointing up!")

    def _compute_XR_YR(df, spatial_column):
        """Computes the correlation between the given spatial column and target selection counts."""
        target_selection = df[df['M_Selection_Role'] == 'target']
        r = np.corrcoef(target_selection[spatial_column].values,
                        target_selection['C_Selection_Target_Count'].values)
        return r[0, 1]

    compute_trialwise(df, lambda x: _compute_XR_YR(x, 'M_Selection_X'), 'C_Trial_XR')
    compute_trialwise(df, lambda x: _compute_XR_YR(x, 'M_Selection_Y'), 'C_Trial_YR')

    df['C_Trial_BestR'] = df[["C_Trial_XR", "C_Trial_YR"]].abs().max(axis=1)

    df['C_Trial_Collection_Direction'] = 'None'  # Will be overwritten virtually always!
    df.loc[(df['C_Trial_XR'].abs() > df['C_Trial_YR'].abs()) & (df['C_Trial_XR'] > 0),
           'C_Trial_Collection_Direction'] = 'left to right'
    df.loc[(df['C_Trial_XR'].abs() > df['C_Trial_YR'].abs()) & (df['C_Trial_XR'] < 0),
           'C_Trial_Collection_Direction'] = 'right to left'
    df.loc[(df['C_Trial_XR'].abs() < df['C_Trial_YR'].abs()) & (df['C_Trial_YR'] < 0),
           'C_Trial_Collection_Direction'] = 'top to bottom'
    df.loc[(df['C_Trial_XR'].abs() < df['C_Trial_YR'].abs()) & (df['C_Trial_YR'] > 0),
           'C_Trial_Collection_Direction'] = 'bottom to top'

    return df

@register_computation('C_Trial_Tau', 'C_Trial_AbsTau')
@requires('M_Selection_Role')
@check_protection('C_Trial_Target_Count')
@log_and_time
@experimental
def compute_Trial_Tau(df, reference_order, id='C_Selection_ID', key='M_Condition_Name'):
    def _compare_lists(df, reference_order, key):
        if type(reference_order) == dict:
            subkey = df[key].values[0]
            if not subkey in reference_order.keys():
                subkey = 'others'
            maxTau = 0
            for sequence in reference_order[subkey]:
                tau = kendalltau(df[id].values, sequence)[0]
                if abs(tau) > maxTau:
                    maxTau = tau
            return maxTau
        else:
            return kendalltau(df[id].values, reference_order)[0]
    dfts = df.loc[df['M_Selection_Role'] == 'target']
    compute_trialwise(dfts, lambda x : _compare_lists(x, reference_order, key), 'C_Trial_Tau')
    dfts['C_Trial_AbsTau'] = dfts['C_Trial_Tau'].abs() 
    return dfts

@register_computation('C_Trial_BestLev')
@requires('M_Selection_Role')
@check_protection('C_Trial_Target_Count')
@log_and_time
@experimental
def compute_Trial_BestLev(df, reference_order, id='C_Selection_ID', key='M_Condition_Name', include_reversals=True):
    from Levenshtein import distance as levenshtein_distance
    def _normalized_levenshtein(obs, gt):
        d = levenshtein_distance("".join(map(str, obs)), "".join(map(str, gt)))
        l = max(len(obs), len(gt))
        return 1 - d / l
    def _compare_lists(df, reference_order, key):
        if type(reference_order) == dict:
            subkey = df[key].values[0]
            if not subkey in reference_order.keys():
                subkey = 'others'
            bestLev = 0
            # Add reversals to the list
            if include_reversals:
                reference_order[subkey] = reference_order[subkey] + [lst[::-1] for lst in reference_order[subkey]]
            for sequence in reference_order[subkey]:
                lev = _normalized_levenshtein(df[id].values, sequence)
                if abs(lev) > bestLev:
                    bestLev = lev
            return bestLev
        else:
            #TODO: Also consider reversals here
            return _normalized_levenshtein(df[id].values, reference_order)
    dfts = df.loc[df['M_Selection_Role'] == 'target']
    compute_trialwise(dfts, lambda x : _compare_lists(x, reference_order, key), 'C_Trial_BestLev')
    return dfts

@register_computation('C_Trial_BestLev')
@requires('M_Selection_Role')
@check_protection('C_Trial_Target_Count')
@log_and_time
@experimental
def compute_Trial_STS(df, reference_order, id='C_Selection_ID', key='M_Condition_Name'):
    def _compare_lists(df, reference_order, key):
        if type(reference_order) == dict:
            subkey = df[key].values[0]
            if not subkey in reference_order.keys():
                subkey = 'others'
            max_sts_value = 0 
            sequence_idx = 0 # Will be set to the idx of the best sequence
            # TODO: Return index also in other scores
            for idx, sequence in enumerate(reference_order[subkey]):
                sts_value = sts(df[id].values, sequence)
                if sts_value > max_sts_value:
                    sequence_idx = idx                
        else:
            sts_value= sts(df[id].values, sequence)
        return sts_value # TODO Also sequence_idx, 

    dfts = df.loc[df['M_Selection_Role'] == 'target']
    compute_trialwise(dfts, lambda x : _compare_lists(x, reference_order, key), 'TMP_Trial_STS')
    return dfts