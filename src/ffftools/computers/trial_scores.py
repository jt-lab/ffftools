import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from ..algorithms.sequence import sts
from ..decorators import *
from tqdm.auto import tqdm
from .helpers import compute_trialwise

@register_computation('C_Trial_Target_Count')
@requires('M_Selection_Role', 'M_Participant_ID', 'M_Condition_Name', 'M_Trial_Index')
@check_protection('C_Trial_Target_Count')
@log_and_time
def compute_Trial_Target_Count(df):
    compute_trialwise(df, lambda d: (d['M_Selection_Role'] == 'target').sum(), 'C_Trial_Target_Count')
    return df

@register_computation('C_Trial_Time_in_Patch')
@requires('M_Participant_ID', 'M_Condition_Name', 'M_Trial_Index', 'M_Selection_Time')
@check_protection('C_Trial_Time_in_Patch')
@log_and_time
def compute_Trial_Time_in_Patch(df):
    """
    Computes the total time spent in a patch for each trial, based on the last selection time.

    Args:
        fff-compatible df (pd.DataFrame).

    Returns:
        pd.DataFrame: Updated DataFrame with 'C_Trial_Time_in_Patch' column.
    """
    
    df['C_Trial_Time_in_Patch'] = pd.NA 

    grouped = df.groupby(['M_Participant_ID', 'M_Condition_Name', 'M_Trial_Index'])

    results = []
    
    for (participant, condition, trial), trial_data in grouped:
        last_time = trial_data['M_Selection_Time'].max()        
        trial_data = trial_data.copy()
        trial_data['C_Trial_Time_in_Patch'] = last_time
        results.append(trial_data)

    return pd.concat(results, ignore_index=True)

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
        fff-compatible df (pd.DataFrame).

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

@register_computation('C_Trial_Path_Length')
@requires('M_Participant_ID', 'M_Condition_Name', 'M_Trial_Index', 'M_Selection_X', 'M_Selection_Y', 'M_Selection_Role')
@check_protection('C_Trial_Path_Length')
@log_and_time
def compute_Trial_Path_Length(df, targets_only=False):
    """
    Computes the total path length (sum of Euclidean distances between consecutive selections)
    for each trial within each participant and condition.

    Args:
        fff-compatible df (pd.DataFrame): Required columns.
        targets_only (bool, optional): If True, only the path through the targets is considered. 
                                       (i.e., distractor selections are removed frmo the path)

    Returns:
        pd.DataFrame: The dataframe with an additional column 'C_Trial_Path_Length' containing 
                      the total path length per trial.
    """

    def _compute_trial_path(trial_df):
        """Helper function to compute path length for a single trial while preserving non-target rows."""
        if targets_only:
            target_df = trial_df[trial_df['M_Selection_Role'] == 'target']
        else:
            target_df = trial_df

        dx = target_df['M_Selection_X'].diff()
        dy = target_df['M_Selection_Y'].diff()
        path_length = np.sqrt(dx**2 + dy**2).sum()

        trial_df['C_Trial_Path_Length'] = path_length
        return trial_df

    df['C_Trial_Path_Length'] = pd.NA  # Initialize column

    # Apply the computation within each (Participant, Condition, Trial), preserving non-targets
    df = df.groupby(['M_Participant_ID', 'M_Condition_Name', 'M_Trial_Index'], group_keys=False).apply(_compute_trial_path)

    return df

@register_computation('C_Trial_BestTau')
@requires('M_Selection_Role')
@check_protection('C_Trial_Target_Count')
@log_and_time
@experimental
def compute_Trial_Tau(df, reference_order, id_column='S_Selection_ID', key='M_Condition_Name'):
    #TODO: Clean up TMP columns; make a decorator so that it can be used in other functions as well ...
    def _compare_lists(df, reference_order, key):
        if type(reference_order) == dict:
            subkey = df[key].values[0]
            if not subkey in reference_order.keys():
                subkey = 'others'
            maxTau = 0
            for sequence in reference_order[subkey]:
                tau = kendalltau(df[id_column].values, sequence)[0]
                if abs(tau) > maxTau:
                    maxTau = tau
            return maxTau
        else:
            return kendalltau(df[id_column].values, reference_order)[0]
    dfts = df.loc[df['M_Selection_Role'] == 'target']
    compute_trialwise(dfts, lambda x : _compare_lists(x, reference_order, key), 'TMP_Trial_Tau')
    dfts['C_Trial_BestTau'] = dfts['TMP_Trial_Tau'].abs() 
    return dfts

import pandas as pd
from ..algorithms.shp_dp import shortest_hamiltonian_path

@register_computation('C_Trial_Optimal_Path_Length')
@requires('M_Participant_ID', 'M_Condition_Name', 'M_Trial_Index', 'M_Selection_X', 'M_Selection_Y')
@check_protection('C_Trial_Optimal_Path_Length')
@log_and_time
def compute_Trial_Optimal_Path_Length(df, method='SHP', use_last_n=5, 
                                      lut=None, user_specified_column='User_Specified', 
                                      selection_id_column='S_Selection_ID'):
    """
    Computes the optimal path per trial using either the shortest Hamiltonian path (SHP) algorithm 
    (or a user-provided Look-Up Table but this is currently disabled).

    Args:
        fff-compatible df (pd.DataFrame).
        method (str): 'SHP' (default) for shortest Hamiltonian path, 'LUT' for lookup-based paths.
        use_last_n (int, optional): Number of last selections to consider (ignored for 'LUT').
        lut (dict, optional): A lookup table mapping user-specified values to reference paths.
        lut_key_column (str): Column name containing the user-specified condition for LUT.
        id_column (str): Column name containing unique selection IDs.

    Returns:
        pd.DataFrame: Updated fff-compatible DataFrame with 'C_Trial_Optimal_Path_Length' column.
    """

    if use_last_n is not None and use_last_n > 5 and method == 'SHP':
        logger.warning("Attention: Optimal path computation might take long and be infeasible for paths longer than 20.")

    df['C_Trial_Optimal_Path_Length'] = pd.NA  
    
    grouped = df.groupby(['M_Participant_ID', 'M_Condition_Name', 'M_Trial_Index'])
    
    results = []
    
    for (participant, condition, trial), trial_data in grouped:
        
        # Compute using LUT method
        if method == 'LUT' and lut is not None:
            raise NotImplementedError("The Look-Up-Table (LUT) method is currently not implemented.")
            # TODO: The last implementation was too messy ... 
        # Compute using SHP method
        elif method == 'SHP':
            if use_last_n is not None:
                trial_data = trial_data.iloc[-use_last_n:]  

            if len(trial_data) > 1:
                start_node = 0  
                optimal_path_length, _ = shortest_hamiltonian_path(trial_data, start_node=start_node)
            else:
                optimal_path_length = 0  # If only one point, path length is 0

        else:
            raise ValueError("Invalid method specified. Choose either 'SHP' or 'LUT'.")

        trial_data = trial_data.copy()
        trial_data['C_Trial_Optimal_Path_Length'] = optimal_path_length
        results.append(trial_data)
    
    return pd.concat(results, ignore_index=True)

@register_computation('C_Trial_BestLev')
@requires('M_Selection_Role')
@check_protection('C_Trial_Target_Count')
@log_and_time
@experimental
def compute_Trial_BestLev(df, reference_order, id_column='S_Selection_ID', key='M_Condition_Name', include_reversals=True):
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
                lev = _normalized_levenshtein(df[id_column].values, sequence)
                if abs(lev) > bestLev:
                    bestLev = lev
            return bestLev
        else:
            #TODO: Also consider reversals here
            return _normalized_levenshtein(df[id_column].values, reference_order)
    dfts = df.loc[df['M_Selection_Role'] == 'target']
    compute_trialwise(dfts, lambda x : _compare_lists(x, reference_order, key), 'C_Trial_BestLev')
    return dfts

@register_computation('C_Trial_STS')
@requires('M_Selection_Role')
@check_protection('C_Trial_Target_Count')
@log_and_time
@experimental
def compute_Trial_STS(df, reference_order, id='S_Selection_ID', key='M_Condition_Name'):
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
                    max_sts_value = sts_value
    
        else:
            max_sts_value = sts(df[id].values, sequence)
        #print(str(subkey) + ": " + str(max_sts_value))
        return max_sts_value # TODO Also sequence_idx, 

    dfts = df.loc[df['M_Selection_Role'] == 'target']
    compute_trialwise(dfts, lambda x : _compare_lists(x, reference_order, key), 'C_Trial_STS')
    return dfts

@register_computation('C_Trial_PAO')
@requires('C_Trial_Path_Length', 'C_Trial_Optimal_Path_Length')
@check_protection('C_Trial_PAO')
@log_and_time
@experimental
def compute_Trial_PAO(df):
    """
    Computes the Percentage Above Optimal (PAO) for each trial.
    
    PAO is calculated as:
        PAO = ((C_Trial_Path_Length / C_Trial_Optimal_Path_Length) - 1) * 100

    Reference: Wiener et al., 2007

    Args:
        fff-compatible df (pd.DataFrame).

    Returns:
        pd.Series: A series containing PAO values for each trial.
    """
    # Avoid division by zero
    valid_mask = df['C_Trial_Optimal_Path_Length'] > 0

    # Compute PAO
    df.loc[valid_mask, 'C_Trial_PAO'] = ((df.loc[valid_mask, 'C_Trial_Path_Length'] / 
                         df.loc[valid_mask, 'C_Trial_Optimal_Path_Length']) - 1) * 100

    return df
