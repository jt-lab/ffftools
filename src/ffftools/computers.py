import numpy as np
from scipy.stats import kendalltau
from tqdm.auto import tqdm
from .algorithms.sequence import sts
from functools import wraps
import logging
import time


import logging
import time

# Try importing colorlog, and if it fails, fall back to standard logging
try:
    import colorlog
    colorlog_available = True
except ImportError:
    colorlog_available = False

# Set up a standard log formatter
log_formatter = logging.Formatter("%(asctime)s [ %(levelname)s ] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# Default to standard logging if colorlog is unavailable
if colorlog_available:
    # Set up a colored log formatter
    log_formatter = colorlog.ColoredFormatter(
        "%(asctime)s [%(log_color)s%(levelname)s%(reset)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG': 'magenta',
            'INFO': 'blue',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red'
        }
    )

# Set up the console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# Get the root logger and set the level to INFO
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)

def log_and_time(func):
    """Logs and times the execution of a function."""
    @wraps(func)
    def wrapper(df, *args, **kwargs):
        logging.info(f"Starting {func.__name__}...")
        start_time = time.time()
        result = func(df, *args, **kwargs)
        elapsed_time = time.time() - start_time
        logging.info(f"Finished {func.__name__} after {elapsed_time:.4f} seconds.")
        return result
    return wrapper

def check_protection(colname):
    def decorator(func):
        @wraps(func)
        def wrapper(df, *args, **kwargs):
            if '!' + colname in df.columns:
                logging.info(f"Attempted to compute '{colname}', but a protected custom column was found. Skipping computation!")
                return df  # Skip computation and return the original df

            return func(df, *args, **kwargs)  # Call the function normally if not protected
        return wrapper
    return decorator

# Global registry to track functions that compute specific columns
COLUMN_FUNCTION_MAP = {}

def register_computation(*output_columns):
    """Registers a function that computes one or more columns passed as separate arguments."""
    def decorator(func):
        for col in output_columns:
            COLUMN_FUNCTION_MAP[col] = func  # Register each column
        @wraps(func)
        def wrapper(df, *args, **kwargs):
            return func(df, *args, **kwargs)
        return wrapper
    return decorator


def requires(*required_columns):
    """Decorator to check and compute required columns if missing."""
    def decorator(func):
        @wraps(func)
        def wrapper(df, *args, **kwargs):
            for col in required_columns:
                if col not in df.columns:
                    logging.info(f"Column '{col}' is missing. Computing it first...")
                    if col in COLUMN_FUNCTION_MAP:
                        df = COLUMN_FUNCTION_MAP[col](df)  # Compute required column
                    else:
                        raise ValueError(f"No registered function to compute '{col}'")
            return func(df, *args, **kwargs)
        return wrapper
    return decorator
    

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
@log_and_time
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
    df['C_Selection_Inter-target_Time'] = -1.0
    itts = dfts['M_Selection_Time'].values[1:] - dfts['M_Selection_Time'].values[0:-1]
    df.loc[df['M_Selection_Role'] == 'target', 'C_Selection_Inter-target_Time'] = np.array([-1] + list(itts), dtype=float)
    df.loc[df['C_Selection_Target_Count'] == 1, 'C_Selection_Inter-target_Time'] = -1.0
    return(df)

def compute_Selection_IT_LT_Ratio(df):
    #if 'C_Selection_Inter-target_Length' not in df:
    #    compute_Selection_ITL(df) #TODO: resolve auto computation via decorator!
    df['C_Selection_Inter-target_LT_Ratio'] = df['C_Selection_Inter-target_Length'] / df['M_Selection_Inter-target_Time']
    df.loc[df['M_Selection_Inter-target_Time'] == -1, 'C_Selection_Inter-target_Length'] = -1
    return(df)

@register_computation('C_Trial_Target_Count')
@log_and_time
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
    df['C_Trial_Collection_Direction'] = 'None' # Will be overwritten virtually always!
    df.loc[(df['C_Trial_XR'].abs() > df['C_Trial_YR'].abs()) & (df['C_Trial_XR'] < 0),\
        'C_Trial_Collection_Direction'] = 'left to right'
    df.loc[(df['C_Trial_XR'].abs() > df['C_Trial_YR'].abs()) & (df['C_Trial_XR'] > 0),\
        'C_Trial_Collection_Direction'] = 'right to left'
    df.loc[(df['C_Trial_XR'].abs() < df['C_Trial_YR'].abs()) & (df['C_Trial_XR'] < 0),\
        'C_Trial_Collection_Direction'] = 'top to bottom'
    df.loc[(df['C_Trial_XR'].abs() > df['C_Trial_YR'].abs()) & (df['C_Trial_XR'] > 0),\
        'C_Trial_Collection_Direction'] = 'bottom to top'
    return(df)

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
    
def compute_Trial_BestLev(df, reference_order, id='C_Selection_ID', key='M_Condition_Name', include_reversals=True):
    print("Best lev started!")
    from Levenshtein import distance as levenshtein_distance
    def _normalized_levenshtein(obs, gt):
        d = levenshtein_distance("".join(map(str, obs)), "".join(map(str, gt)))
        l = max(len(obs), len(gt))
        return 1 - d / l
    def _compare_lists(df, reference_order, key):
        print("working on one trial!")
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
    compute_trialwise(dfts, lambda x : _compare_lists(x, reference_order, key), 'C_Trial_Lev')
    print("Best lev ended")
    return dfts

def compute_Trial_STS(df, reference_order, id='C_Selection_ID', key='M_Condition_Name'):
    def _compare_lists(df, reference_order, key):
        if type(reference_order) == dict:
            subkey = df[key].values[0]
            if not subkey in reference_order.keys():
                subkey = 'others'
            max_sts_value = 0 
            sequence_idx = 0 # Will be set to the idx of the best sequence
            print(subkey)
            # TODO: Return index also in other scores
            for idx, sequence in enumerate(reference_order[subkey]):
                print(sequence)
                sts_value = sts(df[id].values, sequence)
                if sts_value > max_sts_value:
                    sequence_idx = idx                
        else:
            sts_value= sts(df[id].values, sequence)
        return sts_value # TODO ALso sequence_idx, 

    dfts = df.loc[df['M_Selection_Role'] == 'target']
    compute_trialwise(dfts, lambda x : _compare_lists(x, reference_order, key), 'TMP_Trial_STS')
    #dfts['c'], dfts['d'] = zip(*dfts['TMP_Trial_STS'])
    #df[['col1', 'col2']] = pd.DataFrame(df['data'].tolist(), index=df.index)
    return dfts