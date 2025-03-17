import numpy as np

# TODO: This can probably be replaced by faster code using groupby as already 
# done for several computers. Optimally this file and function can be deleted soon.
def compute_trialwise(df, fun, colname):
    for p in df.M_Participant_ID.unique():
        maskp = df.M_Participant_ID == p
        for c in df[maskp].M_Condition_Name.unique():
            maskc = df.M_Condition_Name == c
            for t in df[(maskp & maskc)].M_Trial_Index.unique():
                maskt = df.M_Trial_Index == t
                value = fun(df[(maskp & maskc & maskt)])
                if not np.isscalar(value) and (len(value) != len(df.loc[(maskp & maskc & maskt)])):
                    # TODO: Needs fixing!
                    # df[colname] = df[colname].astype(object) # TODO: only when needed
                    # Manually broadcast tuples
                    value = [value] * len(df.loc[(maskp & maskc & maskt)])
                df.loc[(maskp & maskc & maskt), colname] = value