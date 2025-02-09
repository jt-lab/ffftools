import numpy as np

def levenshtein_distance(observed_sequence, groundtruth_sequence):
    """
    Compute the Levenshtein distance between two Pandas Series based on their values.

    The Levenshtein distance measures how many single-character edits (insertions, deletions, or substitutions) 
    are required to convert one sequence into another. This is commonly used for comparing strings but can be applied 
    to any sequence of values.

    Args:
        observed_sequence: The sequence of values that the participant followed during the task.
        groundtruth_sequence: The correct sequence of values that the participant should follow.

    Returns:
        int: The Levenshtein distance between the two sequences, representing the number of edits required.
    
    Notes:
        The function compares each item in the two sequences and computes the minimal number of edits needed 
        to make the participant's sequence match the groundtruth sequence.
    """
    len1 = len(observed_sequence)
    len2 = len(groundtruth_sequence)
    
    # Initialize the DP table
    dp = [[0 for j in range(len2 + 1)] for i in range(len1 + 1)]
    
    # Base case: the cost of converting an empty sequence to another
    for i in range(len1 + 1):
        dp[i][0] = i  # Cost of deleting all items from observed_sequence
    for j in range(len2 + 1):
        dp[0][j] = j  # Cost of inserting all items to observed_sequence to match groundtruth_sequence

    # Fill the DP table
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if observed_sequence[i - 1] == groundtruth_sequence[j - 1]:
                cost = 0  # No cost if the items are the same
            else:
                cost = 1  # Substitution cost if the items are different
                
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # Deletion from observed_sequence
                dp[i][j - 1] + 1,       # Insertion into observed_sequence
                dp[i - 1][j - 1] + cost # Substitution
            )

    # The final answer is in the bottom-right corner of the DP table
    return dp[len1][len2]




def sts_alt(observed_sequence, groundtruth_sequence):
    """
    Compute the Standard Sequence Tracing Score (STS) by comparing a participant's sequence to a groundtruth sequence.

    The STS measures how well a participant follows the expected order of items in the groundtruth sequence. 
    A score of 1 indicates perfect adherence (i.e., the participant follows the exact order), while a score of 0 
    indicates complete randomness (i.e., no adherence to the expected order).

    Args:
        observed_sequence (list of int): The sequence of items the participant followed during the task.
        groundtruth_sequence (list of int): The correct order of items, or the groundtruth sequence that the participant should follow.

    Returns:
        float: The STS score, normalized between 0 (complete randomness) and 1 (perfect adherence).

    Notes:
        This score is calculated by determining the number of transitions in the participant sequence 
        that deviate from the expected adjacent items in the groundtruth sequence.
    """
    observed_indices = np.searchsorted(groundtruth_sequence, observed_sequence)
    diffs = np.abs(np.diff(observed_indices))
    systematic_count = np.sum(diffs == 1)
    
    return (systematic_count / len(diffs))


def sts(observed_sequence, groundtruth_sequence):
    # To make sure arbitrary IDs can be used
    print(observed_sequence)
    print(groundtruth_sequence)
    id_mapping = {id_: idx for idx, id_ in enumerate(groundtruth_sequence)}
    obs_seq = np.array([id_mapping[id_] for id_ in observed_sequence])

    diffs = np.abs(np.diff(obs_seq))
    return (diffs ==1).sum() / len(diffs)

