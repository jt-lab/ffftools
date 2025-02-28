import unittest
import pandas as pd
import numpy as np
from ffftools import compute_Selection_ITL

class TestComputeSelectionITL(unittest.TestCase):
    def setUp(self):
        """Set up a test DataFrame with structured selection data."""
        self.df = pd.DataFrame({
            'M_Participant_ID': [0] * 5,
            'M_Condition_Name': ["test"] * 5,
            'M_Trial_Index': [0] * 5,
            'M_Selection_Role': ['target', 'target', 'target', 'distractor', 'target'],
            'M_Selection_X': [0, 3, 6, 2, 9],
            'M_Selection_Y': [0, 4, 8, 5, 12],
            'C_Selection_Target_Count': [1, 2, 3, pd.NA, 4]  # NA for non-target
        })

    def test_ITL_computation(self):
        """Test that the inter-target length is correctly calculated."""
        df_result = compute_Selection_ITL(self.df.copy())

        expected_ITL = [pd.NA, 
                        np.sqrt(3**2 + 4**2),  # Distance between (0,0) and (3,4)
                        np.sqrt(3**2 + 4**2),  # Distance between (3,4) and (6,8)
                        pd.NA,  # Distractor row should remain NA
                        np.sqrt(3**2 + 4**2)]  # Distance between (6,8) and (9,12)

        computed_ITL = df_result['C_Selection_Inter-target_Length'].tolist()

        for exp, comp in zip(expected_ITL, computed_ITL):
            if pd.isna(exp):
                self.assertTrue(pd.isna(comp), "Expected NA but got a value")
            else:
                self.assertAlmostEqual(exp, comp, places=6, msg="Incorrect ITL computation")

    def test_ITL_first_selection_is_NA(self):
        """Test that the first target selection in each trial is assigned NA."""
        df_result = compute_Selection_ITL(self.df.copy())
        self.assertTrue(pd.isna(df_result['C_Selection_Inter-target_Length'].iloc[0]),
                        "First selection's ITL should be NA")

    def test_ITL_distractors_remain_NA(self):
        """Test that non-target selections remain NA."""
        df_result = compute_Selection_ITL(self.df.copy())
        self.assertTrue(pd.isna(df_result.loc[self.df['M_Selection_Role'] != 'target',
                                              'C_Selection_Inter-target_Length']).all(),
                        "Non-target selections should have NA ITL values")

if __name__ == '__main__':
    unittest.main()
