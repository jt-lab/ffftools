import unittest
import pandas as pd
from ffftools import compute_Selection_Target_Switch

class TestComputeSelectionTargetSwitch(unittest.TestCase):
    def setUp(self):
        """Set up a sample DataFrame for testing."""
        self.df = pd.DataFrame({
            'M_Participant_ID': [0] * 5,
            'M_Condition_Name': ["test"] * 5,
            'M_Trial_Index': [0] * 5,
            'M_Selection_Role': ['target', 'target', 'target', 'target', 'target'],
            'M_Selection_Type': ['A', 'A', 'B', 'B', 'A'],
            'M_Trial_Index' : 0
        })

    def test_switch_detection(self):
        """Test if the function correctly detects selection switches."""
        df_result = compute_Selection_Target_Switch(self.df.copy())

        expected = [pd.NA, 0, 1, 0, 1]
        computed = df_result['C_Selection_Target_Switch'].tolist()
        
        self.assertEqual(computed, expected, "Target switch values do not match expected results.")

    def test_no_switch(self):
        """Test a case where there are no switches."""
        df_no_switch = pd.DataFrame({
            'M_Participant_ID': [0] * 5,
            'M_Condition_Name': ["test"] * 5,
            'M_Trial_Index': [0] * 5,
            'M_Selection_Role': ['target'] * 5,
            'M_Selection_Type': ['A', 'A', 'A', 'A', 'A'],
            'M_Trial_Index' : 0,
        })

        df_result = compute_Selection_Target_Switch(df_no_switch.copy())

        expected = [pd.NA, 0, 0, 0, 0]
        computed = df_result['C_Selection_Target_Switch'].tolist()

        self.assertEqual(computed, expected, "Function incorrectly detected switches where there are none.")

    def test_first_selection_na(self):
        """Ensure the first selection always gets NA."""
        df_result = compute_Selection_Target_Switch(self.df.copy())

        self.assertTrue(pd.isna(df_result['C_Selection_Target_Switch'].iloc[0]), "First selection should be NA.")

    def test_non_target_rows_unchanged(self):
        """Ensure non-target rows are not modified."""
        df_with_nontargets = self.df.copy()
        df_with_nontargets.loc[2, 'M_Selection_Role'] = 'distractor'  # Change one row to non-target

        df_result = compute_Selection_Target_Switch(df_with_nontargets)

        self.assertTrue(pd.isna(df_result.loc[2, 'C_Selection_Target_Switch']),
                        "Non-target rows should remain unchanged.")

if __name__ == '__main__':
    unittest.main()
