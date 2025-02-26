import unittest
import pandas as pd

from ffftools import compute_Selection_Target_Count  

class TestComputeSelectionTargetCount(unittest.TestCase):
    """Unit tests for compute_Selection_Target_Count function."""

    def test_compute_Selection_Target_Count(self):
        """Tests the compute_Selection_Target_Count function with various cases."""
        
        # Create a test DataFrame
        df = pd.DataFrame({
            'M_Trial_Index': [1, 1, 1, 2, 2, 2, 2],
            'M_Selection_Role': ['target', 'distractor', 'target', 'target', 'distractor', 'target', 'target']
        })

        expected_counts = [1, 1, 2, 1, 1, 2, 3]  # Running count for 'target' per trial
        result_df = compute_Selection_Target_Count(df)

        self.assertIn('C_Selection_Target_Count', result_df.columns, "Output DataFrame is missing 'C_Selection_Target_Count' column.")
        self.assertListEqual(result_df['C_Selection_Target_Count'].tolist(), expected_counts, "Cumulative target counts do not match expected values.")

if __name__ == "__main__":
    unittest.main()