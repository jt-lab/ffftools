import unittest
import pandas as pd

# Import the function (adjust import path as needed)
from ffftools import compute_Selection_Nth_Last_Target  

class TestComputeSelectionNthLastTarget(unittest.TestCase):
    """Unit tests for compute_Selection_Nth_Last_Target function."""

    def test_compute_Selection_Nth_Last_Target(self):
        """Tests that the function correctly computes the Nth-last target index."""
        
        # Create a test DataFrame
        df = pd.DataFrame({
            'C_Selection_Target_Count': [1, 2, 3, 1, 2, 3, 4],
            'C_Trial_Target_Count': [3, 3, 3, 4, 4, 4, 4]  # Trial 1 has 3 targets, Trial 2 has 4
        })

        expected_nth_last = [3, 2, 1, 4, 3, 2, 1]

        result_df = compute_Selection_Nth_Last_Target(df)
        self.assertIn('C_Selection_Nth_Last_Target', result_df.columns, 
                      "Output DataFrame is missing 'C_Selection_Nth_Last_Target' column.")

        self.assertListEqual(result_df['C_Selection_Nth_Last_Target'].tolist(), expected_nth_last, 
                             "Nth-last target counts do not match expected values.")

if __name__ == "__main__":
    unittest.main()
