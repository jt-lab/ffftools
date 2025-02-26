import pandas as pd
import numpy as np
import unittest
import matplotlib.pyplot as plt
import matplotlib
import os
from datetime import datetime

# Set the backend to 'Agg' to run headless (non-interactive mode)
matplotlib.use('Agg')

class TestComputeTrialBestR(unittest.TestCase):

    def generate_data(self, direction, num_rows=5, num_columns=4):
        """Generate a snake-like pattern in the specified direction, centered around (0,0)."""
        
        # Initialize X and Y with an empty list
        x_values = []
        y_values = []
        
        # Generate the pattern
        if direction == 'left to right':
            for col in range(num_columns):
                if col % 2 == 0:
                    for row in range(num_rows):
                        x_values.append(col)
                        y_values.append(row)
                else:
                    for row in range(num_rows-1, -1, -1):
                        x_values.append(col)
                        y_values.append(row)
        
        elif direction == 'right to left':
            for col in range(num_columns - 1, -1, -1):  
                if col % 2 == 0:
                    for row in range(num_rows - 1, -1, -1): 
                        x_values.append(col)
                        y_values.append(row)
                else:
                    for row in range(num_rows):
                        x_values.append(col)
                        y_values.append(row)
        
        elif direction == 'top to bottom':
            for row in range(num_rows - 1, -1, -1):
                if row % 2 == 0:
                    for col in range(num_columns):
                        x_values.append(col)
                        y_values.append(row)
                else:
                    for col in range(num_columns - 1, -1, -1):
                        x_values.append(col)
                        y_values.append(row)
        
        elif direction == 'bottom to top':
            for row in range(num_rows):
                if row % 2 == 0:
                    for col in range(num_columns - 1, -1, -1):
                        x_values.append(col)
                        y_values.append(row)
                else:
                    for col in range(num_columns):
                        x_values.append(col)
                        y_values.append(row)

        # Center the pattern by shifting the coordinates
        center_x = np.mean(x_values)
        center_y = np.mean(y_values)
        x_values = [x - center_x for x in x_values]  # Shift X coordinates
        y_values = [y - center_y for y in y_values]  # Shift Y coordinates
        
        # Build minimal dataframe
        return pd.DataFrame({
            'M_Participant_ID': [1] * len(x_values),
            'M_Condition_Name': ['test'] * len(x_values),
            'M_Trial_Index': [1] * len(x_values),
            'M_Selection_Role': ['target'] * len(x_values),
            'M_Selection_X': x_values,  # Centered X values
            'M_Selection_Y': y_values   # Centered Y values
        })
    

    def test_compute_Trial_BestR(self):
        from ffftools import compute_Trial_BestR  # Assuming this function is imported from ffftools
        
        directions = ['top to bottom', 'bottom to top', 'left to right', 'right to left']
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Create a 2x2 grid of subplots
        
        for idx, direction in enumerate(directions):
            with self.subTest(i=idx):
                ax = axes[idx // 2, idx % 2]  # Get the current axis for subplot
                df = self.generate_data(direction)  # Generate data for the given direction

                # Compute the Best-R values using the function
                df_result = compute_Trial_BestR(df.copy())
                
                # Use the first Best-R value for the plot (instead of the mean)
                best_r_value = df_result['C_Trial_BestR'].iloc[0]
                collection_direction = df_result['C_Trial_Collection_Direction'].iloc[0]
                trial_xr = df_result['C_Trial_XR'].iloc[0]
                trial_yr = df_result['C_Trial_YR'].iloc[0]

                ax.plot(df['M_Selection_X'], df['M_Selection_Y'], linestyle='-', alpha=0.5, label='Path')
                ax.scatter(df['M_Selection_X'], df['M_Selection_Y'], marker='o', label='Selections')
                ax.scatter(df['M_Selection_X'].iloc[0], df['M_Selection_Y'].iloc[0], color='red', s=100, zorder=5, label='Start')

                ax.text(0.5, 0.9, f'Best-R = {best_r_value:.2f}', fontsize=12, color='black', ha='center', va='top', transform=ax.transAxes, backgroundcolor='white')
                ax.text(0.5, 0.85, f'Trial_XR = {trial_xr:.2f}', fontsize=12, color='black', ha='center', va='top', transform=ax.transAxes, backgroundcolor='white')
                ax.text(0.5, 0.8, f'Trial_YR = {trial_yr:.2f}', fontsize=12, color='black', ha='center', va='top', transform=ax.transAxes, backgroundcolor='white')
                ax.text(0.5, 0.75, f'Direction: {collection_direction}', fontsize=12, color='black', ha='center', va='top', transform=ax.transAxes, backgroundcolor='white')

                ax.set_title(f"Direction: {direction.title()}")
                ax.set_xlabel("M_Selection_X")
                ax.set_ylabel("M_Selection_Y")
                ax.legend(loc='right')

                if direction == 'top to bottom':
                    self.assertLess(trial_yr, 0)
                elif direction == 'bottom to top':
                    self.assertGreater(trial_yr, 0)
                elif direction == 'left to right':
                    self.assertGreater(trial_xr, 0)
                elif direction == 'right to left':
                    self.assertLess(trial_xr, 0)
                
                # All patterns have large best-r scores
                self.assertGreater(best_r_value, 0.5)

                # The direction should be correct
                self.assertEqual(collection_direction, direction)

        
        # Save the plot to a file
        if os.getenv("PLOT", "True") == "True":
            plot_filename = f"BestRTest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.test.png"
            plt.tight_layout()
            plt.savefig(plot_filename)
            plt.close()  # Close the plot to prevent it from being displayed

if __name__ == '__main__':
    unittest.main()
