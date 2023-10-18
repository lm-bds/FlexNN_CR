import pandas as pd
import numpy as np


class DataSynthesizer:
    def __init__(self, noise_percentage=1):
        self.noise_percentage = noise_percentage

    def generate_synthetic_data(self, df, num_synthetic_samples=10):
        """
        Generate synthetic data by adding a small percentage of noise to a DataFrame.

        Parameters:
            df (pd.DataFrame): The original DataFrame.
            num_synthetic_samples (int): The number of synthetic DataFrames to generate.

        Returns:
            pd.DataFrame: A DataFrame that is 10 times the size of the original, including noise.
        """
        # Create an empty list to store synthetic DataFrames
        synthetic_dfs = []

        for _ in range(num_synthetic_samples):
            # Create a copy of the original DataFrame
            noisy_df = df.copy()

            # Iterate through each column
            for column in noisy_df.columns:
                # Check if the column contains numeric data
                if pd.api.types.is_numeric_dtype(noisy_df[column]):
                    # Calculate the noise magnitude based on the column's data range
                    min_val = noisy_df[column].min()
                    max_val = noisy_df[column].max()
                    noise_range = (max_val - min_val) * (self.noise_percentage / 100)

                    # Generate random noise with the same shape as the column
                    noise = np.random.uniform(-noise_range, noise_range, size=len(noisy_df))

                    # Add the noise to the column
                    noisy_df[column] = noisy_df[column] + noise

            # Append the synthetic DataFrame to the list
            synthetic_dfs.append(noisy_df)

        # Concatenate the original DataFrame and the synthetic DataFrames
        synthetic_df = pd.concat([df] + synthetic_dfs, ignore_index=True)

        return synthetic_df