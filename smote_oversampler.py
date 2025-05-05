# smote_oversampler.py

from imblearn.over_sampling import SMOTE
import pandas as pd

class SMOTEOversampler:
    """
    A utility class for applying SMOTE oversampling to handle imbalanced datasets.
    """

    def __init__(self, sampling_strategy=0.5, random_state=42):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.smote = SMOTE(sampling_strategy=self.sampling_strategy, random_state=self.random_state)

    def apply(self, X, y):
        """
        Applies SMOTE to the given features and labels.

        Parameters:
        - X: Features (DataFrame or ndarray)
        - y: Labels (Series or ndarray)

        Returns:
        - resampled_df: DataFrame containing resampled features and target
        """
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
        resampled_df['Class'] = y_resampled
        return resampled_df
