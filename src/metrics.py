from typing import Callable, Any

import numpy as np
import pandas as pd


def compute_summary(test_risk_df: pd.DataFrame):
    """Compute summary metrics of risk_df
    test_risk_df: must contain 'Risk' and 'Label' columns corresponding to predicted risk and true label
    """
    test_risk_df.sort_values('Risk', inplace=True, ascending=False)

    top_10_pct = int(np.ceil(.1 * test_risk_df.shape[0]))
    predictions = test_risk_df.head(top_10_pct)
    precision = predictions['Label'].mean()
    recall = predictions['Label'].sum() / test_risk_df['Label'].sum()
    return {'precision@10': precision, 'recall@10': recall}
