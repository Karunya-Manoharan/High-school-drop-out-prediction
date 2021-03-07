from fairlearn.metrics import *
'''
Example Metrics Dict:

metrics = {
    'selection_rate': selection_rate,
    'false_negative_rate': false_negative_rate,
    'false_positive_rate': false_positive_rate,
    'true_positive_rate': true_positive_rate,
    'true_negative_rate': true_negative_rate,
    'demographic_parity_difference': demographic_parity_difference,
    'demographic_parity_ratio': demographic_parity_ratio,
    'equalized_odds_difference': equalized_odds_difference,
    'equalized_odds_ratio': equalized_odds_ratio
}
'''


class FairlearnWrapper(MetricFrame):
    """Fairlearn wrapper."""
    def __init__(self,
                 metric,
                 y_true,
                 y_pred,
                 sensitive_features,
                 control_features=None,
                 sample_params=None):
        """Initializes FairlearnWrapper.

        Inputs:
            metric: callable or dict
                Metric function(s) to be calculated.
                Functions must be callable as:
                    fn(y_true, y_pred, *sample_params).
                See fairlearn docs if more params are needed.
            y_true: np.ndarray, list, pandas dataframe or series
                Ground truth labels.
            y_pred: np.ndarray, list, pandas dataframe or series
                Predicted labels.
            sensitive_features: list, dict of 1d arrays, np.ndarray,
                        pandas series or dataframe
                Sensitive features used to create the subgroups.
            control_features: list, dict of 1d arrays, np.ndarray,
                        pandas series or dataframe
                Divide input data into subgroups, but doesn't perform
                aggregation over these features.
            sample_params: dict
                Parameters for the metric functions used.
        """
        super(FairlearnWrapper,
              self).__init__(metric,
                             y_true,
                             y_pred,
                             sensitive_features=sensitive_features,
                             control_features=control_features,
                             sample_params=sample_params)

    def get_disaggregated_metric_vals(self):
        """Returns collection of disaggregated metric values.

        Inputs: None

        Returns:
            MetricFrame
        """

        return self

    def get_by_group(self):
        """Returns collection of metrics evaluated on each subgroup.

        Inputs: None

        Returns:
            When dict --> pandas dataframe with columns named after metric
            functions and rows indexed by subgroup combinations.
            When callable --> pandas series indexed by subgroup
            combinations.
        """

        return self.by_group

    def get_control_levels(self):
        """Returns list of feature names produced by control features.

        Inputs: None

        Returns:
            List[str] or None
        """

        return self.control_levels

    def get_overall(self):
        """Returns the underlying metrics evaluated on the whole dataset.

        Inputs: None

        Returns:
            Output type depends on whether control features provided and how
            metric functions specified.
        """

        return self.overall

    def get_difference(self, method: str = 'between_groups'):
        """Returns the maximum absolute difference for each parameter.

        Inputs:
            method: str, either 'between_groups' or 'to_overall'

        Returns:
            Difference between groups.
        """

        return self.difference(self, method)

    def get_group_max(self):
        """Returns the maximum value of the metric over sensitive features.

        Inputs: None

        Returns:
            Maximum value over all combinations of sensitive features for
            each underlying metric function in by_group property.
        """

        return self.group_max()

    def get_group_min(self):
        """Returns the minimum value of the metric over sensitive features.

        Inputs: None

        Returns:
            Minimum value over all combinations of sensitive features for
            each underlying metric function in by_group property.
        """
        return self.group_min()
