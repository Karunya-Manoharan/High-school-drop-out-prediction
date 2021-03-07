from scipy import stats
#from sklearn_wrapper import ClfWrapper
import numpy as np

from parse import yamlobj


@yamlobj("!PercentileRankOneFeature")
class PercentileRankOneFeature:
    def __init__(self, feature, descend=False):
        self.feature = feature  # which feature to rank on
        self.descend = (
            descend
        )  # should feature be ranked so lower values -> higher scores
        self.feature_importances_ = None

    def _set_feature_importances_(self, x):
        """Assigns feature importances following the rule: 1 for the feature we
        are ranking on, 0 for all other features."""
        feature_importances = [0] * len(x.columns)

        position = x.columns.get_loc(self.feature)

        feature_importances[position] = 1
        self.feature_importances_ = np.array(feature_importances)

    def fit(self, x, y):
        """Set feature importances and return self."""
        #self._set_feature_importances_(x)
        #return self

        # NOTE: Above expects a DataFrame entering, but we send
        # an array into the model training. This is a moot point
        # because there is not training in the ranker model,
        # just sorting.

        return self

    def predict_proba(self, x):
        """Generate the rank percentile scores and return these."""
        # reduce x to the selected feature, raise error if not found
        # x = x[self.feature]

        # NOTE: Above is when we expect a DataFrame. Since we pass
        # in a numpy array and KNOW the feature will be the first field
        # we use an index. Revert to above if needed.
        x = x[:, 0]

        # we need different behavior depending on rank ordering. percentiles
        # should be able to be interpreted as "proportion of entities ranking
        # BELOW this entity's value". scipy will assign lower ranks to lower
        # values of the feature. so if the entities have values [0, 0, 1, 2, 2],
        # the first two entities will have the lowest ranks (and therefore the
        # lowest risk scores) and the last two will have the highest ranks (and
        # highest risk scores). for the descending method, we need to reverse
        # this, and for both sorting directions, we need to convert the ranks to
        # percentiles.

        # when ascending: tied entities should get the *lowest* rank, so for
        # [0, 0, 1, 2, 2] the ranks should be [1, 1, 3, 4, 4]. these can be
        # converted to the number of entities below each value by subtracting 1
        # from each rank, yielding [0, 0, 2, 3, 3]. from here, we can calculate
        # the proportions by dividing by the length of each list.
        method = "min"
        subtract = 1

        # when descending: tied entities should get the *highest* rank, so for
        # [0, 0, 1, 2, 2] the ranks should be [2, 2, 3, 5, 5]. if we reverse
        # these ranks by substracting all items from the maximum rank (5), we
        # end up with the correct ranks for calculating percentiles:
        # [3, 3, 2, 0, 0]. to simplify the code, we first divide by the length
        # of the list then subtract the result from the maxmimum percentile (1).
        # it produces the same result as subtracting from 5 then dividing:
        #   ([5, 5, 5, 5, 5] -  [2, 2, 3, 5, 5]) / 5  = [0.6, 0.6, 0.4, 0, 0]
        # and
        #    [1, 1, 1, 1, 1] - ([2, 2, 3, 5, 5]  / 5) = [0.6, 0.6, 0.4, 0, 0]
        if self.descend:
            method = "max"
            subtract = 0

        # get the ranks and convert to percentiles
        ranks = stats.rankdata(x, method)
        ranks = [(rank - subtract) / x.shape[0] for rank in ranks]
        if self.descend:
            ranks = [1 - rank for rank in ranks]

        # format it like sklearn output and return
        #return np.array([np.zeros(len(x)), ranks]).transpose()

        # format it like sklearn output and return
        # NOTE: flipped from original above based on pipeline transformations
        risks = np.array(ranks)
        converse_risks = 1 - risks.copy()
        return np.stack([converse_risks, risks], axis=-1)

    # Adding these save funcs with a pass to try to test model otherwise.
    def save_model(self, save_folder: str, save_file: str):
        """Saves model in desired folder and file.

        Inputs:
            save_folder: str
                Folder in which to save model
            save_file
                File in which to save model
        """

        pass

    def save_res(self, save_folder: str, save_file: str, metric: str, res):
        """Saves results in desired folder and file.

        Inputs:
            save_folder: str
                Folder in which to save results
            save_file: str
                File in which to save results
            metric: str
                Metric used to generate results
            res: float, integer, np.ndarray, or list
                Results to save
        """
        pass
