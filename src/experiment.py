import os
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configloader import load_config, create_splits
from load_data import get_graduation_info, connect_cursor
import metrics
from model import train_model
from util.visualization import plot_precision_at_10, plot_roc, plot_feature_importance
from util.interpretability import model_inter, feature_correlation


class Experiment:
    def __init__(self, config_path, secret_path, model_save_dir,
                 result_save_dir):
        self.config_path = config_path
        self.secret_path = secret_path
        self.model_save_dir = model_save_dir
        self.result_save_dir = result_save_dir
        self.loaders, self.labeler, self.splitter, self.preprocessor, self.imputer, self.model = load_config(
            self.config_path)

    def run(self):
        """Run entire pipeline."""
        cur = connect_cursor(self.secret_path)
        splits = create_splits(self.loaders, self.labeler, self.splitter, cur)

        # Get graduation info

        preprocessed_splits = [(year, *(self.preprocessor.transform(
            (train_pair, test_pair))))
                               for year, train_pair, test_pair in splits]

        def impute(split):
            year, (train_df, train_label), (test_df, test_label) = split
            assert train_df.index.equals(train_label.index)
            assert train_df.index.is_unique
            train_df, test_df = self.imputer.transform(train_df, test_df)

            assert train_df.index.equals(
                train_df.index.intersection(train_label.index))
            train_label, test_label = train_label.sort_index().loc[
                train_df.index.intersection(train_label.index), :].copy(
                ), test_label.loc[test_df.index, :].copy()
            assert train_df.index.is_unique
            assert train_label.index.is_unique
            assert train_df.index.equals(train_label.index)
            return year, (train_df, train_label), (test_df, test_label)

        preprocessed_splits = [impute(split) for split in preprocessed_splits]
        for path in [self.model_save_dir, self.result_save_dir]:
            if not os.path.exists(path):
                os.makedirs(path)
        metrics_list = []
        results_list = []
        for year, (train_features,
                   train_labels), (test_features,
                                   test_labels) in preprocessed_splits:
            if not train_features.empty:
                self.model = train_model(
                    self.model,
                    train_features.to_numpy(dtype=np.float),
                    train_labels.to_numpy(dtype=np.int).reshape(-1),
                    model_save_folder=self.model_save_dir,
                    model_save_name=f"self.model_{year}")

                fig = plt.figure(figsize=(20, 12))
                ax = plt.gca()
                feat_imp = plot_feature_importance(ax, self.model,
                                                   train_features.columns, 20)
                ax.set_xlabel('Feature Importance Score')
                fig.savefig(
                    f'{self.result_save_dir}/feature_importance_{year}.png')

                if feat_imp is not None:
                    feat_imp.to_hdf(
                        os.path.join(self.result_save_dir,
                                     f'feature_importance_{year}.hd5'),
                        'result')

                raw_risks = self.model.predict_proba(
                    test_features.to_numpy(dtype=np.float))[:, 1]
                test_risk_df = pd.DataFrame(index=test_features.index,
                                            data=raw_risks,
                                            columns=["Risk"]).join(test_labels)
                test_risk_df['Year'] = year
                feature_correlation(test_risk_df, test_features,
                                    self.result_save_dir, year)
                if year == 2013:
                    model_inter(test_risk_df, test_features,
                                self.result_save_dir)
                summary_metrics = metrics.compute_summary(test_risk_df)
                prior_rate = test_labels['Label'].mean()
                metrics_list.append({
                    'Year': year,
                    'Prior Rate': prior_rate,
                    **summary_metrics
                })

                results_list.append(test_risk_df)
        """Dataframe indexed by (student_lookup) and has columns ['Year', 'Risk', 'Label']"""
        results = pd.concat(results_list)  # ['Year','Risk', 'Label']
        """Dataframe indexed by "year" and has columns corresponding to test/train metrics on each year data split"""
        summary = pd.DataFrame.from_records(metrics_list)
        print(summary)

        # Visaulizations and save results
        candidate_config_path = os.path.join(self.model_save_dir,
                                             'config.yaml')
        if os.path.realpath(
                self.config_path) != os.path.realpath(candidate_config_path):
            copyfile(self.config_path, candidate_config_path)
        plot_precision_at_10(summary, results, self.result_save_dir
                             )  # TODO take directory to save visualizations in
        plot_roc(summary, results, self.result_save_dir)
        results.to_hdf(os.path.join(self.result_save_dir, 'risks.hd5'),
                       'result')
        summary.to_hdf(os.path.join(self.result_save_dir, 'summary.hd5'),
                       'result')
