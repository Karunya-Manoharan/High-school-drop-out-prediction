import pandas as pd
import numpy as np
import os


def model_inter(raw_risks, test_features, result_save_dir):

    res = raw_risks.drop(['Year', 'Label'], 1).join(test_features)
    res = res.astype(float)
    res = res.sort_values(by=['Risk'], ascending=False)
    top_10 = int(round(0.1 * len(res), 0))
    res_10 = res.head(top_10).drop('Risk', 1)
    res_90 = res[top_10 + 1:].drop('Risk', 1)

    mean_10 = res_10.mean(axis=0)
    mean_90 = res_90.mean(axis=0)
    k_diff = pd.DataFrame(mean_10).merge(mean_90.rename('mean_90'),
                                         left_index=True,
                                         right_index=True)
    k_diff.columns = ['mean_10', 'mean_90']
    k_diff = k_diff.replace({'mean_10': {0: 0.0000001}})
    k_diff = k_diff.replace({'mean_90': {0: 0.0000001}})
    k_diff['percentage1'] = (abs(k_diff['mean_10'] - k_diff['mean_90'])
                             ) * 100 / abs(k_diff['mean_10'])
    k_diff['percentage2'] = (abs(k_diff['mean_10'] - k_diff['mean_90'])
                             ) * 100 / abs(k_diff['mean_90'])
    k_diff['percentage'] = k_diff[['percentage1', 'percentage2']].min(axis=1)
    k_diff = k_diff.sort_values(by=['percentage'], ascending=False)
    k_diff = k_diff.drop(['percentage1', 'percentage2', 'percentage'], 1)
    #print(k_diff)
    k_diff.to_hdf(os.path.join(result_save_dir, 'k_diff.hd5'), 'result')

def feature_correlation(raw_risks, test_features, result_save_dir, year):
    res = raw_risks.drop(['Year','Label'],1).join(test_features)
    res = res.astype(float)
    corr = res.corr()['Risk']
    corr.to_hdf(os.path.join(result_save_dir, f'Corr_{year}.hd5'), 'result')