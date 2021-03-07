import numpy as np

from util.fairness.fairlearn_wrapper import FairlearnWrapper, true_positive_rate


def calculate_metrics_by_group(risk_feature_df, percentile_threshold,
                               group_feature_name, metric_map, group_idx_map):
    sorted_df = risk_feature_df.sort_values('Risk', ascending=False)
    threshold_idx = int(sorted_df.shape[0] * .1)
    threshold = sorted_df['Risk'].iloc[threshold_idx]

    group_idxs = np.array([
        group_idx_map[value]
        for value in sorted_df[group_feature_name].tolist()
    ])
    metrics_maker = FairlearnWrapper(metric_map,
                                     sorted_df['Label'],
                                     sorted_df['Risk'] > threshold,
                                     sensitive_features=group_idxs)
    result = metrics_maker.get_by_group()
    return result


def calculate_disparities(group_result, group_idx_map, majority_group):
    disparities = group_result.copy()
    majority_idx = group_idx_map[majority_group]
    for idx in range(group_result.shape[0]):
        if idx != majority_idx:
            disparities.loc[
                idx] = disparities.loc[idx] / disparities.loc[majority_idx]
    disparities.drop(majority_idx, axis=0, inplace=True)
    disparities.rename(
        mapper={value: key
                for key, value in group_idx_map.items()},
        inplace=True)
    return disparities


def get_disparity(risk_feature_df, percentile_threshold, group_feature_name,
                  majority_group):
    metric_map = {
        'TPR': lambda y_true, y_pred: true_positive_rate(y_true, y_pred),
        'FDR': lambda y_true, y_pred: (y_true != y_pred)[y_pred == 1].mean(),
        'Precision': lambda y_true, y_pred:
        (y_true == y_pred)[y_pred == 1].mean(),
    }
    group_idx_map = {
        key: idx
        for idx, key in enumerate(
            set(risk_feature_df[group_feature_name].tolist()))
    }
    group_result = calculate_metrics_by_group(risk_feature_df,
                                              percentile_threshold,
                                              group_feature_name, metric_map,
                                              group_idx_map)
    disparity_df = calculate_disparities(group_result, group_idx_map,
                                         majority_group)
    return disparity_df


# def calculate_bias(joined_df, feature_name, majority_name):
#     precision_prior = joined_df['Label'].mean()
#
#     results = []
#     for _, risks, summary in compare:
#         risks = risks[risks['Year'] == last_year]
#         threshold_idx = int(risks.shape[0] * .1)
#         threshold = risks['Risk'].iloc[threshold_idx]
#         precision = risks['Label'].iloc[:threshold_idx].mean()
#
#         gender_data = risks.join(gender_features, 'student_lookup')['gender']
#         predictions = (risks['Risk'] > threshold).to_numpy(dtype=np.int)
#
#         results.append((precision, fdr_disparity, tpr_disparity))
#     precisions, fdr_disps, tpr_disps = zip(*results)
#     path_name_map = {
#         path: name
#         for name, (path, _) in zip(top_names, top_models)
#     }
#     for path, (config, _, _) in zip(paths, compare):
#         if model_name(config) == 'GPA baseline':
#             path_name_map[path] = 'GPA baseline'
#             break
#     for name, xs, ys, prior_rate in [('FDR', precisions, fdr_disps, 1),
#                                      ('TPR', precisions, tpr_disps, 1)]:
#         fig, ax = plot_model_metrics(paths,
#                                      xs,
#                                      ys,
#                                      path_name_map,
#                                      prior_rate=(precision_prior, prior_rate))
#         ax.set_xlabel('Precision @ 10%')
#         ax.set_ylabel(f'{name} Disparity')
#         handles, labels = ax.get_legend_handles_labels()
#         fig.legend(loc='upper center',
#                    bbox_to_anchor=(0, 0.9, 1, 0.1),
#                    ncol=int(np.ceil(len(labels) / 2)))
#         fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.9))
#         fig.savefig(os.path.join(save_dir, f'{name}.png'))
