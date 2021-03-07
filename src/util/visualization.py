import graphviz
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as metrics
import sklearn.tree as tree
import pandas as pd

import model
from model.sklearn_wrapper import ClfWrapper

matplotlib.rcParams['mathtext.fontset'] = 'cm'


def plot_precision_at_10(summary, result, result_save_dir):
    """Makes a prediction@10 plot."""
    summary['Year'] = summary['Year'].astype(str)
    ax = sns.lineplot(x='Year', y='precision@10', data=summary)
    ax.set(xlabel='Training Cut-off', ylabel='Model Performance')
    ax.set(ylim=(0, 1))
    plt.title('Model Selection')
    plt.savefig(f'{result_save_dir}/precision_at_10.png')
    plt.clf()


def model_name(config):
    cur_model = config[-1]
    if isinstance(cur_model, model.PercentileRankOneFeature):
        return 'GPA baseline'
    else:
        return cur_model._yaml_fields['clf_name']


def get_distinct_handles_labels(handles, labels):
    handles, labels = zip(
        *[(handle, label)
          for label, handle in dict(zip(labels, handles)).items()])
    return handles, labels


bold_keys = ['GPA baseline']
bold_kwargs = {'linewidth': 4, 'linestyle': 'dashed'}


def plot_all_precision_at_10(fig, ax, info, cmap_name="tab10"):
    labels = list(sorted({model_name(inf[0]) for inf in info}))
    cmap = matplotlib.cm.get_cmap(cmap_name)
    color_map = {label: cmap(i) for i, label in enumerate(labels)}

    years, base_rate = (None, None)
    for i, inf in enumerate(info):
        config, risks, summary = inf
        # summary['Year'] = summary['Year'].astype(str)
        extra_kwargs = {} if model_name(
            config) not in bold_keys else bold_kwargs
        ax.plot(summary['Year'],
                summary['precision@10'],
                color=color_map[model_name(config)],
                label=model_name(config),
                **extra_kwargs)

        if i == 0:
            years, base_rate = (summary['Year'], summary['Prior Rate'])

    # Hardcode prior rates temporarily
    ax.plot(years, base_rate, color='black', label='Prior Rate', **bold_kwargs)
    ax.set_xlim(years.min(), years.max())
    ax.set_xticks(np.arange(years.min(), years.max() + 1))
    ax.set_xlabel('Training Cut-off')
    ax.set_ylabel('Precision @ 10%')
    # ax.set_title('Performance of all models across all splits')
    handles, labels = get_distinct_handles_labels(*(
        ax.get_legend_handles_labels()))
    fig.legend(handles,
               labels,
               ncol=2,
               loc='upper center',
               bbox_to_anchor=(0, 0.7, 1, 0.3))
    fig.tight_layout(rect=(0, 0, 1, 0.7))


def plot_top_by_metric(fig,
                       ax,
                       paths,
                       info,
                       metric_fn,
                       baselines=None,
                       top_ct=5,
                       cmap_name="Set2"):
    cmap = matplotlib.cm.get_cmap(cmap_name)
    top_models = sorted(zip(paths, info),
                        key=lambda x: metric_fn(x[1]))[(-1 * top_ct)::][::-1]

    min_year, max_year = (None, None)

    years, base_rate = (None, None)
    for idx, (path, info) in enumerate(top_models):
        rank = idx + 1
        config, risks, summary = info
        ax.plot(summary['Year'],
                summary['precision@10'],
                color=cmap(idx),
                label=f'{rank}_{model_name(config)}')
        if idx == 0:
            min_year, max_year = (summary['Year'].min(), summary['Year'].max())
            years = summary['Year']
            base_rate = summary['Prior Rate']
    if baselines is not None:
        for path, info in baselines:
            config, risks, summary = info
            extra_kwargs = {} if model_name(
                config) not in bold_keys else bold_kwargs

            ax.plot(summary['Year'],
                    summary['precision@10'],
                    color='orange',
                    label=model_name(config),
                    **extra_kwargs)
            ax.plot(years,
                    base_rate,
                    color='black',
                    label='Prior Rate',
                    **bold_kwargs)

    ax.set_xlabel('Training Cut-off')
    ax.set_ylabel('Precision @ 10%')
    # ax.set_title(f'Performance of top {top_ct} models across all splits')
    ax.set_xlim(min_year, max_year)
    ax.set_xticks(np.arange(min_year, max_year + 1))
    fig.legend(ncol=2, loc='upper center', bbox_to_anchor=(0, 0.7, 1, 0.3))
    fig.tight_layout(rect=(0, 0, 1, 0.7))
    return top_models


def plot_roc(summary, result, result_save_dir):
    """Makes a ROC plot."""
    summary['Year'] = summary['Year'].astype(int)
    for year in set(summary['Year'].to_numpy()):
        year_result = result[result['Year'] == year]
        precision, recall, threshold = [
            x[::-1] for x in metrics.precision_recall_curve(
                year_result['Label'], year_result['Risk'])
        ]

        total_ct = year_result.shape[0]
        pop_percents = [0] + [(year_result['Risk'] >= thresh).sum() / total_ct
                              for thresh in threshold] + [1]
        precision = np.append(precision, precision[-1])
        recall = np.append(recall, recall[-1])

        fig = plt.figure(figsize=(5, 3))
        ax = plt.gca()

        ax.plot(pop_percents, precision, color='b', label='precision')
        ax.yaxis.label.set_color('blue')
        ax.tick_params(axis='y', colors='blue')

        recall_ax = ax.twinx()
        recall_ax.yaxis.label.set_color('orange')
        recall_ax.tick_params(axis='y', colors='orange')
        recall_ax.plot(pop_percents, recall, color='orange', label='recall')

        fig.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        ax.set_ylabel('Precision')
        recall_ax.set_ylabel('Recall')
        ax.set_xlabel('Percent of population')

        fig.tight_layout(pad=2.0)
        fig.savefig(f'{result_save_dir}/roc_{year}.png')


def plot_tree(trained_tree, result_save_dir):
    """Makes a plot of a DT."""
    dot_data = tree.export_graphviz(clf, out_file=result_save_dir + '/tree')
    graph = graphviz.Source(dot_data)
    graph.render('decision_tree')


def plot_feature_importance(ax, clf, feature_names, top_features=20):
    if not isinstance(clf, ClfWrapper):
        print(f"Not ClfWrapper, clf is {clf.__class__.__name__}")
    elif not hasattr(clf.estimator, 'feature_importances_'):
        #TODO implement feature importances for models don't have it built in.
        print("{} does not have feature_importances_ attribute".format(
            clf.__class__.__name__))
    else:
        feat_imp = pd.DataFrame(
            {'importance': clf.estimator.feature_importances_})
        feat_imp['feature'] = feature_names
        feat_imp.sort_values(by='importance', ascending=False, inplace=True)
        feat_imp = feat_imp.iloc[:top_features]
        feat_imp.sort_values(by='importance', inplace=True)
        feat_imp = feat_imp.set_index('feature', drop=True)
        feat_imp.plot.barh(ax=ax)  # Wider figure return feat_imp


def plot_model_metrics(ax, paths, xs, ys, path_name_map, prior_rate=None):
    other_xs, other_ys = zip(*[(x, y) for path, x, y in zip(paths, xs, ys)
                               if path not in path_name_map])
    datas = [(other_xs, other_ys, 'Other models')] + [
        ([x], [y], path_name_map[path])
        for x, y, path in zip(xs, ys, paths) if path in path_name_map
    ]
    for xs, ys, name in datas:
        if name == 'Other models':
            ax.scatter(xs, ys, label=name, color='none', edgecolor='black')
        else:
            print(xs, ys, name)
            ax.scatter(xs, ys, label=name)
    if prior_rate is not None:
        x_prior, y_prior = prior_rate
        ax.scatter([x_prior], [y_prior], label='Prior Rate')
