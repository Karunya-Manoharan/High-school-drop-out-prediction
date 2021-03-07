import os
from shutil import copyfile, copytree, rmtree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import click
import yaml

from configloader import load_config, create_splits
from load_data import connect_cursor, get_snapshot_features
from util.fairness.bias import get_disparity
from util.visualization import plot_all_precision_at_10, plot_top_by_metric, model_name, plot_model_metrics


def load_exp_dir(path):
    config_path = os.path.join(path, 'config.yaml')
    config = load_config(config_path)
    risks_path = os.path.join(path, 'risks.hd5')
    summary_path = os.path.join(path, 'summary.hd5')
    risks_df = pd.read_hdf(risks_path, 'result')
    summary_df = pd.read_hdf(summary_path, 'result')

    return config, risks_df, summary_df


def mean_precision_at_10(info):
    _, _, summary = info
    return summary['precision@10'].mean()


@click.command()
@click.option('--paths', required=True, multiple=True)
@click.option('--save_dir', required=True)
@click.option('--secret_path', default="~/.secrets.yaml")
def parse_results(paths: list, save_dir: str, secret_path: str):
    compare = [load_exp_dir(p) for p in paths]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    gpa_baseline = [x for x in compare
                    if model_name(x[0]) == 'GPA baseline'][0]

    fig = plt.figure(figsize=(5, 3))
    ax = plt.gca()
    plot_all_precision_at_10(fig, ax, compare)
    ax.set_ylim(0, 0.4)
    fig.savefig(os.path.join(save_dir, "precision_at_10_all.png"))
    plt.close(fig)

    fig = plt.figure(figsize=(5, 3))
    ax = plt.gca()
    top_models = plot_top_by_metric(fig, ax, paths, compare,
                                    mean_precision_at_10,
                                    [('gpa_baseline', gpa_baseline)], 5)
    ax.set_ylim(0, 0.4)
    fig.savefig(os.path.join(save_dir, "precision_at_10_top_5.png"))
    plt.close(fig)

    top_names = []
    for i, (path, (config, risks, summary)) in enumerate(top_models):
        name = f'{i + 1}_{model_name(config)}'
        new_dir = os.path.join(save_dir, name)
        if os.path.exists(new_dir):
            rmtree(new_dir)
        copytree(path, new_dir)

        importance_path = os.path.join(new_dir, 'feature_importance_2013.hd5')
        if os.path.exists(importance_path):
            imp_df = pd.read_hdf(importance_path)
            fig = plt.figure(figsize=(10, 6))
            ax = plt.gca()
            imp_df.iloc[-10:, :].plot.barh(ax=ax)
            ax.set_xlabel('Feature Importance')
            ax.set_ylabel('')
            ax.get_legend().remove()
            fig.tight_layout(pad=0.5)
            fig.savefig(f'{new_dir}/feature_importance_main.png')
        with open(os.path.join(new_dir, 'rank_stats.yaml'), 'w') as in_f:
            in_f.write(
                yaml.dump({
                    'name':
                    name,
                    'rank':
                    i + 1,
                    'path':
                    path,
                    'mean_precision_at_10':
                    float(mean_precision_at_10((config, risks, summary)))
                }))

        top_names.append(name)

    cur = connect_cursor(secret_path)
    path, ((loaders, labeler, splitter, _, _, _), risks,
           summary) = (paths[-2], compare[-2])

    group_features = ['gender', 'disadvantagement']
    group_majorities = ['M', 'none']
    disparity_dir = os.path.join(save_dir, 'disparity')
    if not os.path.exists(disparity_dir):
        os.makedirs(disparity_dir)
    # Create path name map
    path_name_map = {
        path: name
        for name, (path, _) in zip(top_names, top_models)
    }
    for path, (config, _, _) in zip(paths, compare):
        if model_name(config) == 'GPA baseline':
            path_name_map[path] = 'GPA baseline'
            break

    splits = create_splits(loaders, labeler, splitter, cur)
    years = [split[0] for split in splits]
    gender_features = get_snapshot_features(['gender', 'disadvantagement'],
                                            cur)
    year_df_map = {}
    for split in splits:
        df = split[2][0].join(gender_features, 'student_lookup')
        df.loc[df['disadvantagement'].isnull(), 'disadvantagement'] = 'none'
        year_df_map[split[0]] = df

    for group_feature_name, majority_group in zip(group_features,
                                                  group_majorities):
        for year_idx, year in enumerate([2013]):
            year_features = year_df_map[year]
            year_groups = set(year_features[group_feature_name].tolist())
            year_groups.remove(majority_group)
            # year_groups.remove(np.nan)
            year_results = []
            year_precision_prior = None
            for _, risks, summary in compare:
                model_risks = risks[risks['Year'] == year].copy().sort_values(
                    'Risk', ascending=False)

                joined_df = model_risks.drop('Year', axis=1).join(
                    year_features, 'student_lookup')
                model_disparity = get_disparity(joined_df, .1,
                                                group_feature_name,
                                                majority_group)
                threshold_idx = int(model_risks.shape[0] * .1)
                precision = model_risks['Label'].iloc[:threshold_idx].mean()
                year_results.append((model_disparity, precision))
                if year_precision_prior is None:
                    year_precision_prior = model_risks['Label'].mean()
            for group_name in year_groups:
                precisions, fdrs, tprs = zip(
                    *[(precision, disparity.loc[group_name, 'FDR'],
                       disparity.loc[group_name, 'TPR'])
                      for disparity, precision in year_results])
                for metric_name, xs, ys in [('FDR', precisions, fdrs),
                                            ('TPR', precisions, tprs)]:
                    fig = plt.figure(figsize=(10, 3))
                    ax = plt.gca()
                    plot_model_metrics(ax,
                                       paths,
                                       xs,
                                       ys,
                                       path_name_map,
                                       prior_rate=(year_precision_prior, 1))
                    ax.plot([0, ax.get_xlim()[1]], [1, 1],
                            linestyle='dashed',
                            color='black',
                            linewidth=2)
                    ax.set_ylim(0.1, 2.0)
                    ax.set_xlabel('Precision @ 10%')
                    ax.set_ylabel(f'{metric_name} Disparity')
                    fig.legend(ncol=4,
                               loc='upper center',
                               bbox_to_anchor=(0, 0.85, 1, 0.15))
                    fig.tight_layout(rect=(0, 0, 1, 0.85))

                    fig.savefig(
                        os.path.join(
                            disparity_dir,
                            f'{group_feature_name}_{group_name}_{metric_name}_{year}.png'
                        ))
                    plt.close(fig)


if __name__ == '__main__':
    parse_results()
