from itertools import product
import multiprocess as mp
import os

import click
import yaml

from experiment import Experiment


@click.group()
def main():
    pass


@main.command()
@click.option('--config_path', required=True, type=click.Path())
@click.option('--secret_path', default='.secrets.yaml', type=click.Path())
@click.option('--model_save_dir', default='model', type=click.Path())
@click.option('--result_save_dir', default='result', type=click.Path())
def run(config_path, secret_path, model_save_dir, result_save_dir):
    exp = Experiment(config_path, secret_path, model_save_dir, result_save_dir)
    exp.run()


def make_sklearn_model_config(model_config):
    return "model:\n    !ClfWrapper\n" + '\n'.join(
        [" " * 8 + line for line in yaml.dump(model_config).split('\n')])


def generate_model_configs(module, name, params):
    if not params:
        return [{'clf_module': module, 'clf_name': name, 'clf_params': {}}]
    keys, values = zip(*(list(params.items())))
    model_configs = [{
        'clf_module': module,
        'clf_name': name,
        'clf_params': {key: value
                       for key, value in zip(keys, param_config)}
    } for param_config in product(*values)]
    return model_configs


@main.command()
@click.option('--grid_path', required=True, type=click.Path())
@click.option('--config_path', required=True, type=click.Path())
@click.option('--secret_path', default='.secrets.yaml', type=click.Path())
@click.option('--result_save_dir', default='grid_results', type=click.Path())
def grid(grid_path, config_path, secret_path, result_save_dir):
    with open(grid_path) as in_f:
        grid_list = yaml.load(in_f, Loader=yaml.SafeLoader)
    with open(config_path) as in_f:
        config_text = in_f.read()

    model_configs = [
        model_config for grid_config in grid_list
        for model_config in generate_model_configs(**grid_config)
    ]
    name_ct_map = {}
    paths = []
    for model_config in model_configs:
        name = model_config['clf_name']
        idx = name_ct_map.setdefault(name, 0)
        name_ct_map[name] += 1
        path = os.path.join(result_save_dir, f'{name}_{idx}')
        if not os.path.exists(path):
            os.makedirs(path)
        paths.append(path)
        with open(os.path.join(path, 'config.yaml'), 'w') as in_f:
            in_f.write(config_text + '\n')
            in_f.write(make_sklearn_model_config(model_config))
    args = [(os.path.join(path, 'config.yaml'), secret_path, path, path)
            for path in paths]
    with mp.Pool(5) as p:
        p.starmap(run.callback, args)


if __name__ == '__main__':
    main()
