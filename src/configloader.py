from functools import reduce

import yaml

from load_data import get_loader, get_graduation_info
from model import *
from preprocessor import *


def load_config(config_path: str):
    with open(config_path) as in_f:
        config = yaml.load(in_f, Loader=yaml.FullLoader)
    loaders = config['loaders']
    labeler = config['labeler']
    splitter = config['splitter']
    preprocessor = SeqPreprocessor(config['preprocessor'])
    model = config['model']
    imputer = config['imputer']
    return loaders, labeler, splitter, preprocessor, imputer, model


def create_data_from_loaders(loaders, cur):
    return reduce(lambda x, y: x.join(y, how='outer'), [
        get_loader(loader_key)(cur=cur, **
                               loader_kwargs).set_index('student_lookup')
        for loader_key, loader_kwargs in loaders.items()
    ])


def create_splits(loaders, labeler, splitter, cur):
    df = get_graduation_info(cur)
    grad_info_cols = df.columns

    df['student_lookup'] = df['student_lookup'].astype(int)
    df = df.set_index('student_lookup')
    assert df.index.is_unique
    loaded_df = create_data_from_loaders(loaders, cur)
    assert loaded_df.index.is_unique
    df = df.join(loaded_df)
    assert df.index.is_unique

    df, labels = labeler.transform(df)
    splits = splitter.transform(df, labels)
    """Drop graduation info columns"""
    info_cols = [col for col in grad_info_cols if col != 'student_lookup']
    for year, (train_df, _), (test_df, _) in splits:
        train_df.drop(info_cols, inplace=True, axis=1)
        test_df.drop(info_cols, inplace=True, axis=1)
    return splits
