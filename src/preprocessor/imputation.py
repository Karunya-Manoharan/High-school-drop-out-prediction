from typing import Any, Dict

import numpy as np
import pandas as pd

from parse import yamlobj
from preprocessor import Base
import re

_impute_method_map = {
    'mean': lambda column: column.mean(),
    'median': lambda column: column.median(),
    'mode': lambda column: column.mode(),
    'zero': lambda column: 0
}


def _get_impute_values(df: pd.DataFrame, col_methods: Dict[str, str],
                       regex_methods: Dict[str, str], default_method: str):
    impute_values = {}
    default_method_fn = _impute_method_map[default_method]

    if len(regex_methods)>0:
        conditions = '|'.join(list(regex_methods.keys()))
        regex_arr = df.columns.str.contains(conditions, regex=True)
        regex_columns = df.columns[regex_arr]
        other_columns = df.columns[~regex_arr]

        for column in regex_columns:
            for key in regex_methods.keys():
                if key in column:
                    impute_values[column] = _impute_method_map[
                        regex_methods[key]](df[column])
        
        for column in other_columns:
            if column in col_methods:
                impute_values[column] = _impute_method_map[
                    col_methods[column]](df[column])
            else:
                impute_values[column] = default_method_fn(df[column])

    else:
        for column in df.columns:
            if column in col_methods:
                impute_values[column] = _impute_method_map[
                    col_methods[column]](df[column])
            else:
                impute_values[column] = default_method_fn(df[column])

    return impute_values


def _impute_columns(df: pd.DataFrame, impute_values: Dict[str, Any]):
    for column in df.columns:
        if column in impute_values:
            df[f'{column}_imputed'] = df[column].isnull().copy()
            if not np.isfinite(impute_values[column]):
                df[column] = df[column].fillna(1).copy()
            else:
                df[column] = df[column].fillna(impute_values[column])
            assert not df[column].isnull().values.any(), (df, df[column])


@yamlobj("!DropImputer")
class DropImputer:
    def transform(self, train_df, test_df):
        results = []
        for df in [train_df, test_df]:
            nulls = df.isnull().any(axis=1)
            results.append(df[~nulls])
        return tuple(results)


@yamlobj("!AverageImputer")
class AverageImputer:
    def __init__(self, default_method: str, col_methods: Dict[str, str],
                 regex_methods: Dict[str, str]):

        for key, value in col_methods.items():
            assert value in _impute_method_map, f'Column name {key} with invalid method name {value}'
        assert default_method in _impute_method_map
        self.default_method = default_method
        self.col_methods = col_methods
        self.regex_methods = regex_methods

    def transform(self, train_df, test_df):
        impute_values = _get_impute_values(train_df, self.col_methods,
                                           self.regex_methods,
                                           self.default_method)
        assert train_df.index.is_unique
        _impute_columns(train_df, impute_values)
        assert train_df.index.is_unique
        _impute_columns(test_df, impute_values)
        return train_df, test_df
