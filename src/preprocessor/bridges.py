from typing import Any, Iterable, List

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from model import ClfWrapper
from parse import yamlobj
from preprocessor.base import Base


@yamlobj("!ColumnPreprocessor")
class ColumnPreprocessor(ClfWrapper, Base):
    def transform(self, data):
        (train_df, train_labels), (test_df, test_labels) = data

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numeric_train = train_df.select_dtypes(include=numerics)
        numeric_columns = list(numeric_train.columns.values)
        self.estimator.fit(numeric_train.to_numpy(np.float))
        train_df[numeric_columns] = self.estimator.transform(numeric_train)
        test_df[numeric_columns] = self.estimator.transform(
            test_df[numeric_columns].to_numpy(np.float))
        return (train_df, train_labels), (test_df, test_labels)


@yamlobj("!SeqPreprocessor")
class SeqPreprocessor(Base):
    def __init__(self, preprocessors: List[Base]):
        super(SeqPreprocessor, self).__init__()
        self.preprocessors = preprocessors

    def transform(self, data):
        for preprocessor in self.preprocessors:
            data = preprocessor.transform(data)
        return data


@yamlobj("!DropPreprocessor")
class DropPreprocessor(Base):
    def __init__(self, keep: List[str]):
        super(DropPreprocessor, self).__init__()
        self.keep = keep

    def transform(self, data: Iterable[pd.DataFrame]) -> List[pd.DataFrame]:
        data_list = [df for df in data]
        for df in data_list:
            df.drop(df.columns.difference(self.keep), 1, inplace=True)
        for df in data_list:
            print(df)
            df.dropna(inplace=True)
            print(df)
        return data_list


@yamlobj("!NumpyPreprocessor")
class NumpyPreprocessor(Base):
    def __init__(self, label_col: str):
        self.label_col = label_col

    def transform(self, datasets: Iterable[pd.DataFrame]):
        results = []
        for df in datasets:
            labels = df.loc[:, self.label_col]
            features = df.loc[:, df.columns != self.label_col]
            results.extend(
                [features.to_numpy(copy=True),
                 labels.to_numpy(copy=True)])
        return results


@yamlobj("!MappingPreprocessor")
class MappingPreprocessor(Base):
    def __init__(self, preprocessors: List[Base]):
        super(MappingPreprocessor, self).__init__()
        self.preprocessors = preprocessors

    def transform(self, data: List[Any]):
        return [self.preprocessors.transform(data_item) for data_item in data]
