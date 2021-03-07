from typing import Any, List, Tuple

import pandas as pd

from parse import yamlobj
from preprocessor.base import Base


@yamlobj("!YearSplitter")
class YearSplitter(Base):
    """Preprocessor that splits data into train and test sets based on column
    values."""
    def __init__(self, train_len: int, gap: int, test_len: int,
                 year_col: int) -> 'YearSplitter':
        """
        train_len: length of train set in years
        test_len: length of test set in years
        gap: gap between train and test set in years
        year_col: column for setting year to split on
        """
        self.train_len = train_len
        self.test_len = test_len
        self.gap = gap
        self.year_col = year_col

    def transform(
            self, df: pd.DataFrame, labels: pd.DataFrame
    ) -> List[Tuple[int, pd.DataFrame, pd.DataFrame]]:
        df['Year'] = df[self.year_col]

        min_year, max_year = df['Year'].min(), df['Year'].max()
        assert self.train_len + self.gap + self.test_len <= max_year - min_year + 1, f'Not enough data for train test split, min year: {min_year}, max_year: {max_year}'

        start_year = min_year + self.train_len + self.gap
        end_year = max_year - self.test_len + 1
        splits = []
        for year in range(start_year, end_year + 1):
            train_start = year - self.gap - self.train_len
            train_end = train_start + self.train_len - 1
            train_indices = df[(df['Year'] >= train_start)
                               & (df['Year'] <= train_end)].index.unique()
            train_set = (df.loc[train_indices, :].copy(),
                         labels.loc[train_indices, :].copy())

            test_end = year + self.test_len - 1
            test_indices = df[(df['Year'] >= year)
                              & (df['Year'] <= test_end)].index.unique()
            test_set = (df.loc[test_indices, :].copy(),
                        labels.loc[test_indices, :].copy())
            splits.append((year, train_set, test_set))
            assert not train_set[0].index.duplicated().any()
            assert train_set[0].index.is_unique
            assert not test_set[0].index.duplicated().any()
            assert test_set[0].index.is_unique
        return splits
