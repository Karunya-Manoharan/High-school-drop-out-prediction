from typing import List

import numpy as np
import pandas as pd

from parse import yamlobj
from preprocessor.base import Base


@yamlobj("!GradYearLabeler")
class GradYearLabeler(Base):
    """Label rows based on graduation and entry year."""
    yaml_tag = u'!GradYearLabeler'

    def __init__(self, grad_cols: List[str], entry_col: str, max_diff: int,
                 withdraw_columns: List[str], drop_columns: List[str]):
        """
        grad_col: name of column with graduation year
        entry_col: name of column with entry year
        max_diff: maximum difference between entry and graduation year to be considered successful graduation
        withdraw_columns: list of columns where non-null, shows the student withdrew and did not graduate
        """
        super(GradYearLabeler, self).__init__()
        self.grad_cols = grad_cols
        self.entry_col = entry_col
        self.max_diff = max_diff
        self.withdraw_columns = withdraw_columns
        self.drop_columns = drop_columns

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Labels students based on difference between entry and graduation
        year.

        data: input data indexed by student lookup
        """
        if self.drop_columns:
            data = data[data[self.drop_columns].notna().sum(axis=1) == 0]
        graduated_after = None
        graduated_after_nans = None
        for col_name in self.grad_cols:
            diffs = data[col_name] - data[self.entry_col]
            comparisons = diffs > self.max_diff
            if graduated_after is None:
                graduated_after = comparisons
                graduated_after_nans = diffs.isna().copy()
            else:
                graduated_after = graduated_after | comparisons
                graduated_after_nans = graduated_after_nans & diffs.isna()
        graduated_after[graduated_after_nans] = True
        has_withdraw = data[self.withdraw_columns].notna().sum(axis=1) > 0

        is_dropout = graduated_after | has_withdraw

        labels = pd.DataFrame(index=data.index,
                              data=is_dropout.astype(int),
                              columns=['Label'])
        return data, labels


@yamlobj("!DefaultToGraduationLabeler")
class DefaultToGraduationLabeler(Base):
    """Drops all students that never entered grade 12 per the data then labels
    rows based on graduation and entry year.

    Those without explicit grad year or drop out data are assumed to
    have graduated (i.e., default assumption is graduate).
    """
    yaml_tag = u'!DefaultToGraduationLabeler'

    def __init__(self, ever_12_col: str, grad_col: str, entry_col: str,
                 max_diff: int):
        """
        entered_12_col: name of column indicating student entered grade 12 in data
        grad_col: name of column with graduation year
        entry_col: name of column with entry year
        max_diff: maximum difference between entry and graduation year to be considered successful graduation
        """
        super(DefaultToGraduationLabeler, self).__init__()
        self.entered_12_col = entered_12_col
        self.grad_col = grad_col
        self.entry_col = entry_col
        self.max_diff = max_diff

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Labels students based on difference between entry and graduation
        year.

        data: input data indexed by student lookup
        """
        data = data[data[self.entered_12_col] == 1]
        col = ~(data[self.grad_col] - data[self.entry_col] <= self.max_diff)
        data = data.copy()
        labels = pd.DataFrame(index=col.index,
                              data=col.astype(int),
                              columns=['Label'])
        return data, labels
