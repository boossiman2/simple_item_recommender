import pandas as pd

class Reader():
    def __init__(self, data_path: str, dataset: str):
        self.data_path = data_path
        self.dataset = dataset

    def read_csv(self) -> pd.DataFrame:
        return self.read_csv(self.data_path+self.dataset)

    def drop_columns(self, df: pd.DataFrame, drop_cols: list) -> pd.DataFrame:
        return df.drop(columns=drop_cols, errors='ignore')

    def shuffle_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sample(frac=1).reset_index(drop=True)

class Preprocessor(object):
    def __init__(self, data_path):
        self.data_path = data_path