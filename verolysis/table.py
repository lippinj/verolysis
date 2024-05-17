import json
import pandas as pd
import numpy as np


class Table:

    def __init__(self):
        self.df = None
        self._key_cols = None
        self._value_cols = None

    def load(self, path):
        """Load data from a JSON file"""
        with open(path, "r") as f:
            self.load_json(json.load(f))

    def where(self, **kwargs):
        """Filter dataframe by key attributes"""
        df = self.df
        for key, value in kwargs.items():
            df = df[df[key] == str(value)]
        return df

    def income_class_column(self, year, group, erä, col_name="Mean") -> np.ndarray:
        """Return column that spans income classes"""
        df = self.where(
            Verovuosi = 2022,
            Tulonsaajaryhmä = group,
            Erä = erä,
        )
        df = df[df["Tuloluokka"] != "SS"]
        return df[col_name].to_numpy()

    def load_json(self, j):
        """Load data from a JSON dictionary object

        If the table has been previously populated, the columns of the
        new data must exactly match the old data.
        """
        kcol, vcol = self._load_cols(j["columns"])
        data = self._load_data(j["data"], kcol, vcol)
        df = pd.DataFrame(data=data)
        self._use_frame(df)

    def _use_frame(self, df):
        if self.df is None:
            self.df = df
        else:
            self.df = pd.concat([self.df, df]).drop_duplicates()

    def _load_cols(self, j):
        key_cols, value_cols = Table._parse_cols(j)
        if self.df is not None:
            assert self._key_cols == set(key_cols)
            assert self._value_cols == set(value_cols)
        self._key_cols = set(key_cols)
        self._value_cols = set(value_cols)
        return key_cols, value_cols

    def _load_data(self, j, key_cols, value_cols):
        all_cols = key_cols + value_cols
        data = {code: [] for code in all_cols}
        key = [data[code] for code in key_cols]
        val = [data[code] for code in value_cols]
        for j_data in j:
            for col, v in zip(key, j_data["key"]):
                col.append(v)
            for col, v in zip(val, j_data["values"]):
                col.append(Table._to_value(v))
        return data

    @staticmethod
    def _parse_cols(j):
        key_cols = []
        value_cols = []
        for j_col in j:
            code = j_col["code"]
            typ = j_col["type"]
            if typ == "c":
                value_cols.append(code)
            else:
                key_cols.append(code)
        return key_cols, value_cols

    @staticmethod
    def _to_value(x):
        try:
            return float(x)
        except ValueError:
            return None


def load(path):
    """Load from JSON to Table"""
    table = Table()
    table.load(path)
    return table
