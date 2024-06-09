import pandas as pd
import verolysis


def test_data_full_table():
    df = verolysis.data.ansiotulot(2022)
    assert isinstance(df, pd.DataFrame)

    df = verolysis.data.palkkatulot(2022)
    assert isinstance(df, pd.DataFrame)


def test_data_group_filter():
    df = verolysis.data.ansiotulot(2022, "Y")
    assert len(df.Tulonsaajaryhmä.unique() == 1)

    df = verolysis.data.ansiotulot(2022, 2)
    assert len(df.Tulonsaajaryhmä.unique() == 1)
