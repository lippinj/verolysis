import pandas as pd
import statfin


def ansiotulot(year: int, group=None) -> pd.DataFrame:
    df = _query("HVT_TULOT_70", year)
    if group:
        df = df[df.Tulonsaajaryhmä == str(group)]
    return df


def palkkatulot(year: int, group=None) -> pd.DataFrame:
    df = _query("HVT_TULOT_80", year)
    if group:
        df = df[df.Tulonsaajaryhmä == str(group)]
    return df


def _query(part: str, year: int) -> pd.DataFrame:
    db = statfin.PxWebAPI.Verohallinto()
    table = db.table("Vero", "tulot_101.px")
    df = table.query(
        {
            "Verovuosi": year,
            "Tulonsaajaryhmä": "*",
            "Tuloluokka": "*",
            "Tunnusluvut": "*",
            "Erä": part,
        },
        cache=f"tulot_101.{year}.{part}",
    )
    return df
