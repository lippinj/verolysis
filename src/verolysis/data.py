import pandas as pd
import statfin


def ansiotulot(year: int) -> pd.DataFrame:
    return _query("HVT_TULOT_70", year)


def palkkatulot(year: int) -> pd.DataFrame:
    return _query("HVT_TULOT_80", year)


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
        cache=f"tulot_101.{year}.{part}.df",
    )
    return df
