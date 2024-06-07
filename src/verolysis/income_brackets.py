import scipy
import pandas as pd

from verolysis.piecewise_density import PiecewiseDensity
from verolysis.piecewise_density_optimizer import PiecewiseDensityOptimizer


_FRAC_KEYS = (
    ("Q1", 0.25),
    ("Q2", 0.25),
    ("P10", 0.10),
    ("P20", 0.20),
    ("P30", 0.30),
    ("P40", 0.40),
    ("P50", 0.50),
    ("P60", 0.60),
    ("P70", 0.70),
    ("P80", 0.80),
    ("P90", 0.90),
)


def to_density(df: pd.DataFrame) -> PiecewiseDensity:
    f = PiecewiseDensity()
    for i in range(len(df)):
        row = df.iloc[i]
        if row.Tuloluokka != "Y":
            optimizer, opt = _optimize_row_density(row)
            f.add(optimizer.build())
    return f


def _optimize_row_density(
    row,
) -> tuple[PiecewiseDensityOptimizer, scipy.optimize.OptimizeResult]:
    optimizer = PiecewiseDensityOptimizer(row.N, row.Mean, 0.0)
    for key, frac in _FRAC_KEYS:
        if key in row:
            optimizer.add(row[key], frac)
    return optimizer, optimizer.optimize()
