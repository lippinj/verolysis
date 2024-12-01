import numpy as np
import pandas as pd

from verolysis.piecewise_density import PiecewiseDensity, Segment
from verolysis.piecewise_density_optimizer import PiecewiseDensityOptimizer


_FRAC_KEYS = (
    ("Q1", 0.25),
    ("Q3", 0.75),
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
        if row.Tuloluokka != "SS":
            fd = row_to_density(row)
            if fd is not None:
                f.add(fd)
    return f


def row_to_density(row) -> PiecewiseDensity | None:
    fracs = []
    for key, frac in _FRAC_KEYS:
        if key in row:
            if np.isfinite(row[key]):
                fracs.append((row[key], frac))
    if len(fracs) > 0:
        optimizer = PiecewiseDensityOptimizer(row.N, row.Mean, 0.0)
        for f in fracs:
            optimizer.add(*f)
        opt = optimizer.optimize()
        assert opt.success
        return optimizer.build()
    else:
        density = PiecewiseDensity()
        density.add(Segment(row.Mean, row.Mean, row.N))
        return density
