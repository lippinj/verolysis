import numpy as np
import scipy
from typing import Union
from verolysis.piecewise_density import PiecewiseDensity, Segment


class PiecewiseDensityOptimizer:
    """Optimizing builder for PiecewiseDensity"""

    def __init__(self, N: float, mean: float, amin: float = -np.inf):
        self._N: float = N
        self._mean: float = mean
        self._total: float = N * mean
        self._places: list[tuple[float, float]] = []
        self._amin: float = amin
        self._x: np.ndarray | None = None
        self._fringe_condition: Union["FringeCondition", None] = None

    def add(self, value: float, fractile: float) -> None:
        assert fractile > 0.0
        assert fractile < 1.0
        self._places.append((fractile, value))
        self._places.sort()
        self._opt_params = None

    def optimize(self) -> scipy.optimize.OptimizeResult:
        fr = self._require_fringe_condition()
        if fr.left_coeff < 0.5:
            x0 = [fr.amid, fr.amin, 0.0]
            bounds = fr.bounds3()
        else:
            x0 = [fr.amax]
            bounds = fr.bounds1()
        opt = self._optimize(x0, bounds)
        self._x = opt.x
        return opt

    def build(self):
        f = PiecewiseDensity()
        for seg in self._full_segments():
            f.add(seg)
        return f

    def _optimize(self, x0, bounds) -> scipy.optimize.OptimizeResult:
        opt = scipy.optimize.minimize(
            self._score,
            x0,
            method="trust-constr",
            bounds=bounds,
        )
        if not opt.success:
            raise opt
        if opt.x[0] >= self._places[0][1]:
            raise opt
        if self._compute_b(opt.x) <= self._places[-1][1]:
            raise opt
        return opt

    def _require_fringe_condition(self) -> "FringeCondition":
        if self._fringe_condition is None:
            self._fringe_condition = FringeCondition.for_optimizer(self)
        return self._fringe_condition

    def _compute_b(self, x: np.ndarray) -> float:
        fr = self._require_fringe_condition()
        return fr(x)

    def _compute_amax(self) -> float:
        fr = self._require_fringe_condition()
        return fr.amax

    def _score(self, x):
        if len(x) == 1:
            return self._score1(x)
        else:
            assert len(x) == 3
            return self._score3(x)

    def _score1(self, x):
        fr = self._require_fringe_condition()
        b = fr(x)
        s1 = self._score_line(x[0], self._places[0][1], self._places[1][1])
        s2 = self._score_line(self._places[-2][1], self._places[-1][1], b)
        return s1 + s2

    def _score3(self, x):
        fr = self._require_fringe_condition()
        a1, a0, n0 = x
        b = fr(x)
        s0 = self._score_line(a0, a1, self._places[0][1])
        s1 = self._score_line(a1, self._places[0][1], self._places[1][1])
        s2 = self._score_line(self._places[-2][1], self._places[-1][1], b)
        return s0 + s1 + s2

    def _score_line(self, a, b, c):
        ab = b - a
        bc = c - b
        if ab == 0 or bc == 0:
            return 1.0
        else:
            return (min(ab, bc) / max(ab, bc) - 1) ** 2

    def _fixed_sum(self):
        return np.sum([s.s for s in self._fixed_segments()])

    def _full_segments(self):
        assert self._x is not None
        if len(self._x) == 1:
            return self._full_segments1()
        else:
            assert len(self._x) == 3
            return self._full_segments3()

    def _full_segments1(self):
        x = self._x
        a = x[0]
        b = self._compute_b(x)
        left = (0.0, a)
        right = (1.0, b)
        return self._segments([left] + self._places + [right])

    def _full_segments3(self):
        x = self._x
        a1, a0, n0 = x
        b = self._compute_b(x)
        left0 = (0.0, a0)
        left1 = (n0 / self._N, a1)
        right = (1.0, b)
        return self._segments([left0, left1] + self._places + [right])

    def _fixed_segments(self):
        return self._segments(self._places)

    def _segments(self, places: list[tuple[float, float]]):
        for i in range(len(places) - 1):
            lf, lv = places[i]
            rf, rv = places[i + 1]
            df = rf - lf
            dv = rv - lv
            h = self._N * df / dv
            yield Segment(lv, rv, h)


class FringeCondition:
    """
    Mathematical relationship between left and right fringe

    We always have at least one fringe segment on both the left and the right
    of the fixed region (the region covered by known fractiles). Optionally, we
    can have two segments on the left (sometimes this is necessary because
    otherwise we are obliged to break the lower bound).
    """

    def __init__(
        self,
        N: float,
        m: float,
        s: float,
        fL: float,
        xL: float,
        fR: float,
        xR: float,
        amin: float | None = None,
    ):
        # Weights on the fringes
        nL = N * (fL - 0.0)
        nR = N * (1.0 - fR)

        # Sum budget for the fringes
        C = N * m - s
        nC = nL + nR
        assert nC > 0
        assert amin is None or C > amin * nC

        # Conversion arguments from left fringe to right fringe
        self.ka = -(nL / nR)
        self.kn0 = xL / nR
        self.kn0a0 = -(1 / nR)
        self.c = (2 * C / nR) + (self.ka * xL) - xR

        # One-segment maximum for a
        self.amax = min(xL, (xR - self.c) / self.ka)

        # Miscellaneous constants
        self.nL = nL
        self.xL = xL
        self.amin = amin
        if amin is not None:
            self.left_coeff = (self.amax - amin) / (xL - amin)
            self.amid = (xL + amin) / 2
        else:
            self.left_coeff = 1.0

    def __call__(self, x) -> float:
        if len(x) == 1:
            a = x[0]
            return (self.ka * a) + self.c
        else:
            assert len(x) == 3
            a1 = x[0]
            a0 = x[1]
            n0 = x[2]
            return (self.ka * a1) + (self.kn0 * n0) + (self.kn0a0 * n0 * a0) + self.c

    def bounds1(self) -> tuple[tuple[float, float]]:
        return ((self.amin, self.amax),)

    def bounds3(self) -> scipy.optimize.Bounds:
        return ((self.amin, self.xL), (self.amin, self.xL), (0.0, self.nL))

    @staticmethod
    def for_optimizer(opt: PiecewiseDensityOptimizer) -> "FringeCondition":
        N = opt._N
        m = opt._mean
        amin = opt._amin
        s = opt._fixed_sum()
        fL, xL = opt._places[0]
        fR, xR = opt._places[-1]
        return FringeCondition(N, m, s, fL, xL, fR, xR, amin)
