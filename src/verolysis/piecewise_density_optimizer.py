import numpy as np
import scipy
from verolysis.piecewise_density import PiecewiseDensity, Segment


class PiecewiseDensityOptimizer:
    """Optimizing builder for PiecewiseDensity"""

    def __init__(self, N: float, mean: float, xmin: float = -np.inf):
        self._N: float = N
        self._mean: float = mean
        self._total: float = N * mean
        self._places: list[tuple[float, float]] = []
        self._xmin: float = xmin
        self._left: float | None = None
        self._right: float | None = None
        self._opt_kc: tuple[float, float] | None = None

    def add(self, value: float, fractile: float) -> None:
        assert fractile > 0.0
        assert fractile < 1.0
        self._places.append((fractile, value))
        self._places.sort()
        self._opt_kc = None

    def optimize(self) -> scipy.optimize.OptimizeResult:
        opt = self._optimize()
        self._left = opt.x[0]
        self._right = self._compute_b(opt.x[0])
        return opt

    def build(self):
        f = PiecewiseDensity()
        for seg in self._full_segments():
            f.add(seg)
        return f

    def _optimize(self) -> scipy.optimize.OptimizeResult:
        opt = scipy.optimize.minimize(
            self._score,
            self._compute_amax(),
            method="trust-constr",
            bounds=scipy.optimize.Bounds(self._xmin, self._compute_amax()),
        )
        if not opt.success:
            raise opt
        if opt.x[0] >= self._places[0][1]:
            raise opt
        if self._compute_b(opt.x[0]) <= self._places[-1][1]:
            raise opt
        return opt

    def _compute_constants(self) -> tuple[float, float]:
        fL, xL = self._places[0]
        fR, xR = self._places[-1]
        N = self._N
        m = self._mean

        n1 = N * (fL - 0.0)
        nN = N * (1.0 - fR)
        S = N * m
        M = self._fixed_sum()
        assert n1 > 0
        assert nN > 0

        k = -(n1 / nN)
        assert k < 0
        c = (2 / nN) * (S - M - (n1 / 2) * xL - (nN / 2) * xR)
        self._opt_kc = (k, c)
        return k, c

    def _require_constants(self) -> tuple[float, float]:
        if self._opt_kc is None:
            return self._compute_constants()
        return self._opt_kc

    def _compute_b(self, a: float) -> float:
        k, c = self._require_constants()
        return k * a + c

    def _compute_amax(self) -> float:
        k, c = self._require_constants()
        _, xL = self._places[0]
        _, xR = self._places[-1]
        return min(xL, (xR - c) / k)

    def _score(self, x):
        a = x[0]
        b = self._compute_b(a)
        s1 = self._score_line(a, self._places[0][1], self._places[1][1])
        s2 = self._score_line(self._places[-2][1], self._places[-1][1], b)
        return s1 + s2

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
        assert self._left is not None
        assert self._right is not None
        left = (0.0, self._left)
        right = (1.0, self._right)
        return self._segments([left] + self._places + [right])

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
