import numpy as np
import scipy
import warnings
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
            x0 = fr.init3()
            bounds = fr.bounds3()
            constraints = fr.constraints3()
        else:
            x0 = fr.init1()
            bounds = fr.bounds1()
            constraints = None
        opt = self._optimize(x0, bounds, constraints)
        if opt.success:
            self._x = opt.x
        return opt

    def build(self):
        f = PiecewiseDensity()
        for seg in self._full_segments():
            f.add(seg)
        return f

    def _optimize(self, x0, bounds, constraints) -> scipy.optimize.OptimizeResult:
        warnings.filterwarnings(
            "ignore",
            message="delta_grad == 0.0. Check if the approximated function is linear.",
        )
        opt = scipy.optimize.minimize(
            self._score,
            x0,
            method="trust-constr",
            bounds=bounds,
            constraints=constraints,
        )
        warnings.resetwarnings()
        return opt

    def _require_fringe_condition(self) -> "FringeCondition":
        if self._fringe_condition is None:
            self._fringe_condition = FringeCondition.for_optimizer(self)
        return self._fringe_condition

    def _compute_b(self, x: np.ndarray) -> tuple[float, float]:
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
        return fr.score_left1(x) + fr.score_right(*fr(x))

    def _score3(self, x):
        fr = self._require_fringe_condition()
        return fr.score_left3(x) + fr.score_right(*fr(x))

    def _fixed_sum(self):
        return np.sum([s.s for s in self._fixed_segments()])

    def _full_segments(self):
        assert self._x is not None
        if len(self._x) == 1:
            return self._full_segments1()
        assert len(self._x) == 3
        return self._full_segments3()

    def _full_segments1(self):
        x = self._x
        a = x[0]
        b, _ = self._compute_b(x)
        left = (0.0, a)
        right = (1.0, b)
        return self._segments([left] + self._places + [right])

    def _full_segments3(self):
        x = self._x
        a1, a0, n0 = x
        b, _ = self._compute_b(x)
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
            n = self._N * (rf - lf)
            yield Segment(lv, rv, n)


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
        fixed: list[Segment],
        N: float,
        m: float,
        fL: float,
        xL: float,
        fR: float,
        xR: float,
        amin: float,
    ):
        # Weights on the fringes
        nL = N * (fL - 0.0)
        nR = N * (1.0 - fR)

        # Sum budget for the fringes
        C = N * m - np.sum([seg.s for seg in fixed])
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
        self.fixed = fixed
        self.nL = nL
        self.xL = xL
        self.nR = nR
        self.xR = xR
        self.amin = amin
        if np.isfinite(amin):
            self.left_coeff = (self.amax - amin) / (xL - amin)
            self.amid = (xL + amin) / 2
        else:
            self.left_coeff = 1.0

    def __call__(self, x) -> tuple[float, float]:
        if len(x) == 1:
            a = x[0]
            b = (self.ka * a) + self.c
        else:
            assert len(x) == 3
            a1, a0, n0 = x
            b = (self.ka * a1) + (self.kn0 * n0) + (self.kn0a0 * n0 * a0) + self.c
        nb = self.nR / (b - self.xR)
        return b, nb

    def init1(self) -> list[float]:
        return [self.amax - 1.0]

    def init3(self) -> list[float]:
        return [self.amid, self.amin, self.nL / 2]

    def bounds1(self) -> list[tuple[float | None, float | None]]:
        return [(self.amin, self.amax)]

    def bounds3(self) -> list[tuple[float | None, float | None]]:
        return [(self.amin, self.xL), (self.amin, self.xL), (0.0, self.nL)]

    def constraints3(self) -> list[scipy.optimize.LinearConstraint]:
        # a1 - a0 >= 0
        return scipy.optimize.LinearConstraint([[1, -1, 0]], lb=0.0)

    def score_left1(self, x: np.ndarray) -> float:
        a1 = x[0]
        a2 = self.fixed[0].a
        a3 = self.fixed[1].a
        return self._score_eq(a3 - a2, a2 - a1)

    def score_left3(self, x: np.ndarray) -> float:
        return self.score_left1(x)

    def score_right(self, b: float, nb: float) -> float:
        b2 = self.fixed[-1].b
        b3 = self.fixed[-1].a
        return self._score_eq(b - b2, b2 - b3)

    def _score_eq(self, a: float, b: float) -> float:
        if a == 0 or b == 0:
            return 1.0
        else:
            r = min(a, b) / max(a, b)
            return (r - 1) ** 2

    @staticmethod
    def for_optimizer(opt: PiecewiseDensityOptimizer) -> "FringeCondition":
        N = opt._N
        m = opt._mean
        amin = opt._amin
        fixed = list(opt._fixed_segments())
        fL, xL = opt._places[0]
        fR, xR = opt._places[-1]
        return FringeCondition(fixed, N, m, fL, xL, fR, xR, amin)
