import dataclasses
from typing import Union

import numpy as np


class PiecewiseDensity:
    """
    Piecewise constant density function

    This is a function composed of segments, i.e., a set of tuples (a, b, h)
    such that, when x is in the range [a, b), the value of the function is h.
    The segments do not overlap.

    Outside of any defined segments, the function is zero.
    """

    def __init__(self):
        self._segments = []

    def __len__(self):
        return len(self._segments)

    def __getitem__(self, i):
        return self._segments[i]

    @property
    def empty(self) -> bool:
        return len(self._segments) == 0

    @property
    def xmin(self):
        return None if self.empty else self[0].a

    @property
    def xmax(self):
        return None if self.empty else self[-1].b

    @property
    def va(self):
        return np.array([seg.a for seg in self._segments])

    @property
    def vb(self):
        return np.array([seg.b for seg in self._segments])

    @property
    def vw(self):
        return np.array([seg.w for seg in self._segments])

    @property
    def vh(self):
        return np.array([seg.h for seg in self._segments])

    def icount(self, n, left=None, right=None) -> float:
        """
        Inverse count function

        Finds x such that count(None, x) == n.
        """
        if n < 0:
            return self.xmin if left is None else left
        i = 0
        for seg in self._segments:
            j = i + seg.n
            if j >= n:
                dn = n - i
                dx = dn / seg.h if seg.w > 0 else 0
                return seg.a + dx
            i = j
        return self.xmax if right is None else right

    def uniform_sample(
        self, k, leftpad=None, rightpad=None, left=None, right=None
    ) -> float:
        """Uniform sampling of values"""
        assert leftpad is None or rightpad is None
        count = self.count()
        a = count - leftpad if leftpad else 0
        b = rightpad if rightpad else count
        step = (b - a) / k
        return np.array(
            [
                self.icount(a + (i + 0.5) * step, left=left, right=right)
                for i in range(k)
            ]
        )

    def count(self, a=None, b=None) -> float:
        """
        Integral of the density function, from a to b

        None means infinity (negative infinity for a, positive for b).
        """
        n = 0.0
        for seg in self._segments:
            overlap = seg.overlap(a, b)
            if overlap:
                n += overlap.n
        return n

    def sum(self, a=None, b=None) -> float:
        """
        Integral of the density function times x, from a to b

        None means infinity (negative infinity for a, positive for b).
        """
        n = 0.0
        for seg in self._segments:
            overlap = seg.overlap(a, b)
            if overlap:
                n += overlap.s
        return n

    def sum_above(self, a: float) -> float:
        """Sum of only the excess of values above a"""
        return self.sum(a, None) - (a * self.count(a, None))

    def mean(self, a=None, b=None) -> float:
        """
        Average of the function

        None means infinity (negative infinity for a, positive for b).
        """
        N = self.count(a, b)
        if N > 0.0:
            return self.sum(a, b) / N
        else:
            return np.nan

    def tail_ratio(self, x: float) -> float:
        """
        Ratio of mean above x to x

        This fraction occurs in Saez (2001) as the ratio of mean income above
        z to z, i.e., z_m / z. It is related to the pareto parameter.
        """
        return self.mean(x, None) / x

    def pareto(self, x: float) -> float:
        """Pareto parameter of the tail above x"""
        assert x >= 0
        if x > 0:
            r = self.tail_ratio(x)
            return r / (r - 1)
        else:
            return 1.0

    def add(self, arg: Union["Segment", "PiecewiseDensity"]) -> None:
        """Sum a new segment into this function"""
        if isinstance(arg, Segment):
            return self._add_segment(arg)
        else:
            for seg in arg._segments:
                self._add_segment(seg)

    def _add_segment(self, seg: "Segment") -> None:
        if len(self._segments) == 0:
            self._segments = [seg]
        else:
            self._assimilate(seg)

    def _assimilate(self, incoming: Union["Segment", None]) -> None:
        segments = []
        for existing in self._segments:
            if incoming is None:
                segments.append(existing)
            else:
                processed, incoming = Segment.merge(existing, incoming)
                segments += processed
        if incoming:
            segments.append(incoming)
        self._segments = segments


@dataclasses.dataclass
class Segment:
    """A segment of PiecewiseDensity"""

    a: float
    b: float
    n: float

    @property
    def w(self):
        return self.b - self.a

    @property
    def m(self):
        return (self.a + self.b) / 2

    @property
    def h(self):
        return self.n / self.w

    @property
    def s(self):
        return self.n * self.m

    def overlap(self, a, b):
        if self.w > 0:
            a = self.a if a is None else max(a, self.a)
            b = self.b if b is None else min(b, self.b)
            return Segment(a, b, self.h * (b - a)) if b > a else None
        if a is None and b is None:
            return self
        if a is None:
            return self if b > self.a else None
        if b is None:
            return self if a <= self.a else None
        if a == b:
            return self if a == self.a else None
        return self if a <= self.a and b > self.b else None

    @staticmethod
    def merge(
        s: "Segment", t: "Segment"
    ) -> tuple[list["Segment"], Union["Segment", None]]:
        S = Segment
        if s.w == 0 and t.w == 0:
            if s.a < t.a:
                return [s], t
            elif s.a == t.a:
                return [S(s.a, s.b, s.n + t.n)], None
            else:
                return [t], s
        elif s.w == 0:
            if t.b <= s.a:
                return [t], s
            elif t.a >= s.a:
                return [s], t
            else:
                return [t.overlap(t.a, s.a), s], t.overlap(s.b, t.b)
        elif t.w == 0:
            return Segment.merge(t, s)
        if s.a < t.a:
            if s.b <= t.a:
                return [s], t
            elif s.b < t.b:
                return [
                    S(s.a, t.a, s.h * (t.a - s.a)),
                    S(t.a, s.b, (s.h + t.h) * (s.b - t.a)),
                ], S(s.b, t.b, t.h * (t.b - s.b))
            elif s.b == t.b:
                return [
                    S(s.a, t.a, s.h * (t.a - s.a)),
                    S(t.a, s.b, (s.h + t.h) * (s.b - t.a)),
                ], None
            else:
                return [
                    S(s.a, t.a, s.h * (t.a - s.a)),
                    S(t.a, t.b, (s.h + t.h) * (t.b - t.a)),
                ], S(t.b, s.b, s.h * (s.b - t.b))
        elif s.a == t.a:
            if s.b < t.b:
                return [S(s.a, s.b, (s.h + t.h) * (s.b - s.a))], S(
                    s.b, t.b, t.h * (t.b - s.b)
                )
            elif s.b == t.b:
                return [S(s.a, s.b, (s.h + t.h) * (s.b - s.a))], None
            else:
                return [S(s.a, t.b, (s.h + t.h) * (t.b - s.a))], S(
                    t.b, s.b, s.h * (s.b - t.b)
                )
        elif s.a < t.b:
            if s.b < t.b:
                return [
                    S(t.a, s.a, t.h * (s.a - t.a)),
                    S(s.a, s.b, (t.h + s.h) * (s.b - s.a)),
                ], S(s.b, t.b, t.h * (t.b - s.b))
            elif s.b == t.b:
                return [
                    S(t.a, s.a, t.h * (s.a - t.a)),
                    S(s.a, t.b, (t.h + s.h) * (t.b - s.a)),
                ], None
            else:
                return [
                    S(t.a, s.a, t.h * (s.a - t.a)),
                    S(s.a, t.b, (t.h + s.h) * (t.b - s.a)),
                ], S(t.b, s.b, s.h * (s.b - t.b))
        else:
            return [t], s
