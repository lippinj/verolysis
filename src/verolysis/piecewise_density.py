import dataclasses
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

    @property
    def empty(self):
        return len(self._segments) == 0

    @property
    def xmin(self):
        return None if self.empty else self._segments[0].a

    @property
    def xmax(self):
        return None if self.empty else self._segments[-1].b

    def sum(self, a=None, b=None) -> float:
        """
        Integral of the density function from a to b

        None means infinity (negative infinity for a, positive for b).
        """
        n = 0.0
        for seg in self._segments:
            overlap = seg.overlap(a, b)
            if overlap:
                n += overlap.n
        return n

    def add(self, arg: "Segment" | "PiecewiseDensity") -> None:
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

    def _assimilate(self, incoming: "Segment") -> None:
        segments = []
        for existing in self._segments:
            if incoming is None:
                segments.append(existing)
            else:
                processed, incoming = Segment.merge(existing, incoming)
                segments += processed
        self._segments = segments


@dataclasses.dataclass
class Segment:
    """A segment of PiecewiseDensity"""

    a: float
    b: float
    h: float

    @property
    def w(self):
        return self.b - self.a

    @property
    def n(self):
        return self.w * self.h

    def overlap(self, a, b):
        a = self.a if a is None else max(a, self.a)
        b = self.b if b is None else min(b, self.b)
        return Segment(a, b, self.h) if b > a else None

    @staticmethod
    def merge(s: "Segment", t: "Segment") -> tuple[list["Segment"], "Segment" | None]:
        if s.a < t.a:
            if s.b <= t.a:
                return [s], t
            elif s.b < t.b:
                return [Segment(s.a, t.a, s.h), Segment(t.a, s.b, s.h + t.h)], Segment(
                    s.b, t.b, t.h
                )
            elif s.b == t.b:
                return [Segment(s.a, t.a, s.h), Segment(t.a, s.b, s.h + t.h)], None
            else:
                return [Segment(s.a, t.a, s.h), Segment(t.a, t.b, s.h + t.h)], Segment(
                    t.b, s.b, s.h
                )
        elif s.a == t.a:
            if s.b < t.b:
                return [Segment(s.a, s.b, s.h + t.h)], Segment(s.b, t.b, t.h)
            elif s.b == t.b:
                return [Segment(s.a, s.b, s.h + t.h)], None
            else:
                return [Segment(s.a, t.b, s.h + t.h)], Segment(t.b, s.b, s.h)
        elif s.a < t.b:
            if s.b < t.b:
                return [Segment(t.a, s.a, t.h), Segment(s.a, s.b, t.h + s.h)], Segment(
                    s.b, t.b, t.h
                )
            elif s.b == t.b:
                return [Segment(t.a, s.a, t.h), Segment(s.a, t.b, t.h + s.h)], None
            else:
                return [Segment(t.a, s.a, t.h), Segment(s.a, t.b, t.h + s.h)], Segment(
                    t.b, s.b, s.h
                )
        else:
            return [t], s
