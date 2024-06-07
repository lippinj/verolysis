from verolysis.piecewise_density import PiecewiseDensity, Segment


class PiecewiseDensityBuilder:
    """
    Build a PiecewiseDensity out of increasing steps
    """

    def __init__(self):
        self._segments = []
        self._prev = None

    def add(self, count, value):
        if self._prev is not None:
            pcount, pvalue = self._prev
            assert count > pcount
            assert value > pvalue
            density = (count - pcount) / (value - pvalue)
            self._segments.append(Segment(pvalue, value, density))
        self._prev = (count, value)

    def build(self) -> PiecewiseDensity:
        f = PiecewiseDensity()
        for seg in self._segments:
            f.add(seg)
        return f
