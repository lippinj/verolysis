from verolysis import PiecewiseDensityOptimizer
from numpy.testing import assert_almost_equal


def test_optimize_1():
    N = 279_305
    m = -17_467
    p25 = -16_951
    p50 = -8_141
    p75 = -3_630

    opt = PiecewiseDensityOptimizer(N, m)
    opt.add(p25, 0.25)
    opt.add(p50, 0.50)
    opt.add(p75, 0.75)
    o = opt.optimize()
    assert o.success

    f = opt.build()
    print(f._segments)
    assert len(f._segments) == 4
    assert f._segments[0].b == p25
    assert f._segments[1].a == p25
    assert f._segments[1].b == p50
    assert f._segments[2].a == p50
    assert f._segments[2].b == p75
    assert f._segments[3].a == p75
    assert_almost_equal(f.count(), N, decimal=2)
    assert_almost_equal(f.sum(), N * m, decimal=2)


def test_optimize_2():
    opt = PiecewiseDensityOptimizer(4_777_805, 31_781, 0.0)
    opt.add(14_569, 0.25)
    opt.add(41_707, 0.75)
    opt.add(7_624, 0.10)
    opt.add(12_151, 0.20)
    opt.add(16_700, 0.30)
    opt.add(21_503, 0.40)
    opt.add(26_755, 0.50)
    opt.add(32_222, 0.60)
    opt.add(38_199, 0.70)
    opt.add(45_982, 0.80)
    opt.add(59_444, 0.90)
    o = opt.optimize()
    assert o.success

    f = opt.build()
    assert len(f._segments) == 12
    assert f._segments[0].b == 7_624
    assert f._segments[11].a == 59_444
    assert f._segments[0].a >= 0
    assert f._segments[0].a < 7_624
    assert f._segments[11].b > 59_444
    assert_almost_equal(f.count(), 4_777_805, decimal=2)
    assert_almost_equal(f.sum(), 4_777_805 * 31_781, decimal=2)
