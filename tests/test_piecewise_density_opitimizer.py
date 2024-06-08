from verolysis.piecewise_density_builder import PiecewiseDensityBuilder
from verolysis.piecewise_density_optimizer import PiecewiseDensityOptimizer
from numpy.testing import assert_almost_equal


def assert_segments_are_sane(segments):
    for seg in segments:
        assert seg.a < seg.b
    for i in range(len(segments) - 1):
        assert segments[i].b == segments[i + 1].a


def test_simple():
    opt = PiecewiseDensityOptimizer(100, 0.0)
    opt.add(-1, 0.25)
    opt.add(0, 0.50)
    opt.add(1, 0.75)
    o = opt.optimize()
    assert o.success

    f = opt.build()
    assert len(f) == 4
    assert_almost_equal(f.count(), 100, decimal=2)
    assert_almost_equal(f.sum(), 0, decimal=2)


def test_simple_nonnegative():
    opt = PiecewiseDensityOptimizer(100, 0.5, 0.0)
    opt.add(0.25, 0.25)
    opt.add(0.50, 0.50)
    opt.add(0.75, 0.75)
    o = opt.optimize()
    assert o.success

    f = opt.build()
    assert len(f) == 4
    assert_almost_equal(f.count(), 100, decimal=2)
    assert_almost_equal(f.sum(), 50, decimal=2)


def test_bunched_nonnegative():
    ref_builder = PiecewiseDensityBuilder()
    ref_builder.add(0, 0.0)
    ref_builder.add(24, 0.1)
    ref_builder.add(25, 1.0)
    ref_builder.add(50, 1.1)
    ref_builder.add(75, 1.2)
    ref_builder.add(100, 1.21)
    ref = ref_builder.build()
    assert_almost_equal(ref.count(), 100)

    opt = PiecewiseDensityOptimizer(ref.count(), ref.mean(), 0.0)
    opt.add(1.00, 0.25)
    opt.add(1.10, 0.50)
    opt.add(1.20, 0.75)
    o = opt.optimize()
    assert o.success

    f = opt.build()
    assert len(f) == 5
    assert_almost_equal(f.count(), ref.count(), decimal=2)
    assert_almost_equal(f.sum(), ref.sum(), decimal=2)


def test_case_a():
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
    assert len(f._segments) == 4
    assert_segments_are_sane(f._segments)

    assert f._segments[0].b == p25
    assert f._segments[1].a == p25
    assert f._segments[1].b == p50
    assert f._segments[2].a == p50
    assert f._segments[2].b == p75
    assert f._segments[3].a == p75
    assert_almost_equal(f.count(), N, decimal=2)
    assert_almost_equal(f.sum(), N * m, decimal=2)


def test_case_b():
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
    assert_segments_are_sane(f._segments)

    assert f._segments[0].b == 7_624
    assert f._segments[11].a == 59_444
    assert f._segments[0].a >= 0
    assert f._segments[0].a < 7_624
    assert f._segments[11].b > 59_444
    assert_almost_equal(f.count(), 4_777_805, decimal=2)
    assert_almost_equal(f.sum(), 4_777_805 * 31_781, decimal=2)


def test_case_c():
    opt = PiecewiseDensityOptimizer(256_083, 22_398, 0.0)
    opt.add(20_128, 0.25)
    opt.add(27_894, 0.75)
    opt.add(7_535, 0.10)
    opt.add(17_195, 0.20)
    opt.add(22_436, 0.30)
    opt.add(24_898, 0.40)
    opt.add(25_832, 0.50)
    opt.add(26_683, 0.60)
    opt.add(27_498, 0.70)
    opt.add(28_284, 0.80)
    opt.add(29_076, 0.90)
    o = opt.optimize()
    assert o.success

    f = opt.build()
    assert len(f._segments) == 13
    assert_segments_are_sane(f._segments)

    assert f._segments[1].b == 7_535
    assert f._segments[12].a == 29_076
    assert f._segments[0].a >= 0
    assert f._segments[1].a < 7_535
    assert f._segments[12].b > 29_076
    assert_almost_equal(f.count(), 256_083, decimal=2)
    assert_almost_equal(f.sum(), 256_083 * 22_398, decimal=2)
