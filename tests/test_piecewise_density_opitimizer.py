from verolysis import PiecewiseDensityOptimizer


def test_optimize():
    N = 279_305
    m = -17_467
    p25 = -16_951
    p50 = -8_141
    p75 = -3_630

    opt = PiecewiseDensityOptimizer(N, m)
    opt.add(p25, 0.25)
    opt.add(p50, 0.50)
    opt.add(p75, 0.75)
    opt.optimize()

    f = opt.build()
    print(f._segments)
    assert len(f._segments) == 4
    assert f._segments[0].b == p25
    assert f._segments[1].a == p25
    assert f._segments[1].b == p50
    assert f._segments[2].a == p50
    assert f._segments[2].b == p75
    assert f._segments[3].a == p75
    assert f.count() == N
    assert f.sum() == N * m
