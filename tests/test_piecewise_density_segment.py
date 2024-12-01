from verolysis.piecewise_density import Segment


def test_segment_overlap():
    assert Segment(1, 5, 4).overlap(None, None) == Segment(1, 5, 4)

    assert Segment(1, 5, 4).overlap(None, 0) == None
    assert Segment(1, 5, 4).overlap(None, 1) == None
    assert Segment(1, 5, 4).overlap(None, 2) == Segment(1, 2, 1)
    assert Segment(1, 5, 4).overlap(None, 5) == Segment(1, 5, 4)
    assert Segment(1, 5, 4).overlap(None, 6) == Segment(1, 5, 4)

    assert Segment(1, 5, 4).overlap(0, None) == Segment(1, 5, 4)
    assert Segment(1, 5, 4).overlap(1, None) == Segment(1, 5, 4)
    assert Segment(1, 5, 4).overlap(2, None) == Segment(2, 5, 3)
    assert Segment(1, 5, 4).overlap(5, None) == None
    assert Segment(1, 5, 4).overlap(6, None) == None

    assert Segment(1, 5, 4).overlap(0, 1) == None
    assert Segment(1, 5, 4).overlap(0, 2) == Segment(1, 2, 1)
    assert Segment(1, 5, 4).overlap(0, 5) == Segment(1, 5, 4)
    assert Segment(1, 5, 4).overlap(0, 6) == Segment(1, 5, 4)

    assert Segment(1, 5, 4).overlap(1, 2) == Segment(1, 2, 1)
    assert Segment(1, 5, 4).overlap(1, 5) == Segment(1, 5, 4)
    assert Segment(1, 5, 4).overlap(1, 7) == Segment(1, 5, 4)

    assert Segment(1, 5, 4).overlap(2, 3) == Segment(2, 3, 1)
    assert Segment(1, 5, 4).overlap(2, 5) == Segment(2, 5, 3)
    assert Segment(1, 5, 4).overlap(2, 8) == Segment(2, 5, 3)

    assert Segment(1, 5, 4).overlap(5, 6) == None

    assert Segment(1, 1, 10).overlap(None, None) == Segment(1, 1, 10)

    assert Segment(1, 1, 10).overlap(None, 0) == None
    assert Segment(1, 1, 10).overlap(None, 1) == None
    assert Segment(1, 1, 10).overlap(None, 2) == Segment(1, 1, 10)

    assert Segment(1, 1, 10).overlap(0, None) == Segment(1, 1, 10)
    assert Segment(1, 1, 10).overlap(1, None) == Segment(1, 1, 10)
    assert Segment(1, 1, 10).overlap(2, None) == None

    assert Segment(1, 1, 10).overlap(0, 1) == None
    assert Segment(1, 1, 10).overlap(1, 1) == Segment(1, 1, 10)
    assert Segment(1, 1, 10).overlap(1, 2) == Segment(1, 1, 10)


def test_merge():
    S = Segment

    assert S.merge(S(1, 3, 2), S(4, 7, 6)) == ([S(1, 3, 2)], S(4, 7, 6))
    assert S.merge(S(1, 4, 3), S(4, 7, 6)) == ([S(1, 4, 3)], S(4, 7, 6))
    assert S.merge(S(1, 5, 4), S(4, 7, 6)) == ([S(1, 4, 3), S(4, 5, 3)], S(5, 7, 4))
    assert S.merge(S(1, 7, 6), S(4, 7, 6)) == ([S(1, 4, 3), S(4, 7, 9)], None)
    assert S.merge(S(1, 8, 7), S(4, 7, 6)) == ([S(1, 4, 3), S(4, 7, 9)], S(7, 8, 1))

    assert S.merge(S(4, 5, 1), S(4, 7, 6)) == ([S(4, 5, 3)], S(5, 7, 4))
    assert S.merge(S(4, 7, 3), S(4, 7, 6)) == ([S(4, 7, 9)], None)
    assert S.merge(S(4, 8, 4), S(4, 7, 6)) == ([S(4, 7, 9)], S(7, 8, 1))

    assert S.merge(S(5, 6, 1), S(4, 7, 6)) == ([S(4, 5, 2), S(5, 6, 3)], S(6, 7, 2))
    assert S.merge(S(5, 7, 2), S(4, 7, 6)) == ([S(4, 5, 2), S(5, 7, 6)], None)
    assert S.merge(S(5, 8, 3), S(4, 7, 6)) == ([S(4, 5, 2), S(5, 7, 6)], S(7, 8, 1))

    assert S.merge(S(7, 8, 1), S(4, 7, 6)) == ([S(4, 7, 6)], S(7, 8, 1))

    assert S.merge(S(8, 9, 1), S(4, 7, 6)) == ([S(4, 7, 6)], S(8, 9, 1))

    assert S.merge(S(1, 3, 2), S(1, 1, 4)) == ([S(1, 1, 4)], S(1, 3, 2))
    assert S.merge(S(1, 3, 2), S(3, 3, 4)) == ([S(1, 3, 2)], S(3, 3, 4))
    assert S.merge(S(1, 3, 2), S(2, 2, 4)) == ([S(1, 2, 1), S(2, 2, 4)], S(2, 3, 1))

    assert S.merge(S(1, 1, 2), S(2, 2, 4)) == ([S(1, 1, 2)], S(2, 2, 4))
    assert S.merge(S(2, 2, 2), S(1, 1, 4)) == ([S(1, 1, 4)], S(2, 2, 2))
    assert S.merge(S(1, 1, 2), S(1, 1, 4)) == ([S(1, 1, 6)], None)
