from verolysis.piecewise_density import Segment


def test_segment_overlap():
    assert Segment(1, 5, 1).overlap(None, None) == Segment(1, 5, 1)

    assert Segment(1, 5, 1).overlap(None, 0) == None
    assert Segment(1, 5, 1).overlap(None, 1) == None
    assert Segment(1, 5, 1).overlap(None, 2) == Segment(1, 2, 1)
    assert Segment(1, 5, 1).overlap(None, 5) == Segment(1, 5, 1)
    assert Segment(1, 5, 1).overlap(None, 6) == Segment(1, 5, 1)

    assert Segment(1, 5, 1).overlap(0, None) == Segment(1, 5, 1)
    assert Segment(1, 5, 1).overlap(1, None) == Segment(1, 5, 1)
    assert Segment(1, 5, 1).overlap(2, None) == Segment(2, 5, 1)
    assert Segment(1, 5, 1).overlap(5, None) == None
    assert Segment(1, 5, 1).overlap(6, None) == None

    assert Segment(1, 5, 1).overlap(0, 1) == None
    assert Segment(1, 5, 1).overlap(0, 2) == Segment(1, 2, 1)
    assert Segment(1, 5, 1).overlap(0, 5) == Segment(1, 5, 1)
    assert Segment(1, 5, 1).overlap(0, 6) == Segment(1, 5, 1)

    assert Segment(1, 5, 1).overlap(1, 2) == Segment(1, 2, 1)
    assert Segment(1, 5, 1).overlap(1, 5) == Segment(1, 5, 1)
    assert Segment(1, 5, 1).overlap(1, 7) == Segment(1, 5, 1)

    assert Segment(1, 5, 1).overlap(2, 3) == Segment(2, 3, 1)
    assert Segment(1, 5, 1).overlap(2, 5) == Segment(2, 5, 1)
    assert Segment(1, 5, 1).overlap(2, 8) == Segment(2, 5, 1)

    assert Segment(1, 5, 1).overlap(5, 6) == None

    assert Segment(1, 5, 1).overlap(6, 8) == None


def test_merge():
    S = Segment

    assert S.merge(S(1, 3, 1), S(4, 7, 2)) == ([S(1, 3, 1)], S(4, 7, 2))
    assert S.merge(S(1, 4, 1), S(4, 7, 2)) == ([S(1, 4, 1)], S(4, 7, 2))
    assert S.merge(S(1, 5, 1), S(4, 7, 2)) == ([S(1, 4, 1), S(4, 5, 3)], S(5, 7, 2))
    assert S.merge(S(1, 7, 1), S(4, 7, 2)) == ([S(1, 4, 1), S(4, 7, 3)], None)
    assert S.merge(S(1, 8, 1), S(4, 7, 2)) == ([S(1, 4, 1), S(4, 7, 3)], S(7, 8, 1))

    assert S.merge(S(4, 5, 1), S(4, 7, 2)) == ([S(4, 5, 3)], S(5, 7, 2))
    assert S.merge(S(4, 7, 1), S(4, 7, 2)) == ([S(4, 7, 3)], None)
    assert S.merge(S(4, 8, 1), S(4, 7, 2)) == ([S(4, 7, 3)], S(7, 8, 1))

    assert S.merge(S(5, 6, 1), S(4, 7, 2)) == ([S(4, 5, 2), S(5, 6, 3)], S(6, 7, 2))
    assert S.merge(S(5, 7, 1), S(4, 7, 2)) == ([S(4, 5, 2), S(5, 7, 3)], None)
    assert S.merge(S(5, 8, 1), S(4, 7, 2)) == ([S(4, 5, 2), S(5, 7, 3)], S(7, 8, 1))

    assert S.merge(S(7, 8, 1), S(4, 7, 2)) == ([S(4, 7, 2)], S(7, 8, 1))

    assert S.merge(S(8, 9, 1), S(4, 7, 2)) == ([S(4, 7, 2)], S(8, 9, 1))
