from contextlib import contextmanager

from pynblast import ResamplingArbor


def get_nodes(resampler):
    return {tup[0] for tup in resampler.skeleton()}


def assert_nodes(resampler, contains=None, not_contains=None):
    if not contains:
        contains = set()
    if not not_contains:
        not_contains = set()

    nodes = get_nodes(resampler)

    assert nodes.issuperset(contains)
    assert nodes.isdisjoint(not_contains)


def test_construction(skeleton):
    ResamplingArbor(skeleton)


def test_prune_at(resampler):
    prune_at = [23329984]
    with assert_removes_nodes(resampler, not_contains=prune_at):
        resampler.prune_at(prune_at)
    assert_nodes(resampler, not_contains=prune_at)


def test_prune_branches_containing(resampler):
    prune_containing = [6208611, 6208792, 3537598]
    with assert_removes_nodes(resampler, not_contains=prune_containing):
        resampler.prune_branches_containing(prune_containing)


@contextmanager
def assert_removes_nodes(resampler, contains=None, not_contains=None):
    before = get_nodes(resampler)
    yield
    after = get_nodes(resampler)
    assert after.issubset(before)
    assert len(after) < len(before)
    if contains:
        assert after.issuperset(contains)
    if not_contains:
        assert after.isdisjoint(not_contains)


def test_prune_below_strahler(resampler):
    with assert_removes_nodes(resampler):
        resampler.prune_below_strahler(2)


def test_prune_beyond_branches(resampler):
    with assert_removes_nodes(resampler):
        resampler.prune_beyond_branches(3)


def test_prune_twigs(resampler):
    with assert_removes_nodes(resampler):
        resampler.prune_twigs(5000)


def test_prune_beyond_distance(resampler):
    with assert_removes_nodes(resampler):
        resampler.prune_beyond_distance(10_000)


def test_cable_length(resampler):
    # todo: actually check length
    assert resampler.cable_length()


def test_copy(resampler):
    cp = resampler.copy()
    assert resampler.skeleton() == cp.skeleton()


def test_points(resampler):
    resampler.points()


def test_points_resample(resampler):
    before = resampler.points()
    after = resampler.points(1000)
    assert len(after) < len(before)


def test_skeleton(resampler):
    resampler.skeleton()
