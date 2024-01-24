from pynblast import NblastArena
from pynblast.util import Format

import pytest


@pytest.mark.benchmark(group="construct")
def test_empty_arena(benchmark, score_mat):
    benchmark(NblastArena, score_mat, k=5)


@pytest.mark.benchmark(group="construct")
def test_clone_arena(benchmark, arena: NblastArena):
    benchmark(arena.copy, True)


@pytest.mark.benchmark(group="construct")
def test_add_points(benchmark, points, arena):
    pt = points[0][1].to_numpy()

    def fn():
        ar2 = arena.copy(True)
        ar2.add_points(pt)

    benchmark(fn)


@pytest.mark.benchmark(group="construct")
def test_add_points_tangents_alphas(benchmark, points, arena: NblastArena):
    pt = points[0][1].to_numpy()
    empty = arena.copy(True)
    idx = arena.add_points(pt)
    t = arena.tangents(idx)
    a = arena.alphas(idx)

    def fn():
        ar = empty.copy(True)
        ar.add_points_tangents_alphas(pt, t, a)

    benchmark(fn)


@pytest.mark.benchmark(group="construct")
def test_add_serialized(benchmark, points, arena: NblastArena):
    pt = points[0][1].to_numpy()
    empty = arena.copy(True)
    idx = arena.add_points(pt)
    b = arena._impl.serialize_neuron(idx, Format.CBOR)

    def fn():
        ar = empty.copy(True)
        # ar.add_serialized_neuron(BytesIO(b), Format.CBOR)
        ar._impl.add_serialized_neuron(b, Format.CBOR)

    benchmark(fn)


@pytest.mark.benchmark(group="all_v_all")
def test_all_v_all_serial(benchmark, arena_names_factory):
    arena: NblastArena
    (arena, _) = arena_names_factory(False, None)

    benchmark(arena.all_v_all)


@pytest.mark.benchmark(group="smatbuilder")
def test_smatbuilder_bench(benchmark, smatbuilder):
    benchmark(smatbuilder.build)
