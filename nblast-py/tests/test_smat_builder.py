def test_smatbuilder(smatbuilder):
    smat = smatbuilder.build()
    assert len(smat.dist_thresholds) == 7
    assert len(smat.dot_thresholds) == 5
    assert smat.values.shape == (6, 4)
