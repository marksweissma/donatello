from donatello.examples import intent_simple


def test_data_load():
    s = intent_simple.load_sculpture()
    n = len(s.dataset)
    k = len(s.dataset.data.columns)

    xk = len(s.dataset.designData.columns)
    xn = len(s.dataset.designData)

    assert n == xn
    assert k == (xk + 1)

    xkt = len(s.dataset.designTrain.columns)
    xnt = len(s.dataset.designTrain)

    d2 = s.dataset.subset('train')
    x2kt = len(d2.designData.columns)
    x2nt = len(d2.designData)

    assert x2kt == xkt
    assert x2nt == xnt
