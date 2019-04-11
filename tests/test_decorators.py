import pytest
import pandas as pd
from pandas import testing as pdt

from donatello.utils import decorators


@pytest.fixture
def arr():
    return pd.np.array(range(3))


@pytest.fixture
def series():
    return pd.Series(range(3), name='a')


@pytest.fixture
def df():
    return pd.DataFrame(pd.np.arange(9).reshape(3, 3), columns=['a', 'b', 'c'], index=['d', 'e', 'f'])


def test_series(arr, series):

    @decorators.pandas_series
    def f(arr, name=''):
        return arr
    pdt.assert_series_equal(f(arr), series.rename(''))
    pdt.assert_series_equal(f(arr, name='a'), series)


def test_coelsece():
    value = []

    @decorators.coelesce(a=value)
    def f(a=None):
        return a
    a1 = f()
    a2 = f()
    a3 = f(1)
    assert a1 is not a2
    assert a1 == a2
    assert a1 == value
    assert a3 == 1


def test_fallback():

    class O(object):
        def __init__(self):
            self.a = 1
            self.b = 2

        @decorators.fallback('a', c='b')
        def f(self, a=None, c=None):
            return a, c

    o = O()

    a1, c1 = o.f()
    a2, c2 = o.f(a='a')
    a3, c3 = o.f(c='a')
    a4, c4 = o.f('a')

    assert a1 is o.a and c1 is o.b
    assert a2 == 'a' and c2 is o.b
    assert a3 is o.a and c3 == 'a'
    assert a4 == 'a' and c4 is o.b
