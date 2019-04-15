import pytest
import pandas as pd
from donatello.utils import helpers


@pytest.fixture
def lst():
    return []


@pytest.fixture
def e():
    return pd.np.array([])


@pytest.fixture
def a():
    return pd.np.array(['a'])


@pytest.fixture
def ab():
    return pd.np.array(['a', 'b'])


class O(object):
    a = 1

    def __init__(self, b=2):
        self.b = b

    @property
    def o(self):
        return O()

    def v(self, on=True):
        return 'w' if on else 'u'


def cb(value, flip=False):
    return value + 1 if not flip else -1 * (value + 1)


@pytest.fixture
def o():
    return O()


def test_nvl(lst, e, a, ab):
    assert lst == helpers.nvl(None, lst)
    assert None is helpers.nvl(lst, None)

    assert a == helpers.nvl(None, a)
    assert a == helpers.nvl(a, None)

    assert (ab == helpers.nvl([], ab)).all()
    assert (ab == helpers.nvl(ab, [])).all()


def test_has(o):
    assert helpers.has(o, 'a', ())
    assert helpers.has({'a': None}, 'a', slicers=(dict,))
    assert not helpers.has(o, 'c', ())
    assert not helpers.has({'a': None}, 'c', slicers=(dict,))


def test_get(o):
    assert helpers.get(o, 'a', (), 'ignore', None) == 1
    assert helpers.get({'a': 1}, 'a', (dict,), 'ignore', None) == 1

    assert helpers.get(o, 'c', (), 'ignore', None) is None
    assert helpers.get({'a': None}, 'c', (dict,), 'ignore', None) is None

    with pytest.raises(AttributeError):
        helpers.get(o, 'c', (), 'raise', None)

    with pytest.raises(KeyError):
        helpers.get({'a': None}, 'c', (dict,), 'raise', None)


def test_access(o):
    assert helpers.access(o, []) is o
    assert helpers.access(o, ['']) is o
    assert helpers.access(o, '') is o
    assert helpers.access(o, ['a']) == 1

    assert helpers.access(o, method='v') == 'w'
    assert helpers.access(o, method='v', methodArgs=(False,)) == 'u'
    assert helpers.access(o, method='v', methodKwargs=dict(on=False)) == 'u'

    assert helpers.access(o, ['a'], cb=cb) == 2
    assert helpers.access(o, ['a'], cb=cb, cbArgs=(True,)) == -2
    assert helpers.access(o, ['a'], cb=cb, cbKwargs=dict(flip=True)) == -2
