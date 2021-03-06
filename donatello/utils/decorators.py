import inspect
import pandas as pd
from copy import deepcopy
from uuid import uuid4
from wrapt import decorator

if hasattr(inspect, 'signature'):
    funcsigs = inspect
else:
    import funcsigs


from donatello.utils.helpers import now_string, nvl, find_value, replace_value


@decorator
def init_time(wrapped, instance, args, kwargs):
    """
    Add _initTime attribute to object, format prescribed by
    **strFormat** kwarg
    """

    initTime = find_value(wrapped, args, kwargs, 'initTime')
    timeFormat = find_value(wrapped, args, kwargs, 'timeFormat')

    if not initTime:
        payload = {'strFormat': timeFormat} if timeFormat else {}
        initTime = now_string(**payload)

    instance.timeFormat = timeFormat
    instance.initTime = initTime
    result = wrapped(*args, **kwargs)
    return result


def as_series(arr, index, name):
    pass


def as_df(arr, index, columns):
    pass


@decorator
def to_pandas(wrapped, instance, args, kwargs):

    dataset = find_value(wrapped, args, kwargs, 'dataset')
    X = find_value(wrapped, args, kwargs, 'X')
    y = find_value(wrapped, args, kwargs, 'y')

    if 'name' in kwargs:
        name = kwargs.pop('name', '')
    elif 'columns' in kwargs:
        columns = kwargs.pop('columns', '')

    result = wrapped(*args, **kwargs)

    if isinstance(result, pd.np.ndarray):
        dims = result.shape
        if len(dims) == 1 or dims[1] == 1:
            result = as_series(result, index, name)
        else:
            result = as_df(result, index, columns)
    return result


# todo @to_pandas(shape)
@decorator
def pandas_series(wrapped, instance, args, kwargs):
    """
    Enforce output as :py:class:`pandas.Series`
    """
    X = args[0]
    index = kwargs.pop('index', 'index')
    name = kwargs.pop('name', '')
    yhat = wrapped(*args, **kwargs)

    name = nvl(name, instance.name if instance is not None else '')
    index = X[index] if index in X else getattr(X, 'index', None)
    result = pd.Series(yhat, index=index, name=name)
    return result


@decorator
def pandas_df(wrapped, instance, args, kwargs):
    """
    Enforce output as :py:class:`pandas.DataFrame`
    """
    X = args[0]
    index = kwargs.pop('index', 'index')
    columns = kwargs.pop('columns', [])
    columns = nvl(columns, X.columns)
    index = X[index] if index in X else X.index

    _df = wrapped(*args, **kwargs)
    result = pd.DataFrame(_df, index=index, columns=columns)
    return result


def coelesce(**defaults):
    """
    Key value pairs to enforce null value logic
    """
    @decorator
    def _wrapper(wrapped, instance, args, kwargs):
        for key, default in defaults.items():
            value = find_value(wrapped, args, kwargs, key)
            args, kwargs = replace_value(wrapped, args, kwargs, key, nvl(value, deepcopy(default)))

        result = wrapped(*args, **kwargs)
        return result
    return _wrapper


def fallback(*defaults, **replacements):
    """
    Keyword arguments of attribute to fallback to of object
    """
    @decorator
    def _wrapper(wrapped, instance, args, kwargs):
        sig = funcsigs.signature(wrapped)
        for default in defaults:
            index = list(sig.parameters).index(default)
            if index >= len(args) and default not in kwargs and hasattr(instance, default):
                kwargs[default] = getattr(instance, default)

        for key, replacement in replacements.items():
            index = list(sig.parameters).index(key)
            if index >= len(args) and key not in kwargs and hasattr(instance, replacement):
                kwargs[key] = getattr(instance, replacement)

        result = wrapped(*args, **kwargs)
        return result
    return _wrapper


# @coelesce(existing={})
# def mem_cache(existing=None):
    # """
    # memoization cache

    # Args:
        # existing: __getitem__ sliceable cache, defaults to :py:class:`dict`
    # """
    # attr = '_'.join(['cache', str(uuid4())[:4]])

    # @decorator
    # def memoize(wrapped, instance, args, kwargs):
        # if not hasattr(wrapped, attr):
            # setattr(wrapped, attr, existing)
        # cache = getattr(wrapped, attr)
        # spec = inspect.getargspec(wrapped)
        # key = tuple((str(i), str(find_value(wrapped, args, kwargs, i))) for i in spec.args)
        # if key not in cache:
            # cache[key] = wrapped(*args, **kwargs)
        # return cache[key]

    # return memoize
