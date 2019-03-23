import inspect
import pandas as pd
from uuid import uuid4
from wrapt import decorator

from donatello.utils.helpers import now_string, nvl, find_value, replace_value


@decorator
def init_time(wrapped, instance, args, kwargs):
    """
    Add _initTime attribute to object, format prescribed by
    **strFormat** kwarg
    """
    signature = kwargs.get('timeFormat', None)
    payload = {'strFormat': signature} if signature else {}
    result = wrapped(*args, **kwargs)
    instance._initTime = now_string(**payload)
    return result


@decorator
def pandas_series(wrapped, instance, args, kwargs):
    """
    Enforce output as :py:class:`pandas.Series`
    """
    X = args[0]
    index = kwargs.pop('index', 'index')
    name = kwargs.pop('name', '')
    yhat = wrapped(*args, **kwargs)

    name = nvl(name, instance.name)
    index = X[index] if index in X else X.index
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
            args, kwargs = replace_value(wrapped, args, kwargs, key, nvl(value, default))

        result = wrapped(*args, **kwargs)
        return result
    return _wrapper


def fallback(*defaults, **replacements):
    """
    Keyword arguments of attribute to fallback to of object
    """
    @decorator
    def _wrapper(wrapped, instance, args, kwargs):
        spec = inspect.getargspec(wrapped)
        n = len(spec.args) - 1
        for default in defaults:
            index = spec.args.index(default)
            if index > len(args) and default not in kwargs:
                kwargs[default] = getattr(instance, default, spec.defaults[n - index])

        for key, replacement in replacements.items():
            index = spec.args.index(key)
            if index > len(args) and key not in kwargs:
                kwargs[key] = getattr(instance, replacement, spec.defaults[n - index])

        result = wrapped(*args, **kwargs)
        return result
    return _wrapper


# fix this, should check - default - execute
@decorator
def name(wrapped, instance, args, kwargs):
        result = wrapped(*args, **kwargs)
        _name = getattr(instance, '_name', instance.__class__.__name__)
        instance._name = _name
        return result


@coelesce(existing={})
def mem_cache(existing=None):
    """
    memoization cache

    Args:
        existing: __getitem__ sliceable cache, defaults to :py:class:`dict`
    """
    attr = '_'.join(['cache', str(uuid4())[:4]])

    @decorator
    def memoize(wrapped, instance, args, kwargs):
        if not hasattr(wrapped, attr):
            setattr(wrapped, attr, existing)
        cache = getattr(wrapped, attr)
        spec = inspect.getargspec(wrapped)
        key = tuple((str(i), str(find_value(wrapped, args, kwargs, i))) for i in spec.args)
        if key not in cache:
            cache[key] = wrapped(*args, **kwargs)
        return cache[key]

    return memoize


@decorator
def to_kwargs(wrapped, instance, args, kwargs):
    spec = inspect.getargspec(wrapped)
    offset = int(bool(instance))
    update = {argSpec: arg for argSpec, arg in zip(spec.args[offset:], args)}
    kwargs.update(update)
    return wrapped(**kwargs)
