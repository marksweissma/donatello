import inspect
import pandas as pd
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


def fallback(*defaults):
    """
    Keyword arguments of attribute to fallback to of object
    """
    @decorator
    def _wrapper(wrapped, instance, args, kwargs):
        spec = inspect.getargspec(wrapped)
        for default in defaults:
            index = spec.args.index(default)
            if index > len(args):
                kwargs[default] = kwargs.get(default, getattr(instance, default, None))

        result = wrapped(*args, **kwargs)
        return result
    return _wrapper


# fix this, should check - default - execute
@decorator
def name(wrapped, instance, args, kwargs):
        spec = inspect.getargspec(wrapped)
        _name = getattr(instance, '_name', instance.__class__.__name__)
        instance._name = _name
        result = wrapped(*args, **kwargs)
        return result


@decorator
def to_kwargs(wrapped, instance, args, kwargs):
    spec = inspect.getargspec(wrapped)
    offset = int(bool(instance))
    update = {argSpec: arg for argSpec, arg in zip(spec.args[offset:], args)}
    kwargs.update(update)
    return wrapped(**kwargs)
