import inspect
import pandas as pd
from wrapt import decorator

from donatello.utils.helpers import now_string, nvl


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
def split_dataset(wrapped, instance, args, kwargs):
    """
    Split dataset data into train and test sets
    """
    dataset = kwargs.pop('dataset', None)

    if dataset and dataset.hasData and instance.splitter:
        dataset.unpack_splits(next(instance.splitter.fit_split(dataset)))
    else:
        dataset.designData = dataset.data
    result = wrapped(dataset=dataset, *args, **kwargs)
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
        spec = inspect.getargspec(wrapped)
        for key, default in defaults.items():
            index = spec.args.index(default)
            if index > len(args):
                kwargs[key] = kwargs.get(key, default)
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


@decorator
def name(wrapped, instance, args, kwargs):
        _name = getattr(instance, '_name', instance.__class__.__name__)
        instance._name = _name
        result = wrapped(*args, **kwargs)
        return result
