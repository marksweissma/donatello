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
def split_data(wrapped, instance, args, kwargs):
    """
    Split data contents into train and test sets
    """
    data = kwargs.pop('data', None)

    if data and data.hasContents and instance.splitter:
        data.unpack_splits(next(instance.splitter.fit_split(data)))
    else:
        data.designData = data.contents
    result = wrapped(data=data, *args, **kwargs)
    return result


@decorator
def prepare_design(wrapped, instance, args, kwargs):
    """
    Apply `combiner` to data to create final design (if applicable)
    """
    data = kwargs.pop('data', None)
    if getattr(instance, 'combiner', None):
        data = instance.combiner.fit_transform(data)
    result = wrapped(data=data, *args, **kwargs)
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
    result =  pd.Series(yhat, index=index, name=name)
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
        for default in defaults:
            kwargs[default] = kwargs.get(default, getattr(instance, default, None))
        result = wrapped(*args, **kwargs)
        return result
    return _wrapper


def update(**defaults):
    """
    update first arg payload with attr from obj
    """
    @decorator
    def _wrapper(wrapped, instance, args, kwargs):
        args = list(args)
        for default, index in defaults.items():
            args[index] = args[index] if args[index] else {}
            args[index][default] = args[index][default] if default in args[index] else getattr(instance, default, None)
        result = wrapped(*tuple(args), **kwargs)
        return result
    return _wrapper
