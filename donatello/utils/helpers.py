import os
import inspect

import pandas as pd

from sklearn.externals import joblib


def now_string(strFormat="%Y_%m_%d_%H_%M"):
    """
    Pandas formatted string from time

    Args:
        strFormat (str): format for time
    """
    return pd.datetime.now().strftime(strFormat)


def nvl(*args):
    """
    SQL like coelesce / redshift NVL, returns first non Falsey arg
    """
    for arg in args:
        try:
            if arg:
                return arg
        except ValueError:
            if arg is not None:
                return arg
    return args[-1]


def has_nested_attribute(obj, attrPath, separator='_'):
    """
    Check whether nested attribute exists via string name

    Args:
        obj (object): Object to traverse.
        attrPath (str): Path to attribute location split by separator,\
            use empty string "" to return obj
        separator (str): separator for nesting levels in string representation
    """
    if hasattr(obj, attrPath):
        return True
    elif not attrPath:
        return obj is not None
    else:
        nextLevel, tail = attrPath.split(separator, 1)
        return _get_nested_attribute(getattr(obj, nextLevel, None),
                                     tail, separator)


def _get_nested_attribute(obj, attrPath, separator='_'):
    if attrPath == "":
        return obj
    elif hasattr(obj, attrPath):
        return getattr(obj, attrPath)
    else:
        nextLevel, tail = attrPath.split(separator, 1)
        return _get_nested_attribute(getattr(obj, nextLevel, None),
                                     tail, separator)


def get_nested_attribute(obj, attrPath, separator='_'):
    """
    Get nested attribute via string name. passing empty string returns obj

    Args:
        obj (object): Object to traverse.
        attrPath (str): Path to attribute location split by separator,\
            use empty string "" to return obj
        separator (str): separator for nesting levels in string representation
    """
    if attrPath and not has_nested_attribute(obj, attrPath, separator):
        raise AttributeError('{obj} does not have {attrPath}'.format(
                             obj=obj, attrPath=attrPath)
                             )
    else:
        return _get_nested_attribute(obj, attrPath, separator)


# dispatch on shape make registers :/
def reformat_aggs(d, idx=0, sortValues=None, indexName=None, filterNulls=.95):
    information = [pd.Series(value[value.columns[idx]], name=key) for key, value in d.items()]
    df = pd.concat(information, axis=1, sort=True)
    df = df.sort_values(sortValues) if sortValues else df
    if indexName:
        df.index.name = indexName
    if filterNulls:
        rates = df.isnull().mean()
        df = df.drop([column for column, rate in zip(df, rates) if rate > filterNulls], axis=1)
    return df


def has(obj, attr, slicers):
    condition = attr in obj if (slicers and isinstance(obj, slicers)) else hasattr(obj, attr)
    return condition


def _get(obj, attr, slicers):
    value = obj[attr] if isinstance(obj, slicers) else getattr(obj, attr)
    return value


def get(obj, attr, slicers, errors, default):
    """
    Get from object
    """
    condition = (errors == 'raise') or has(obj, attr, slicers)
    value = _get(obj, attr, slicers) if condition else default
    return value


def access(obj=None, attrPath=None,
           method=None, methodArgs=None, methodKwargs=None,
           cb=None, cbArgs=None, cbKwargs=None,
           slicers=(dict, list, tuple, pd.np.ndarray, pd.Series, pd.DataFrame, pd.Panel),
           default=None, errors='raise'):
    """
    Access information from nested object

    Args:
        obj (object): object to access from
        attrPath (list): sequence of traversal
        method (str): (optional) method to call at end of path
        methodArgs (tuple): positional args for method
        methodKwargs (tuple): keyword args for method
        cb (str): (optional) cb to call at end of path
        cbArgs (tuple): positional args for cb
        cbKwargs (tuple): keyword args for cb
        slicers (tuple): object types to use ``__getitem__`` slice rather than getattr
        default (obj): option to return default (if not rasiing errors)
        errors (str): option to raise errors ('raise') or ignore ('ignore')

    Returns:
        obj: value of given prescription
    """

    if not attrPath or not attrPath[0]:
        obj = obj if not method else getattr(obj, method)(*nvl(methodArgs, ()), **nvl(methodKwargs, {}))
        value = obj if not cb else cb(obj, *nvl(cbArgs, ()), **nvl(cbKwargs, {}))

    else:
        head, attrPath = attrPath[0], attrPath[1:]
        obj = get(obj, head, slicers, errors, default)

        value = access(obj, attrPath=attrPath,
                       method=method, methodArgs=methodArgs, methodKwargs=methodKwargs,
                       cb=cb, cbArgs=cbArgs, cbKwargs=cbKwargs,
                       slicers=slicers, errors=errors, default=default)

    return value


def find_value(func, args, kwargs, accessKey, how='name'):
    """
    Find a value from a function signature
    """
    spec = inspect.getargspec(func)
    _args = spec.args[1:] if inspect.ismethod(func) else spec.args

    try:
        index = _args.index(accessKey) if how == 'name' else accessKey
        offset = len(_args) - len(nvl(spec.defaults, []))
        default = spec.defaults[index - offset] if index >= offset else None
        value = kwargs.get(accessKey, default) if index >= len(args) else args[index]
    except ValueError:
        value = kwargs.get(accessKey, None)
    return value


def replace_value(func, args, kwargs, accessKey, accessValue):
    """
    Replace a value from a function signature
    """
    spec = inspect.getargspec(func)
    _args = spec.args[1:] if inspect.ismethod(func) else spec.args
    index = _args.index(accessKey)
    if index >= len(args):
        kwargs[accessKey] = accessValue
    else:
        args = list(args)
        args[index] = accessValue
        args = tuple(args)
    return args, kwargs


def persist(obj=None, attr="", root='.', name='', extension='pkl', *writeArgs, **writeKwargs):
    """
    Write an object (or attribute) to persist.
    """
    obj = access(obj, [attr])
    name = name if name else ".".join([getattr(obj, 'name', obj.__class__.__name__), extension])
    local = os.path.join(root, name)
    joblib.dump(obj, local, *writeArgs, **writeKwargs)


def view_sk_metric(bunch):
    """
    Unnest a sklearn metric (or other single value only returning metric)

    Args:
        bunch (bunch): dict of aggregated scores

    Returns:
        pandas.DataFrame: flattened view
    """
    df = pd.DataFrame({'score': {key: value.values[0][0]
                                 for key, value in
                                 bunch.items()}
                       }
                      )
    return df
