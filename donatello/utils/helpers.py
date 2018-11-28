import pandas as pd


def has(obj, attr, slicers):
    condition = attr in obj if isinstance(obj, slicers) else hasattr(obj, attr)
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
           slicers=(dict, list, tuple, pd.np.ndarray, pd.Series, pd.DataFrame),
           default=None, errors='raise'):
    """
    Access information from nested object

    :param object obj: object to access from
    :param list attrPath: sequence of traversal
    :param str method: (optional) method to call at end of path
    :param tuple methodArgs: positional args for method 
    :param tuple methodKwargs: keyword args for method 
    :param str cb: (optional) cb to call at end of path
    :param tuple cbArgs: positional args for cb 
    :param tuple cbKwargs: keyword args for cb 
    :param tuple slicers: object types to use ``__getitem__`` slice rather than getattr
    :param obj default: option to return default (if not rasiing errors)
    :param str errors: option to raise errors ('raise') or ignore ('ignore')

    :return: value of given prescription
    """

    if not attrPath:
        obj = obj if not method else getattr(obj, method)(*methodArgs, **methodKwargs)
        value = obj if not cb else cb(obj, *cbArgs, **cbKwargs)

    else:
        head, tail = attrPath[0], attrPath[1:]

        obj = get(obj, head, slicers, errors, default)

        value = access(obj, attrPath=tail,
                       method=method, methodArgs=methodArgs, methodKwargs=methodKwargs,
                       cb=cb, cbArgs=cbArgs, cbKwargs=cbKwargs,
                       slicers=slicers, errors=errors, default=default)

    return value


def now_string(strFormat="%Y_%m_%d_%H_%M"):
    """
    Pandas formatted string from time

    :param str strFormat: format for time
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

    :param object obj: Object to traverse.
    :param str attrPath: Path to attribute location split by separator,\
            use empty string "" to return obj
    :param str separator: separator for nesting levels in string representation
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

    :param object obj: Object to traverse.
    :param str attrPath: Path to attribute location split by separator,\
            use empty string "" to return obj
    :param str separator: separator for nesting levels in string representation
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
