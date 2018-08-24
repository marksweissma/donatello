import pandas as pd


def now_string(strFormat="%Y_%m_%d_%H_%M"):
    return pd.datetime.now().strftime(strFormat)


def nvl(*args):
    for arg in args:
        if arg:
            return arg


def has_nested_attribute(obj, attrPath, separator='_'):
    """
    Get nested attributes via string name

    :param object obj: Object to traverse.
    :param str attrPath: Path to attribute location split by separator, use empty string "" to return obj
    :param str separator: separator for nesting levels in string representation
    """

    if attrPath == "":
        return obj is not None
    elif hasattr(obj, attrPath):
        return True
    else:
        nextLevel, tail = attrPath.split(separator, 1)
        return get_nested_attribute(getattr(obj, nextLevel, None), tail, separator)


def get_nested_attribute(obj, attrPath, separator='_'):
    """
    Get nested attributes via string name

    :param object obj: Object to traverse.
    :param str attrPath: Path to attribute location split by separator, use empty string "" to return obj
    :param str separator: separator for nesting levels in string representation
    """

    if attrPath == "":
        return obj
    try:
        return getattr(obj, attrPath)
    except AttributeError:
        nextLevel, tail = attrPath.split(separator, 1)
        return get_nested_attribute(getattr(obj, nextLevel), tail, separator)


def default_values(dicts, defaults):
    for name, kwargs in dicts.iteritems():
        for kwarg, value in defaults.iteritems():
            if kwarg not in kwargs:
                kwargs[kwarg] = value
    return dicts
