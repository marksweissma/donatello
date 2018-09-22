import pandas as pd


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
