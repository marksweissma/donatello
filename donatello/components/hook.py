from sklearn.externals import joblib
from donatello.utils.helpers import get_nested_attribute


class Local(object):
    """
    Object to provide disk interface

    :param func reader: default function to read files
    :param func writer: default function to write files
    """
    def __init__(self, reader=joblib.load, writer=joblib.dump):
        self.reader = reader
        self.writer = writer

    def write(self, obj=None, attr="", root='.', extension='pkl', writeKwargs={'protocol': 2}):
        obj = get_nested_attribute(obj, attr)
        name = ".".join([getattr(obj, 'name', obj.__class__.__name__), extension])
        localPath = "/".join([root, name])
        self.writer(obj, localPath, **writeKwargs)

    def read(self, localPath):
        obj = self.reader(localPath)
        return obj
