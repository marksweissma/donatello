import os
from sklearn.externals import joblib
from donatello.utils.helpers import access


class Local(object):
    """
    Object to provide disk interface

    Args:
        reader (func): default function to read files
        writer (func): default function to write files
    """
    def __init__(self, reader=joblib.load, writer=joblib.dump):
        self.reader = reader
        self.writer = writer

    def write(self, obj=None, attr="", root='.', extension='pkl', *writeArgs, **writeKwargs):
        obj = access(obj, [attr])
        name = ".".join([getattr(obj, 'name', obj.__class__.__name__), extension])
        localPath = os.path.join(root, name)
        self.writer(obj, localPath, *writeArgs, **writeKwargs)

    def read(self, localPath, *args, **kwargs):
        obj = self.reader(localPath, *args, **kwargs)
        return obj
