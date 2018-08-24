from sklearn.externals import joblib as dill
# import dill
from donatello.utils.helpers import get_nested_attribute


class Hook(object):
    def __init__(self, reader=dill.load, writer=dill.dump):
        self.reader = reader
        self.writer = writer

    def write(self, obj=None, attr="", root='.', extension='pkl', writeKwargs={'protocol': 2}):
        obj = get_nested_attribute(obj, attr)
        name = ".".join([getattr(obj, 'name', obj.__class__.__name__), extension])
        localPath = "/".join([root, name])
        self.writer(obj, localPath, **writeKwargs)
        # s = self.writer(obj, **writeKwargs)
        # with open(localPath, 'w') as f:
            # f.write(s)

    def read(self, localPath):
        obj = self.reader(localPath)
        return obj
