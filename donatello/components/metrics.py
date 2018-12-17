from donatello.utils.base import Dobject
from donatello.utils.decorators import coelesce


class Metric(Dobject):
    @coelesce(columns=['score'], kwargs={})
    def __init__(self, name='', columns=None, kwargs=None):
        super(Metric, self).__init__()
        self.name = name
        self.columns = columns
        self.kwargs = kwargs


    def parse_scored(self, scored):
        pass

    def calculate(self, scored=None, estimators=None):
        pass

    def __call__(self):
        pass
