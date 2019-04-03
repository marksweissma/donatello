from sklearn.model_selection import (KFold,
                                     StratifiedKFold,
                                     GroupShuffleSplit,
                                     TimeSeriesSplit)

from donatello.utils.base import Dobject, RANDOM_SEED
from donatello.utils.helpers import access
from donatello.utils.decorators import fallback, init_time, coelesce




