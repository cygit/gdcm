import importlib
from dataset.base_dataset import BaseDataset


def get_dataset(dataset):
    module_path = "dataset.%s" % dataset
    module = importlib.import_module(module_path)
    for name in dir(module):
        obj = getattr(module, name)
        try:
            if issubclass(obj, BaseDataset) and obj != BaseDataset:
                return obj
        except TypeError:  # If 'obj' is not a class
            pass
    raise ImportError("Cannot find a subclass of %s in %s" % (BaseDataset, module_path))
