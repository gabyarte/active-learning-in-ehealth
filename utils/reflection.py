import inspect
from os.path import dirname, basename, isfile, join
import os, importlib
import glob
from managers.corpus import CorpusManager

def load_modules_from_directory(directory):
    modules_name = glob.glob(join(dirname(directory), "*.py"))

    modules = []
    for path in modules_name:
        modules.append(importlib.import_module(f'{directory[:-1]}.{basename(path)[:-3]}'))

    return modules

def get_init_args(module, parent_class=None):
    classes = get_classes(module, parent_class)

    model_args = {}
    for name, model in classes.items():
        full_args = inspect.getfullargspec(model.__init__)

        args = full_args.args
        defaults = full_args.defaults if full_args.defaults is not None else []

        n = len(defaults)
        model_args[name] = [(arg, None) for arg in args[1:-n]] + list(zip(args[-n:], defaults))
    
    return model_args

def get_classes(module, parent_class=None):
    modules = module if isinstance(module, list) else [module]

    classes = {}
    for mod in modules:
        for i in inspect.getmembers(mod, inspect.isclass):
            if parent_class is not None:
                if issubclass(i[1], parent_class) and not (i[1] is parent_class):
                    classes[i[0]] = i[1]
            else:
                classes[i[0]] = i[1]
    return classes