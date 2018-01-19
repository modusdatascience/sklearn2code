import imp
from six import exec_

def exec_module(name, code):
    module = imp.new_module(name)
    exec_(code, module.__dict__)
    return module

