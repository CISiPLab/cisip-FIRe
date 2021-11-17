network_names = {}


def register_network(names):
    def f(network):
        nonlocal names
        if not isinstance(names, list):
            names = [names]

        for name in names:
            if name in network_names:
                print(f'Error when registering name for class {network.__class__.__name__}')
                raise Exception(
                    f'Network name "{name}" is registered with class {network_names[name].__class__.__name__}')
            network_names[name] = network

        return network

    return f


from models.architectures.arch_base import *
from models.architectures.arch_ce import *
from models.architectures.arch_cibhash import *
from models.architectures.arch_orthohash import *
from models.architectures.arch_dpn import *
from models.architectures.arch_gh import *
from models.architectures.arch_jmlh import *
from models.architectures.arch_norm_unsupervised import *
from models.architectures.arch_single import *
from models.architectures.arch_tbh import *
