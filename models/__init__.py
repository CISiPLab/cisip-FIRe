# network_names = {}
#
#
# def register_network(name):
#     def f(network):
#         if name in network_names:
#             print(f'Error when registering name for class {network.__class__.__name__}')
#             raise Exception(f'Network name "{name}" is registered with class {network_names[name].__class__.__name__}')
#         network_names[name] = network
#         return network
#
#     return f
#
#
# from models.arch.adsh import *
# from models.arch.barlowtwins import *
# from models.arch.bihalf import *
# from models.arch.bitquery import *
# from models.arch.ce import *
# from models.arch.cibhash import *
# from models.arch.csq import *
# from models.arch.delg import *
# from models.arch.dino import *
# from models.arch.dpn import *
# from models.arch.gh import *
# from models.arch.mae import *
# from models.arch.moco import *
# from models.arch.nare import *
# from models.arch.orthohash import *
# from models.arch.pairwise import *
# from models.arch.sdp import *
# from models.arch.semicon import *
# from models.arch.semicon_ce import *
# from models.arch.ssdh import *
# from models.arch.tbh import *
# from models.arch.unsup_orthohash import *
# from models.arch.vae import *
# from models.arch.vicreg import *
