'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
from torchvision import torch

from models.elic import TestModel as ELICModel
from models.gg18 import ScaleHyperpriorSTE
# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)

class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

elic_paths = [
    'bins/ELIC_0008_ft_3980_Plateau.pth.tar',
    'bins/ELIC_0016_ft_3980_Plateau.pth.tar',
    'bins/ELIC_0032_ft_3980_Plateau.pth.tar',
    'bins/ELIC_0150_ft_3980_Plateau.pth.tar',
    'bins/ELIC_0450_ft_3980_Plateau.pth.tar',
]

@register_operator(name='elic')
class CodecOperator(NonLinearOperator):
    def __init__(self, q, device):
        self.elic = ELICModel()
        self.elic.load_state_dict(torch.load(elic_paths[q-1]))
        self.elic = self.elic.cuda()
        self.elic.eval()
        print("load elic: {}".format(elic_paths[q-1]))
        
    def forward(self, data, **kwargs):
        enc_out = self.elic((data + 1.0) / 2.0, "enc", False)
        dec_out = self.elic(enc_out["y_hat"], "dec", False)
        return (dec_out["x_bar"] * 2.0) - 1.0

gg18_paths = [
    'bins/bmshj2018-hyperprior-1-7eb97409.pth.tar',
    'bins/bmshj2018-hyperprior-2-93677231.pth.tar',
    'bins/bmshj2018-hyperprior-3-6d87be32.pth.tar',
    'bins/bmshj2018-hyperprior-4-de1b779c.pth.tar',
    'bins/bmshj2018-hyperprior-5-f8b614e1.pth.tar',
    'bins/bmshj2018-hyperprior-6-1ab9c41e.pth.tar',
    'bins/bmshj2018-hyperprior-7-3804dcbd.pth.tar',
    'bins/bmshj2018-hyperprior-8-a583f0cf.pth.tar',
]

Ns, Ms = [128,128,128,128,128,192,192,192], [192,192,192,192,192,320,320,320]

@register_operator(name='gg18')
class CodecOperator(NonLinearOperator):
    def __init__(self, q, device):
        self.codec = ScaleHyperpriorSTE(Ns[q-1], Ms[q-1])
        self.codec.load_state_dict_gg18(torch.load(gg18_paths[q - 1]))
        self.codec = self.codec.cuda()
        self.codec.eval()
        print("load gg18 q: {}".format(q))
        
    def forward(self, data, **kwargs):
        out = self.codec((data + 1.0) / 2.0)
        return (out["x_bar"] * 2.0) - 1.0
