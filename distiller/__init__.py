"""
@author supermantx
@date 2024/4/23 14:42
"""

from distiller.KD import KD
from distiller.KD_TLoss import KD_TLoss
from distiller.fcd import Fcd
from .MLKD import MLKD
from .TLoss import TLoss


def get_distiller(cfg, student, teacher):
    if cfg.DISTILLER.CLASS == "KD":
        return KD(student, teacher, cfg)
    elif cfg.DISTILLER.CLASS == "KD_TLoss":
        return KD_TLoss(student, teacher, cfg)
    elif cfg.DISTILLER.CLASS == "fcd":
        return Fcd(student, teacher, cfg)
    elif cfg.DISTILLER.CLASS == "MLKD":
        return MLKD(student, teacher, cfg)
    else:
        raise ValueError(f"Unknown distiller: {cfg.DISTILLER.NAME}")
