from typing import Literal

from pypana.analysis.lognormal.mixturelognormalfit import MixtureLognormalFit
from pypana.analysis.lognormal.modelognormalfit import ModeLognormalFit

type LogNormalFit = ModeLognormalFit | MixtureLognormalFit

LogNormalFitType = Literal["mode", "mixture"]
