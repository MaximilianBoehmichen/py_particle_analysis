from typing import Literal

from pypana.analysis.lognormal.mixturelognormalfit import MixtureLognormalFit
from pypana.analysis.lognormal.modelognormalfit import ModeLogNormalFit

type LogNormalFit = ModeLogNormalFit | MixtureLognormalFit

LogNormalFitType = Literal["mode", "mixture"]
