from abc import ABC, abstractclassmethod, abstractmethod
from dataclasses import Field, dataclass
import numpy as np
from numpy import ndarray, pi
   

@dataclass(frozen=True)
class Time:
    time: ndarray
    fs: float
    # t_max: float = Field(init=False, repr=False)
    def max(self):
        return self.time.max() + 1/self.fs

    @property
    def t_max(self):
        return self.time.max()

    @property
    def t_0(self):
        return self.time.min()


class IQ(ABC):
    @property
    def i(self):
        return np.real(self.iq)

    @property
    def q(self):
        return np.imag(self.iq)

@dataclass(frozen=True)
class SoftSymbol(IQ):
    iq: ndarray
    fchip: float
    sps: int


@dataclass(frozen=True)
class Wave(IQ):
    iq: ndarray
    time: Time
    fc: float = None

    @property
    def fs(self):
        return self.time.fs
