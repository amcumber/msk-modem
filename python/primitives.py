from abc import ABC
from dataclasses import field, dataclass
import numpy as np
from numpy import ndarray
   

@dataclass(frozen=True)
class IQ(ABC):
    iq: ndarray
    fs: float
    fc: float = None
    t0: float = field(default=0.0, repr=False)

    @property
    def t(self):
        return np.arange(len(self)) / self.fs + self.t0

    def _validate_other(self, other) -> None:
        if len(self.iq) != len(self.iq) or self.fs != other.fs:
            err = "Can not add Waves with incompatible properties"
            raise AttributeError(err)
        return None

    @property
    def i(self):
        return np.real(self.iq)

    @property
    def q(self):
        return np.imag(self.iq)

    def __len__(self):
        return len(self.iq)

    def __add__(self, other):
        self._validate_other(other)
        new_iq = self.iq + other.iq
        new_fc = self.fc
        if self.fc != other.fc:
            new_fc = np.nan
        return Wave(new_iq, fs=self.fs, fc=new_fc)

    def __sub__(self, other):
        self._validate_other(other)
        new_iq = self.iq - other.iq
        new_fc = self.fc
        if self.fc != other.fc:
            new_fc = np.nan
        return Wave(new_iq, fs=self.fs, fc=new_fc)

    def __mul__(self, other):
        self._validate_other(other)
        new_iq = self.iq * other.iq
        new_fc = self.fc
        if self.fc != other.fc:
            new_fc = np.nan
        return Wave(new_iq, fs=self.fs, fc=new_fc)

    def __mul__(self, other):
        self._validate_other(other)
        new_iq = self.iq * other.iq
        new_fc = self.fc
        if self.fc != other.fc:
            new_fc = np.nan
        return Wave(new_iq, fs=self.fs, fc=new_fc)

    def __div__(self, other):
        self._validate_other(other)
        new_iq = self.iq / other.iq
        new_fc = self.fc
        if self.fc != other.fc:
            new_fc = np.nan
        return Wave(new_iq, fs=self.fs, fc=new_fc)

    def comp_mul(self, other):
        """Component Multiply"""
        self._validate_other(other)
        new_i = self.i * other.i
        new_q = self.q * other.q
        new_iq = new_i + 1j * new_q
        new_fc = self.fc
        if self.fc != other.fc:
            new_fc = np.nan
        return Wave(new_iq, fs=self.fs, fc=new_fc)


# @dataclass(frozen=True)
# class SoftSymbol(IQ):
#     sps: int = None

#     @property
#     def fchip(self):
#         return self.fc


@dataclass(frozen=True)
class Wave(IQ):
    """Wave Class holding IQ data"""

class RandomBits:
    @staticmethod
    def get_bits(l):
        return np.random.randint(0, 2, l)
