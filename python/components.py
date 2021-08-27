from abc import ABC, abstractclassmethod
import numpy as np
from numpy import pi
import pandas as pd
from primitives import Wave


# class TimeFrame:
#     @classmethod
#     def get_time(
#         cls,
#         fs: float,
#         size: int = None,
#         t_max: float = None,
#         t_0: float = 0.0,
#     ) -> Time:
#         """
#         Create a time array at a given sample frequency, fs, given either size
#         of array or max time (exclusive)

#         Parameters
#         ----------
#         fs : float
#             sample frequency, Hz
#         size : int, default = None
#             size of array
#         t_max : float, default = None
#             maximum time to generate time array (exlusive)
#         t_0 : float, default=0.0
#             time offset to adjust start time of sequence
#         """
#         if size is not None:
#             return cls._size_to_time(fs=fs, size=size, t_0=t_0)
#         elif t_max is not None:
#             return cls._tmax_to_time(fs=fs, t_max=t_max, t_0=t_0)
#         else:
#             raise AttributeError("`size` or `t_max` not specified")

#     @staticmethod
#     def _size_to_time(fs: float, size: int, t_0: float=0.0) -> Time:
#         """see `get_time`"""
#         n_array = np.arange(size)
#         t = (n_array / fs) + t_0
#         return Time(time=t, fs=fs)

#     @staticmethod
#     def _tmax_to_time(fs: float, t_max: float, t_0: float=0.0) -> Time:
#         """see `get_time`"""
#         t = np.arange(t_0, t_max, step=1/fs)
#         return Time(time=t, fs=fs)


class Interpolator(ABC):
    """
    Class to upsample a wave
    __Note: this may be improved with pandas resample allowing the use of
            ffill, bfill, and fillna methods__
    """
    # tg: TimeFrame = TimeFrame()

    @abstractclassmethod
    def upsample():
        """Upsample Sequence"""


class WaveInterpolator(Interpolator):
    @classmethod
    def upsample(cls, wave: Wave, n) -> Wave:
        new_iq = np.repeat(wave.iq, n)
        new_fs = wave.fs * n
        return Wave(iq=new_iq, fs=new_fs, fc=wave.fc, t0=wave.t0)

# class SymbolInterpolator(Interpolator):
#     @classmethod
#     def upsample(cls, symbols: SoftSymbol, n) -> SoftSymbol:
#         new_iq = np.repeat(symbols.iq, n)
#         new_fs = symbols.fs * n
#         new_sps = symbols.sps * n
#         return SoftSymbol(
#             iq=new_iq, fs=new_fs, fchip=symbols.fchip,
#             sps=new_sps, t0=symbols.t0
#         )


class SignalGenerator:
    @classmethod
    def gen_cos_wave(
        cls, n: int, fs: float, fc: float, phi: float = 0.0, t0: float = 0.0
    ) -> Wave:
        """
        Generate a Carrier wave with frequency, fc, and a sampling rate of fs

        $y=cos(\omega t + \phi)$, where $\omega = 2 \pi fc$, $\phi$ is phase, 
        $f_c$ is the carrier frequency and t is time, sampled at the sampling
        frequency, fs.
        """
        t = cls._clock(n, fs)
        x = 2 * pi * fc * t + phi
        return Wave(np.cos(x) + 1j * np.sin(x), fs=fs, fc=fc, t0=t0)

    @staticmethod
    def _clock(n: int, fs:float, t0: float = 0.0):
        """Generate a clock"""
        return np.arange(n) / fs + t0

