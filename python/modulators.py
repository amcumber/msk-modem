from abc import ABC, abstractclassmethod, abstractmethod
from dataclasses import Field, dataclass
import numpy as np
from numpy import ndarray, pi

from primitives import Time, Wave, SoftSymbol
   

@dataclass
class Modulator(ABC):
    """
    Abstract Modulator Class used to modulate data into soft symbols and
    waveforms

    Parameters
    ----------
    sps : int
        number of samples per symbol
    f_chip : float
        chip frequency
    """
    sps: int
    fchip : float

    @abstractmethod
    def soft_symbol(self, bitstream: ndarray) -> SoftSymbol:
        """Modulate bitstream into soft symbols"""

    @abstractmethod
    def modulate(self, soft_symbols: ndarray)-> Wave:
        """Modulate softsymbols into iq waveform"""

    @staticmethod
    def get_nez(bitstream: ndarray) -> ndarray:
        return (bitstream * 2) - 1

    @staticmethod
    def extract_even(bits: ndarray) -> ndarray:
        even_bits = range(0, len(bits), 2)
        return bits[even_bits]

    @staticmethod
    def extract_odd(bits: ndarray) -> ndarray:
        odd_bits = range(1, len(bits), 2)
        return bits[odd_bits]

    @classmethod
    def get_time(
        cls,
        fs: float,
        size: int = None,
        t_max: float = None,
        t_0: float = 0.0,
    ) -> Time:
        """
        Create a time array at a given sample frequency, fs, given either size
        of array or max time (exclusive)

        Parameters
        ----------
        fs : float
            sample frequency, Hz
        size : int, default = None
            size of array
        t_max : float, default = None
            maximum time to generate time array (exlusive)
        t_0 : float, default=0.0
            time offset to adjust start time of sequence
        """
        if size is not None:
            return cls._size_to_time(fs=fs, size=size, t_0=t_0)
        elif t_max is not None:
            return cls._tmax_to_time(fs=fs, t_max=t_max, t_0=t_0)
        else:
            raise AttributeError("`size` or `t_max` not specified")

    def gen_soft_symbol(self, iq):
        return SoftSymbol(iq, fchip=self.fchip, sps=self.sps)

    @staticmethod
    def _size_to_time(fs: float, size: int, t_0: float=0.0) -> Time:
        """see `get_time`"""
        n_array = np.arange(size)
        t = (n_array / fs) + t_0
        return Time(time=t, fs=fs)

    @staticmethod
    def _tmax_to_time(fs: float, t_max: float, t_0: float=0.0) -> Time:
        """see `get_time`"""
        t = np.arange(t_0, t_max, step=1/fs)
        return Time(time=t, fs=fs)


    @staticmethod
    def get_cos_wave(fc: float, t: Time, phi: float = 0.0) -> Wave:
        """Generate cos wave with carrier frequency, fc, given time array, t"""
        return Wave(np.cos(2 * pi * fc * t.time + phi), t.time, fc=fc)

    @staticmethod
    def get_sin_wave(fc: float, t: Time, phi: float = 0.0) -> Wave:
        """Generate sin wave with carrier frequency, fc, given time array, t"""
        return Wave(np.sin(2 * pi * fc * t.time + phi), t.time, fc=fc)

    @property
    def fsps(self):
        return self.sps * self.fchip

    @property
    def chip_period(self):
        """Chip period, aka time of transmission of 1 symbol"""
        return 1 / self.fchip

# class validate_soft_symbol:
#     def __call__(self, mod: Modulator, func: callable):
#         def wrapper(iq, *args, **kwargs):
#             if iq.sps != self.sps or iq.fchip != self.fchip:
#                 err = (f"{iq.__class__.__name__} does not match modulator "
#                         "properties, check class metadata")
#                 raise AttributeError(err)
#             return func(iq, *args, **kwargs)
#         return wrapper




@dataclass
class QPSKModulator(Modulator):
    """
    Quaternary Phase Shift Keying Modulator

    Parameters
    ----------
    sps : int
        number of samples per symbol
    f_chip : float
        chip frequency, aka chip rate
    """
    def soft_symbol(self, bitstream: ndarray) -> SoftSymbol:
        nez = self.get_nez(bitstream)
        even, odd = self.extract_even(nez), self.extract_odd(nez)
        s_i, s_q = even.repeat(self.sps), odd.repeat(self.sps)
        return self.gen_soft_symbol(s_i + 1j * s_q)

    def modulate(self):
        ...


@dataclass
class MSKModulator(Modulator):
    """
    Minimum Shift Keying Modulator

    Parameters
    ----------
    sps : int, even
        number of samples per symbol - must be even to allow for division of
        frequency between odd and even pulses
    f_chip : float
        chip frequency, Hz, aka chip rate

    Derived Parameters
    ------------------
    Available upon instantiation
    fs : float
        sample frequency, Hz, equal to  fc * sps
    period : float
        carrier period, s, equal to 1/fc

    Definition
    ----------

    $s(t) = a_I(t) cos(\frac{\pi t}{2 T}) cos(2 \pi f_c t) - 
            a_Q(t) sin(\frac{\pi t}{2 T}) cos(2 \pi f_c t)$
    where,
    $a_I(t)$ and $a_Q(t)$ are squre pulses of duration 2T.
    $a_I(t)$ has pulse edges on $t = [-T, T, 3T,...]$ and
    $a_Q(t)$ has pulse edges on $t = [0, 2T, 4T,...]$. $f_c$ is the carrier
    frequency.

    this equation can be rewritten as:
    $s(t) = cos(2 \pi f_c t + b_k(t) \frac{\pi t}{2 T} + \phi_k)$
    where,
    $b_k = 1$ when $a_I(t) = a_Q(t)$ and
    $b_k = -1$ when $a_I(t) != a_Q(t)$ (opposite sign)
    and,
    $\phi_k = 0$ when $a_I(t) = 1$ and $\phi_k = \pi$ otherwise
    """
    def soft_symbol(self, bitstream: ndarray) -> SoftSymbol:
        """
        Generate soft symbols at sps rates in a 2D array

        return
        ------
        ndarray[complex128] : 1,n
            IQ soft symbols for msk
        """
        half_sps = self.sps//2
        nez = self.get_nez(bitstream)
        even, odd = self.extract_even(nez), self.extract_odd(nez)
        s_i = even.repeat(self.sps)[half_sps:]
        s_q = odd.repeat(self.sps)[:-half_sps]
        return self.gen_soft_symbol(s_i + 1j * s_q)


    def modulate(
        self,
        iq: SoftSymbol,
        fc: float,
        fs:float,
        phi: float = 0.0,
        t_0: float = 0.0
    ) -> Wave:
        """
        Modulate a soft symbol stream into msk wavform centered around a 
        carrier frequency, fc, to generate a waveform, $s(t)$: 

        $s(t) = a_I(t) cos(\frac{\pi t}{2 T}) cos(2 \pi f_c t) - 
                a_Q(t) sin(\frac{\pi t}{2 T}) sin(2 \pi f_c t)$

        where,
        time, $t = i/f_c$, is equal to the sample index, $i$, divided by
        the carrier frequency, $f_c$ (stored in the instance parameter `fc`), 
        and the period,
        $T = N_{sps}/f_c$, is equal to the samples per second, $N_{sps}$,
        stored in the instance parameter `sps`.

        
        Parameters
        ----------
        soft_symbol: ndarray(dtype=complex128), 1xn
            complex iq stream of soft symbols
        fc : float
            carrier frequency
        fs : float
            sample frequency for carrier waveform
        phi : float, default = 0.0
            phase offset for carrier frequency
        t_0 : float, default = 0.0
            time offset for carrier frequency

        Returns
        -------
        ndarray[complex128], 1xn
            IQ waveform mixed with carrier frequency

        Returns
        -------
        ndarray: 1xn
            Complex wave in chip frequency
        """
        n = iq.iq.size

        t_chip = self.get_time(fs=self.fsps, size=n)

        n_upsample = int(fs // self.fsps)
        a_i, a_q = np.repeat(iq.i, n_upsample), np.repeat(iq.q, n_upsample)

        # carrier time wrt carrier frequency
        t = self.get_time(fs=fs, t_max=t_chip.max(), t_0=t_0)

        cos_fc = self.get_cos_wave(fc, t, phi=phi).iq # extract iq only
        sin_fc = self.get_sin_wave(fc, t, phi=phi).iq # extract iq only

        # 2 pi fc t = pi t / 2 T 
        # 2 fc = Rb / 2
        # fc = Rb/4
        # cos(pi * t / 2T) = cos(2pi Rb/4 t)
        cos_i = self.get_cos_wave(self.fchip/4, t, phi=phi).iq
        sin_q = self.get_sin_wave(self.fchip/4, t, phi=phi).iq
        # FIXME
        # 1. Wave is not maintaining correct phase
        # 2. Wave is not adjusting frequency - apparently
        # 3. Wave is only real - how to add imag??
        s_i = a_i * cos_i * cos_fc
        s_q = a_q * sin_q * sin_fc
        return Wave(s_i + s_q, time=t, fc=fc)



class RandomBits:
    @staticmethod
    def get_bits(l):
        return np.random.randint(0, 2, l)
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fchip = 10
    fcarrier = fchip * 10
    fs = fcarrier * 20
    mod = MSKModulator(sps=2, fchip=fchip)
    bitstream = np.array([
        0, 0, 1, 0, 1, 1, 1, 0, 0, 0,
    ])
    iq = mod.soft_symbol(bitstream)
    s = mod.modulate(iq, fc=fcarrier, fs=fs)
    print(s)
    print(iq)
    plt.plot(s.iq)
    plt.show()
