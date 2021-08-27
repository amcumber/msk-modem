from abc import ABC, abstractclassmethod, abstractmethod
from dataclasses import field, dataclass
import numpy as np
from numpy import bitwise_and, ndarray

from primitives import Wave
from components import SignalGenerator
   

# Classes
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
    fs: float
    fchip : float
    sg: SignalGenerator = field(
        init=False, repr=False, default=SignalGenerator()
    )

    @property
    def sps(self):
        return self.fs // self.fchip

    @abstractmethod
    def modulate(self, soft_symbols: ndarray)-> Wave:
        """Modulate softsymbols into iq waveform"""

    @staticmethod
    def get_bipolar(bitstream: ndarray) -> ndarray:
        """Convert Binary Stream [0,1] to bipolar stream [-1, 1]"""
        return (bitstream * 2) - 1

    @staticmethod
    def extract_odd(bits: ndarray) -> ndarray:
        odd_bits = range(0, len(bits), 2)
        return bits[odd_bits]

    @staticmethod
    def extract_even(bits: ndarray) -> ndarray:
        odd_bits = range(1, len(bits), 2)
        return bits[odd_bits]

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
    def get_soft_symbol(self, bitstream: ndarray) -> Wave:
        nez = self.get_bipolar(bitstream)
        odd, even = nez.reshape(2, -1)
        s_i, s_q = odd.repeat(self.sps), even.repeat(self.sps)
        return Wave(s_i + 1j * s_q, fs=self.fs, fc=self.fchip)

    def modulate(self):
        ...


@dataclass
class MSKModulator(Modulator):
    """ Minimum Shift Keying Modulator

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

    @classmethod
    def _bits_to_symbol(cls, bitstream) -> ndarray:
        """Convert bitstream to msk IQ hard symbols"""
        nez = cls.get_bipolar(bitstream)
        odd, even = nez.reshape(-1, 2).T
        a_i, a_q = (cls._phase_adjust_symbol(a) for a in (odd, even))
        return np.array([a_i, a_q])

    @staticmethod
    def _phase_adjust_symbol(stream):
        a_x = np.zeros_like(stream)
        prev = 0
        # -1 -1 -1 1 1 1 -> -1 1 -1 -1 1 -1
        # [-1 -1 -1 1 1 -1 1 -1] -> [-cos(0) -cos(pi) -cos(0) +cos(pi)]
        # for i in [-1 -1 -1 1 1 -1 1 -1]:
        #   if i == prev:
        #     flip i (ie -1 -1 -> -1 1)
        #   elise i != prev and wrong phase:
        #     flip i (ie -1 1 -> -1 -1 )
        #     flip i (ie -1 1 -> -1 * cos(0), -1 * cos(pi) )
        #   else  i != prev and right phase:
        #     do nothing (i e -1 -1 1 -> -1 1 1)
        #     do nothing (i e -1 -1 1 -> -1 * cos(0), -1 * cos(pi), 1 * cos(0))
        phi = 0
        for idx, a in enumerate(stream):
            if idx == 0:
                phi = 0
            elif a == prev:
                phi += np.pi
                bk = 1
            else:
                phi -= np.pi
            prev = a
            a_x[idx] = a * np.cos(phi)
        return a_x

    def square_wave(self, bitstream: ndarray) -> Wave:
        """
        Generate soft symbols at sps rates in a 2D array

        Parameters
        ----------
        bitstream : ndarray[bool | int] 1,n
            array of bits representing data

        Return
        ------
        Wave[complex128] : 1,n
            IQ soft symbols for msk
        """
        half_sps = int(self.sps//2)
        odd, even = self._bits_to_symbol(bitstream)
        s_i = odd.repeat(self.sps)[half_sps:]
        s_i = np.append(s_i, [s_i[-1]] * half_sps)
        # This last step is needed to extend to one additional half period to
        # transmit the last symbol
        s_q = even.repeat(self.sps)
        iq = s_i + 1j * s_q
        return Wave(iq=iq, fs=self.fs, fc=self.fchip)

    def baseband_modulation(self, an: Wave) -> Wave:
        """ Return baseband modulation in IQ provided modulator properties
        Properties
        ----------
        an : Wave
            square wave of corrected msk bipolar values (ie -1 -1 -> -1 +1)
        """
        msk_shapes = self._get_msk_shape_wave(n=len(an))
        return an.comp_mul(msk_shapes)

    def _get_msk_shape_wave(
        self, n: int, phi:float = 0.0, t0: float = 0.0
    ) -> Wave:
        # MSK wave - should be 1/2 Period to half period === chiprate (why div 4)
        return self.sg.gen_cos_wave(
            n, fs=self.fs, fc=self.fchip, phi=phi, t0=t0
        )

    def get_carrier_wave(
        self, n: int, fc: float, phi:float = 0.0, t0: float = 0.0
    ) -> Wave:
        """Bridge to component - generating a carrier_wave"""
        return self.sg.gen_cos_wave(n=n, fs=self.fs, fc=fc, phi=phi, t0=t0)

    def corrected_carrier(
        self, square_wave: Wave, carrier: Wave
    ) -> Wave:
        """Correct a carrier wave with square wave pulse of bitstream"""
        return carrier.comp_mul(square_wave)

    @staticmethod
    def _get_cos(x: ndarray, fs: float, fc: float) -> Wave:
        return Wave(np.cos(x) + 1j* np.cos(x), fs, fc)

    def modulate(self, bitstream: ndarray, fc: float) -> Wave:
        """ Modulate a soft symbol stream into msk wavform centered around a 
        carrier frequency, fc, to generate a waveform, $s(t)$: 
        from: https://en.wikipedia.org/wiki/Minimum-shift_keying

        $s(t) = a_I(t) cos(\frac{\pi t}{2 T}) cos(2 \pi f_c t) - 
                a_Q(t) sin(\frac{\pi t}{2 T}) sin(2 \pi f_c t)$

        or from https://www.dsprelated.com/showarticle/1016.php

        $s(t) = cos(2\pi f_c t) * cos(2 \pi \frac{R_b}{4} t) cos(\Theta_n) - 
                sin(2\pi f_c t) * sin(2 \pi \frac{R_b}{4} t) a_n cos(\Theta_n)$

        where,
        time, $t = i/f_c$, is equal to the sample index, $i$, divided by
        the carrier frequency, $f_c$ (stored in the instance parameter `fc`), 
        and the period,
        $T = N_{sps}/f_c$, is equal to the samples per second, $N_{sps}$,
        stored in the instance parameter `sps`.

        
        Parameters
        ----------
        a_n: Wave[complex128], 1xn
            complex iq stream of soft symbols
        fc : float
            carrier frequency

        Returns
        -------
        Wave[complex128], 1xn
            IQ waveform mixed with carrier frequency

        Returns
        -------
        ndarray: 1xn
            Complex wave in chip frequency
        """
        # Get Square pulses of IQ and Phase
        an = self.square_wave(bitstream)
        baseband = self.baseband_modulation(an)
        n = len(baseband)
        carrier = self.get_carrier_wave(n, fc=fc)
        correct_carrier = self.corrected_carrier(an, carrier)
        return baseband.comp_mul(correct_carrier)
    
if __name__ == "__main__":
    ...
