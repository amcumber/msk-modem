import numpy as np
import unittest
from unittest import TestCase

from modulators import MSKModulator, QPSKModulator
# from components import TimeFrame


class TestModulator(TestCase):
    def test_main(self):
        """Test main"""
        # Test Modulate
        ...

    # def test_random_bit_len(self) -> None:
    #     # Random Bits
    #     for l in range(10,20):
    #         # print(f"Testing RandomBit generation for lengh=={l}")
    #         random_bits = RandomBits.get_bits(l=l)
    #         assert random_bits.size == l
    #         assert np.unique(random_bits).size <= 2

    def test_qpsk_sps(self) -> None:
        #Test qpsk sps
        for sps in range(5):
            qmod = QPSKModulator(sps, fchip=10)
            # print(f"Testing SPS for {qmod}...")
            bitstream = np.array([0, 0, 0, 1, 1, 0, 1, 1, 0, 0])
            qpsk_symb = qmod.get_soft_symbol(bitstream)
            # print(bitstream)
            # print(qpsk_symb)
            expected = np.array([
                    -1 - 1j,
                    -1 + 1j,
                    1 - 1j,
                    1 + 1j,
                    -1 - 1j,
            ])
            expected = expected.repeat(qmod.sps)
            assert all(expected == qpsk_symb.iq)

    def test_msk_sps(self) -> None:
        #Test msk sps
        for sps in range(2, 10, 2):
            fchip = 2
            mod = MSKModulator(fchip * sps, fchip=fchip)
            # print(f"Testing SPS for {mod}...")
            bitstream = np.array([
                0, 0, 0, 1,
                1, 0, 1, 1,
                0, 0, 1, 0,
                0, 1, 1, 0,
                1, 0, 0, 1,
                0, 1,
            ])
            msk_symb = mod.square_wave(bitstream)
            expected = np.array([
                -1 - 1j, 1 - 1j, 1 - 1j, 1 - 1j,
                 1 - 1j, -1 - 1j, -1 - 1j, -1 - 1j,
                -1 - 1j, -1 - 1j, -1 + 1j, -1 + 1j,
                -1 + 1j, -1 + 1j, -1 + 1j, 1 + 1j,
                 1 - 1j, 1 - 1j, 1 - 1j, -1 - 1j,
                -1 + 1j, -1 + 1j,
            ])
            sps = mod.sps
            expected = expected.repeat(sps//2)
            assert all(expected == msk_symb.iq)

    def test_image_sps(self):
        return # Comment out to visualize - no test
        import matplotlib.pyplot as plt
        bitstream = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0])
        sps = 100
        mod = MSKModulator(sps, fchip=1000)
        msk_symb = mod._square_wave(bitstream)
        t = np.arange(len(bitstream), step=2/sps)
        plt.figure()
        plt.plot(t, msk_symb.i)
        plt.plot(t, msk_symb.q)
        plt.show()

    def test_modulate(self) -> None:
        import matplotlib.pyplot as plt
        fchip = 100
        fcarrier = fchip * 10
        fs = fcarrier * 20 
        mod = MSKModulator(fs=fs, fchip=fchip)
        bitstream = np.array([
            1, 1,
            0, 0,
            1, 1,
            0, 0,
            0, 0,
            0, 1,
            0, 1,
            1, 0,
            1, 0,
        ])
        s = mod.modulate(bitstream, fc=fcarrier)
        tx = s.i + s.q
        an = mod.square_wave(bitstream)
        sn = np.roll(s.iq, -1)
        diff = s.iq - sn
        # assert False
        plt.figure()
        plt.plot(s.i+s.q)
        plt.plot(an.i)
        plt.plot(an.q)
        plt.title('MSK Modulation')
        plt.figure()
        plt.plot(s.t, diff)
        plt.title('First Moment of MSK Modulation')
        fft = np.fft.fft(s.iq)
        plt.plot(fft)
        plt.title('FFT')
        plt.show()

    @classmethod
    def main(cls):
        ...
    


if __name__ == "__main__":
    # TestModulators().test_main()
    # TestModulators().main()
    unittest.main()
