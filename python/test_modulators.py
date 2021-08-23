import numpy as np
import unittest
from unittest import TestCase

from modulators import Modulator, RandomBits, MSKModulator, QPSKModulator


class TestModulator(TestCase):
    def test_main(self):
        """Test main"""
        # Test Modulate
        ...

    def test_random_bit_len(self) -> None:
        # Random Bits
        for l in range(10,20):
            # print(f"Testing RandomBit generation for lengh=={l}")
            random_bits = RandomBits.get_bits(l=l)
            assert random_bits.size == l
            assert np.unique(random_bits).size <= 2

    def test_qpsk_sps(self) -> None:
        #Test qpsk sps
        for sps in range(5):
            qmod = QPSKModulator(sps, fchip=10)
            # print(f"Testing SPS for {qmod}...")
            bitstream = np.array([0, 0, 0, 1, 1, 0, 1, 1, 0, 0])
            qpsk_symb = qmod.soft_symbol(bitstream)
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
            mod = MSKModulator(sps, fchip=1000)
            # print(f"Testing SPS for {mod}...")
            bitstream = np.array([ 0, 0, 0, 1, 1, 0, 1, 1, 0, 0 ])
            msk_symb = mod.soft_symbol(bitstream)
            expected = np.array([
                -1 - 1j, -1 - 1j, -1 + 1j,
                1 + 1j,  1 - 1j,  1 - 1j,
                1 + 1j, -1 + 1j, -1 - 1j,
            ])
            sps = mod.sps
            expected = expected.repeat(sps//2)
            assert all(expected == msk_symb.iq)
    
    def test_time_fs_n(self) -> None:
        def _test(fs, n):
            mod = Modulator
            # print(f"Testing time generation for {mod}")
            ni = np.arange(n)
            t = mod.get_time(fs=fs, size=n)
            expected = ni / fs
            assert all(expected == t.time)
        [_test(fs=fs, n=10) for fs in (1,10,100,1000,1e6)]
        [_test(fs=10, n=n) for n in (10, 50, 100)]

    def test_time_tmax(self) -> None:
        # Test Time
        for t_max in (1, 2, 5):
            mod = Modulator
            fs = 100
            expected = np.arange(t_max, step=1/fs)
            t = mod.get_time(fs=fs, t_max=t_max)
            assert all(expected == t.time)

    def test_time_t0(self) -> None:
        for t_0 in (10, 50, 100):
            mod = Modulator
            fs = 100
            t_max = t_0 + 1
            expected = np.arange(t_0, t_max, step=1/fs)
            t = mod.get_time(fs=fs, t_max=t_max, t_0=t_0)
            assert all(expected == t.time)

    def test_modulate(self) -> None:
        import matplotlib.pyplot as plt
        fchip = 10
        fcarrier = fchip * 10
        fs = fcarrier * 10 
        mod = MSKModulator(sps=2, fchip=fchip)
        bitstream = np.array([
            0, 0, 1, 0, 0, 1, 1, 1, 0, 0,
            0, 0, 0, 0, 1, 1, 1, 0, 1, 0,
            0, 1, 0, 1, 1, 1, 1, 1, 0, 0,
        ])
        iq = mod.soft_symbol(bitstream)
        s = mod.modulate(iq, fc=fcarrier, fs=fs)
        # print(s)
        # print(iq)
        plt.plot(s.time.time, s.iq)
        plt.show()

    @classmethod
    def main(cls):
        ...
    


if __name__ == "__main__":
    # TestModulators().test_main()
    # TestModulators().main()
    unittest.main()
