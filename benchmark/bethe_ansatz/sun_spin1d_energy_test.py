from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from benchmark.bethe_ansatz import sun_spin1d_energy


class SUNSpin1DEnergyTest(tf.test.TestCase):

    def test_compute_ground_state_energy(self):
        spin_nums = [5, 5]
        self.assertAlmostEqual(
            sun_spin1d_energy.compute_ground_state_energy(spin_nums),
            -4.03089270898, 8)
        spin_nums = [3, 3, 3]
        self.assertAlmostEqual(
            sun_spin1d_energy.compute_ground_state_energy(spin_nums),
            -6.57943575839, 8)
        spin_nums = [2, 2, 2, 2]
        self.assertAlmostEqual(
            sun_spin1d_energy.compute_ground_state_energy(spin_nums),
            -6.92100974042, 8)
        spin_nums = [2, 2, 2, 2, 2]
        self.assertAlmostEqual(
            sun_spin1d_energy.compute_ground_state_energy(spin_nums),
            -9.11783645395, 8)
        spin_nums = [30, 30]
        self.assertAlmostEqual(
            sun_spin1d_energy.compute_ground_state_energy(spin_nums),
            -23.2051463312, 8)
        spin_nums = [20, 20, 20]
        self.assertAlmostEqual(
            sun_spin1d_energy.compute_ground_state_energy(spin_nums),
            -42.2293649617, 8)
        spin_nums = [15, 15, 15, 15]
        self.assertAlmostEqual(
            sun_spin1d_energy.compute_ground_state_energy(spin_nums),
            -49.5483540601, 8)
        spin_nums = [12, 12, 12, 12, 12]
        self.assertAlmostEqual(
            sun_spin1d_energy.compute_ground_state_energy(spin_nums),
            -53.1277468675, 8)


if __name__ == "__main__":
    tf.test.main()
