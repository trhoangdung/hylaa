'''
This module contains some DAE benchmarks with the index from 1 to 3
Dung Tran Nov-2017

'''

import numpy as np


class index_2_daes(object):
        'some index-2 DAE benchmarks'

        def __init__(self):
            self.E = None
            self.A = None
            self.B = None
            self.C = None

        def two_interconnected_rotating_masses(self, J1, J2):
            'Two interconnected rotating masses, 4-dimensional DAE'
            # Benchmark is from the paper : A modeling and Filtering
            # Framework for Linear Differential-Algebraic Equations, CDC2003

            isinstance(J1, float)
            isinstance(J2, float)
            assert (J1 > 0) and (J2 > 0)

            self.E = np.array([[J1, 0, 0, 0], [0, J2, 0, 0],
                               [0, 0, 0, 0], [0, 0, 0, 0]])

            self.A = np.array([[0, 0, 1, 0], [0, 0, 0, 1],
                               [0, 0, -1, -1], [-1, 1, 0, 0]])

            self.B = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])

            self.C = np.transpose(self.B)

            return self.E, self.A, self.B, self.C

        def RL_network(self, R, L):
            'RL network, 3-dimensional DAE'

            # From the thesis: Index-aware Model Order Reduction Methods
            # Section 5.3

            isinstance(R, float)
            isinstance(L, float)
            assert (R > 0) and (L > 0)

            self.E = np.array([[0, 0, 0], [0, 0, 0], [0, 0, L]])
            self.A = np.array([[-R, R, 0], [R, -R, -1], [0, 1, 0]])
            self.B = np.array([[1], [0], [0]])
            self.C = np.array([1, 1, 1])

            return self.E, self.A, self.B, self.C


class index_3_daes(object):
    'some index-3 DAE benchmarks'

    def __init__(self):
        self.E = None
        self.A = None
        self.B = None
        self.C = None

    def car_pendulum(self, m1, m2, L):
        'This is 7-dimensional car pendulum benchmark'

        # This benchmark is from the paper:
        # A gernal mathematical framework for modeling, simulation and control
        # Automatisierungstecknik, 2009
        # This benchmark is presented in the thesis:
        # Index-aware Model Order Reduction Methods
        # Section 5.4

        isinstance(m1, float)
        isinstance(m2, float)
        isinstance(L, float)
        assert (m1 > 0) and (m2 > 0) and (L > 0)
        g = 9.81

        self.E = np.diag((1, 1, 1, m1, m2, m2, 0))
        self.A = np.array([[0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0],
                               [-m2 * g / L, m2 * g / L, 0, 0, 0, 0, 0],
                               [m2 * g / L, (-m2) * g / L, 0, 0, 0, 0, 0],
                               [0, 0, m2 * g / L, 0, 0, 0, 2 * L],
                               [0, 0, -2 * L, 0, 0, 0, 0]])

        self.B = np.array([[0], [0], [0], [1], [0], [0], [0]])
        self.C = np.array([[0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]])

        return self.E, self.A, self.B, self.C

    def generator(self, J, L, R1, R2, k):
        'This is 9-dimensional generator benchmark'

        # This benchmark is presented in the thesis:
        # Index-aware Model Order Reduction Methods
        # Example 3.2.2
        # Originally from the thesis: Parameter estimation in Linear Descriptor Systems
        # Linkoping University, Sweden, 2004

        isinstance(J, float)
        isinstance(L, float)
        isinstance(R1, float)
        isinstance(R2, float)
        isinstance(k, float)
        assert (J > 0) and (L > 0) and (R1 > 0) and (R2 > 0) and (k > 0)

        self.E = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0],
                               [0, 0, J, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, L, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0]])

        self.A = np.array([[0, 0, 0, -1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0],
                               [0, -1, 0, 0, k, 0, 0, 0, 0],
                               [0, 0, k, 0, 0, -1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, R1, 0, 0, -1, 0],
                               [0, 0, 0, 0, R2, 0, 0, 0, -1],
                               [0, 0, 0, 0, 0, -1, 1, 1, 1]])

        self.B = np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0]])
        self.C = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])

        return self.E, self.A, self.B, self.C


if __name__ == '__main__':

    bm = index_2_daes()

    # two interconnected rotating masses

    E, A, B, C = bm.two_interconnected_rotating_masses(1, 1)
    print "\nTwo interconnected rotating masses benchmark:"
    print "\nE = {} \nA ={} \nB={} \nC={}".format(E, A, B, C)

    # RL network
    E, A, B, C = bm.RL_network(1, 1)
    print "\nRL network:"
    print "\nE = {} \nA ={} \nB={} \nC={}".format(E, A, B, C)

    bm = index_3_daes()

    # Car Pendulum benchmark
    E, A, B, C = bm.car_pendulum(2, 1, 1)
    print "\nCar pendulum benchmark:"
    print "\nE = {} \nA ={} \nB={} \nC={}".format(E, A, B, C)
