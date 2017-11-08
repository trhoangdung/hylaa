'''
This module contains some DAE benchmarks with the index from 1 to 3
Dung Tran Nov-2017

'''

import math
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse import hstack, vstack


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

        def stoke_equation_2d(self, length, numOfCell):
            'A index-2 large scale DAE, stoke quation in a square region'

            # This benchmark is from the paper:
            # Balanced Truncation Model Reduction for Systems in Descriptor Form

            # Boundary condition : at the boundary we have v( 0, y, t) = 0, p (0, y, t) = 1 (Dirichlet boundary conditions)
            # More boundary conditions can be found in the paper: Various boundary conditions for
            # Navier-stokes equation in bounded Lipschitz domains

            isinstance(numOfCell, int)
            isinstance(length, float)
            assert numOfCell > 2, 'number of mesh points shoubld be large than 2'
            assert length > 0, 'length of square should be large than 0'

            # For simplicity, we use the same number of mesh point for velocity and pressure: n_v = n_p = numOfMeshPoint
            n = numOfCell
            h = length / (n + 1)    # discretization step

            k = 1 / h**2
            l = 1 / h

            # handle dynamic part: dv/dt = div^2 v - div p + f
            # using MAC scheme

            num_var = n * (n + 1)    # number of variables (points) along one x/y - axis

            V_x = lil_matrix((num_var, num_var), dtype=float)    # matrix corresponds to velocity v_x
            V_y = lil_matrix((num_var, num_var), dtype=float)    # matrix corresponds to velocity v_y

            # filling V_x
            for i in xrange(0, num_var):
                y_pos = int(math.ceil(i / n))    # y-position of i-th state variable
                x_pos = i - y_pos * n    # x_position of i-th state variable
                print "\nV_x: the {}th variable is the velocity of the flow at the point ({}, {})".format(i, x_pos, y_pos)
                V_x[i, i] = -4

                if x_pos - 1 >= 0:
                    V_x[i, i - 1] = 1    # boundary condition at x = 0, v_x = 1

                if x_pos + 1 <= n - 1:
                    V_x[i, i + 1] = 1    # boundary condition at x = 1, v_x = 0

                if y_pos - 1 >= 0:
                    V_x[i, (y_pos - 1) * n + x_pos] = 1
                else:
                    V_x[i, i] = V_x[i, i] - 1    # extrapolation at y = 0

                if y_pos + 1 <= n:
                    V_x[i, (y_pos + 1) * n + x_pos] = 1
                else:
                    V_x[i, i] = V_x[i, i] - 1    # extrapolation at y = 1

            # filling V_y

            for i in xrange(0, num_var):
                y_pos = int(math.ceil(i / (n + 1)))    # y-position of i-th state variable
                x_pos = i - y_pos * (n + 1)    # x-position of i-th state variable
                print "\nV_y: the {}th variable is the velocity of the flow at the point ({}, {})".format(i, x_pos, y_pos)
                V_y[i, i] = -4

                if x_pos - 1 >= 0:
                    V_y[i, i - 1] = 1    # boundary condition at x = 0, v_y = 0

                if x_pos + 1 <= n:
                    V_y[i, i + 1] = 1    # boundary condition at x = 1, v_y = 0

                if y_pos - 1 >= 0:
                    V_y[i, (y_pos - 1) * (n + 1) + x_pos] = 1
                else:
                    V_y[i, i] = V_y[i, i] - 1    # extrapolation at y = 0

                if y_pos + 1 <= n - 1:
                    V_y[i, (y_pos + 1) * (n + 1) + x_pos] = 1
                else:
                    V_y[i, i] = V_y[i, i] - 1    # extrapolation at y = 1

            P_x = lil_matrix((n * (n + 1), n * n), dtype=float)    # matrix corresponds to pressure px
            P_y = lil_matrix((n * (n + 1), n * n), dtype=float)    # matrix corresponds to pressure py

            # filling P_x


            return V_x, V_y, P_x, P_y


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
        'This is 9-dimensional electric generator benchmark'

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

    def damped_mass_spring(self, numOfMasses):
        'a large scale index-3 damped mass spring system, dimension = 2 * numOfMasses + 1'

        # This benchmark is from paper: Balanced Truncation Model Reduction for Large-Scale Systems
        # in Descriptor From.
        # It is also presented in the thesis: Index-aware Model Order Reduction Methods
        # Section 2.4.3

        isinstance(numOfMasses, int)
        assert numOfMasses > 0

        g = numOfMasses
        n = 2 * g + 1    # system's dimension

        # Assume we have a uniformed damped mass-spring
        # system's parameters :
        # m1 = m2 = ... = mg = 100
        # k1 = k2 = ... = k_(g-1)  = h2 = h3 = ... = h_(g-1) = k, h1 = hg = 2 * k
        # d1 = d2 = ... = d_(g-1) = l2 = l3 = ... l_(g-1) = l_g = d, l1 = lg = 2 * d
        m = 100    # mass
        k = 2      # stiffness
        d = 5      # damping

        # damping and sfiffness matrices (tridiagonal sparse matrices)
        K = lil_matrix((g, g), dtype=float)
        D = lil_matrix((g, g), dtype=float)
        for i in xrange(0, g):
            K[i, i] = -3 * k
            D[i, i] = -3 * d
            if i > 0:
                K[i, i - 1] = 2 * k
                D[i, i - 1] = 2 * d
            if i < g - 2:
                K[i, i + 1] = k
                D[i, i + 1] = d

        K = K.tocsr()
        D = D.tocsr()

        # mass matrix diagonal sparse matrices
        M = lil_matrix((g, g), dtype=float)
        for i in xrange(0, g):
            M[i, i] = m
        M = M.tocsr()

        # Constraint matrix: this means that:
        # The first and the last masses move a same amount of distance at all time
        G = lil_matrix((1, g), dtype=float)
        G[0, 0] = 1
        G[0, g - 1] = -1
        G = G.tocsr()

        # force matrix : B2 = e1, at t=0, the first mass is pushed with a force 1 <= f <= 1.1
        B2 = lil_matrix((g, 1), dtype=float)
        B2[0, 0] = 1
        B2 = B2.tocsr()

        # identity matrix of g-dimension
        I = lil_matrix((g, g), dtype=float)
        for i in xrange(0, g):
            I[i, i] = 1
        I = I.tocsr()

        # zero matrix of g-dimension and zero vector
        Z = csr_matrix((g, g), dtype=float)
        zero_vec = csr_matrix((g, 1), dtype=float)

        # Constructing system's matrices: E, A, B, C

        E1 = hstack([I, Z, zero_vec])
        E2 = hstack([Z, M, zero_vec])
        E3 = csr_matrix((1, n), dtype=float)
        E = vstack([E1, E2, E3])
        self.E = E

        A1 = hstack([Z, I, zero_vec])
        A2 = hstack([K, D, csr_matrix.transpose(-G)])
        A3 = hstack([G, csr_matrix.transpose(zero_vec), csr_matrix((1, 1), dtype=float)])
        A = vstack([A1, A2, A3])
        self.A = A

        B = vstack([zero_vec, B2, csr_matrix((1, 1), dtype=float)])
        self.B = B

        C = lil_matrix((2, n), dtype=float)
        p = math.ceil(g / 2)    # we are interested in the position and velocity of the middle mass
        C[0, p] = 1        # position of the middle mass
        C[1, g + p] = 1    # velocity of the middle mass
        C = C.tocsr()
        self.C = C

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

    # Stoke equation 2d
    V_x, V_y, P_x, P_y = bm.stoke_equation_2d(1.0, 3)
    print "\n2-dimensional stoke equation:"
    print "\nV_x = {}, \nV_y = {}, \nP_x = {}, \nP_y = {}".format(V_x.todense(), V_y.todense(),
                                                                  P_x.todense(), P_y.todense())

    print "\ntranspose_P_y = {}".format(np.transpose(P_y.todense()))

    bm = index_3_daes()

    # Car Pendulum benchmark
    E, A, B, C = bm.car_pendulum(2, 1, 1)
    print "\nCar pendulum benchmark:"
    print "\nE = {} \nA ={} \nB={} \nC={}".format(E, A, B, C)

    # Damped mass-spring Systems
    E, A, B, C = bm.damped_mass_spring(3)
    print "\n Damped mass-spring system:"
    print "\nE = {} \nA ={} \nB={} \nC={}".format(E.todense(),
                                                  A.todense(), B.todense(), C.todense())
