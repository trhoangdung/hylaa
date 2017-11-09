'''
This module contains some DAE benchmarks with the index from 1 to 3
Dung Tran Nov-2017

'''

import math
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, eye
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

        def stoke_equation_2d(self, length, numOfCells):
            'A index-2 large scale DAE, stoke quation in a square region'

            # This benchmark is from the paper:
            # Balanced Truncation Model Reduction for Systems in Descriptor Form
            # system's dimension = 3n^2 + 2n, n = numOfCells

            isinstance(numOfCells, int)
            isinstance(length, float)
            assert numOfCells > 1, 'number of mesh points shoubld be >= 2'
            assert length > 0, 'length of square should be large than 0'

            n = numOfCells
            h = length / (n + 1)    # discretization step

            k = 1 / h**2
            l = 1 / h

            # handle dynamic part: dv/dt = div^2 v - div p + f
            # using MAC scheme
            ##################################################################
            #    v = v_x + v_y (velocity vector)
            #    dv_x/dt = div^2 (v_x) - div_x (p) + f_x
            #    dv_y/dt = div^2 (v_y) - div_y (p) + f_y
            #    div_x(v_x) + div_y(v_y) = 0
            #
            #    state space format:
            #    dv_x/dt = matrix_V_x * v_x + matrix_P_x * p + f_x
            #    dv_y/dt = matrix_V_y * v_y + matrix_P_y * p + f_y
            #    transpose(matrix_P_x)* v_x + transpose(matrix_P_y) * v_y = 0
            #
            #    Let x = [v_x v_y p]^T we have:
            #    E * dx/dt = A * x + B * u(t)
            #    E = [I 0; 0 0]; A = [V_x 0 P_x; 0 V_y P_y; P_x^T P_y^T 0]
            ##################################################################

            # Boundary conditions : at the boundary we have v( 0, y, t) = 0, v(1, y, t) = 0 (Dirichlet boundary conditions)
            # condition for the force f: x = 0, f_x(0, y, t) = 1, f_y(0, y, t) = 1; x > 0, f = 0
            # More boundary conditions can be found in the paper: Various boundary conditions for
            # Navier-stokes equation in bounded Lipschitz domains

            num_var = n * (n + 1)    # number of variables (points) along one x/y - axis

            V_x = lil_matrix((num_var, num_var), dtype=float)    # matrix corresponds to velocity v_x
            V_y = lil_matrix((num_var, num_var), dtype=float)    # matrix corresponds to velocity v_y
            B_x = lil_matrix((num_var, 1), dtype=float)    # matrix corresponds to the force fx
            B_y = lil_matrix((num_var, 1), dtype=float)    # matrix corresponds to the force fy

            # filling V_x, B_x
            for i in xrange(0, num_var):
                y_pos = int(math.ceil(i / n))    # y-position of i-th state variable
                x_pos = i - y_pos * n    # x_position of i-th state variable
                print "\nV_x: the {}th variable is the velocity of the flow at the point ({}, {})".format(i, x_pos, y_pos)
                V_x[i, i] = -4

                if y_pos == 0:
                    B_x[i, 0] = 1    # input force fx at boundary x = 0

                if x_pos - 1 >= 0:
                    V_x[i, i - 1] = 1    # boundary condition at x = 0, v_x = 0

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

                if y_pos == 0:
                    B_y[i, 0] = 1    # input force fy at boudary x = 0

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
            for i in xrange(0, num_var):
                y_pos = int(math.ceil(i / n))    # y-position of i-th state variable
                x_pos = i - y_pos * n    # x_position of i-th state variable

                if i <= n * n - 1:
                    P_x[i, i] = 1

                if y_pos - 1 >= 0:
                    P_x[i, (y_pos - 1) * n + x_pos] = -1

            # filling P_y
            for i in xrange(0, num_var):
                y_pos = int(math.ceil(i / (n + 1)))    # y-position of i-th state variable
                x_pos = i - y_pos * (n + 1)    # x-position of i-th state variable

                if x_pos <= n - 1:
                    j = y_pos * n + x_pos    # the j-th correpsonding pressure variable
                    P_y[i, j] = 1
                    if x_pos - 1 >= 0:
                        P_y[i, j - 1] = -1
                else:
                    P_y[i, y_pos * n + x_pos - 1] = -1

            V_x.tocsr()
            V_y.tocsr()
            P_x.tocsr()
            P_y.tocsr()
            matrix_V_x = V_x.multiply(k)    # scale matrix with k = 1/h**2
            matrix_V_y = V_y.multiply(k)
            matrix_P_x = P_x.multiply(l)    # scale matrix with l = 1/h
            matrix_P_y = P_y.multiply(l)

            # constructing matrix E, A, B, C
            identity_mat = eye(2 * num_var, dtype=float, format='csr')
            zero_mat = csr_matrix((2 * num_var, n * n), dtype=float)
            E1 = hstack([identity_mat, zero_mat])
            E2 = hstack([csr_matrix.transpose(zero_mat), csr_matrix((n * n, n * n), dtype=float)])
            E = vstack([E1, E2])
            self.E = E.tocsr()

            V1 = hstack([matrix_V_x, csr_matrix((num_var, num_var), dtype=float)])
            V2 = hstack([csr_matrix((num_var, num_var), dtype=float), matrix_V_y])
            matrix_V = vstack([V1, V2])
            matrix_P = vstack([matrix_P_x, matrix_P_y])
            A1 = hstack([matrix_V, matrix_P])
            A2 = hstack([csr_matrix.transpose(matrix_P.tocsr()), csr_matrix((n * n, n * n), dtype=float)])
            A = vstack([A1, A2])
            self.A = A.tocsr()

            zero_vec = csr_matrix((n * n, 1), dtype=float)
            B = vstack([B_x, B_y, zero_vec])
            self.B = B.tocsr()

            # we are interested in the velocity v_x and v_y of the middle point of the middle cell

            centre = int(math.ceil((n - 1) / 2))
            vx1_index = centre * n + centre
            vx2_index = (centre + 1) * n + centre
            Cx = lil_matrix((1, num_var), dtype=float)
            Cx[0, vx1_index] = 0.5
            Cx[0, vx2_index] = 0.5

            vy1_index = (centre + 1) * n + centre
            vy2_index = (centre + 1) * n + centre + 1
            Cy = lil_matrix((1, num_var), dtype=float)
            Cy[0, vy1_index] = 0.5
            Cy[0, vy2_index] = 0.5
            zero_mat2 = csr_matrix((1, num_var), dtype=float)
            zero_mat3 = csr_matrix((1, n * n), dtype=float)
            C1 = hstack([Cx, zero_mat2, zero_mat3])
            C2 = hstack([zero_mat2, Cy, zero_mat3])
            C = vstack([C1, C2])
            self.C = C.tocsr()

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

    # index 2 benchmarks
    bm = index_2_daes()

    # two interconnected rotating masses

    E, A, B, C = bm.two_interconnected_rotating_masses(1, 1)
    print "\n########################################################"
    print "\nTWO INTERCONNECTED ROTATING MASSES:"
    print "\ndimensions: {}".format(E.shape[0])
    print "\nE = {} \nA ={} \nB={} \nC={}".format(E, A, B, C)

    # RL network
    E, A, B, C = bm.RL_network(1, 1)
    print "\n########################################################"
    print "\nRL network:"
    print "\ndimensions: {}".format(E.shape[0])
    print "\nE = {} \nA ={} \nB={} \nC={}".format(E, A, B, C)

    # Stoke equation 2d
    numOfCells = 3
    length = 1.0
    E, A, B, C = bm.stoke_equation_2d(length, numOfCells)
    print "\n########################################################"
    print "\n2-DIMENSIONAL STOKE EQUATION:"
    print"\nnumber of cells in one direction = {}, \nsystem's dimension = {}".format(numOfCells, A.shape[0])
    print "\nE = {}, \nA = {}, \nB = {}, \nC = {}".format(E.todense(), A.todense(), B.todense(), C.todense())

    # index 3 benchmarks
    bm = index_3_daes()

    # Car Pendulum benchmark
    E, A, B, C = bm.car_pendulum(2, 1, 1)
    print "\n########################################################"
    print "\nCAR PENDULUM:"
    print "\ndimensions: {}".format(E.shape[0])
    print "\nE = {} \nA ={} \nB={} \nC={}".format(E, A, B, C)

    # Damped mass-spring Systems
    numOfMasses = 3
    E, A, B, C = bm.damped_mass_spring(numOfMasses)
    print "\n########################################################"
    print "\nDAMPED MASS-SPRING SYSTEM:"
    print "\nnumber of masses: {}".format(numOfMasses)
    print "\ndimensions: {}".format(E.shape[0])
    print "\nE = {} \nA ={} \nB={} \nC={}".format(E.todense(),
                                                  A.todense(), B.todense(), C.todense())
