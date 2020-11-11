# --- NRSur7dq2.py ---

"""
A module for evaluating the NRSur7dq2 surrogate model of gravitational waves
from numerical relativity simulations of binary black hole mergers, as
published in Blackman et al. 2017 PRD.
"""

__copyright__ = "Copyright (C) 2017 Jonathan Blackman"
__email__     = "jonathan.blackman.0@gmail.com"
__author__    = "Jonathan Blackman"
__version__   = "1.0.6"
__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import os
import numpy as np
import h5py
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import _NRSur7dq2_utils as _utils
import warnings
from .harmonics import sYlm

SOLAR_TIME_IN_SECONDS = 4.925491025543576e-06
SOLAR_DISTANCE_IN_MEGAPARSECS = 4.785415917274702e-20


###############################################################################
# Simple quaternion functions

def multiplyQuats(q1, q2):
    return np.array([
            q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
            q1[2]*q2[3] - q2[2]*q1[3] + q1[0]*q2[1] + q2[0]*q1[1],
            q1[3]*q2[1] - q2[3]*q1[1] + q1[0]*q2[2] + q2[0]*q1[2],
            q1[1]*q2[2] - q2[1]*q1[2] + q1[0]*q2[3] + q2[0]*q1[3]])

def quatInv(q):
    """Returns QBar such that Q*QBar = 1"""
    qConj = -q
    qConj[0] = -qConj[0]
    normSqr = multiplyQuats(q, qConj)[0]
    return qConj/normSqr

###############################################################################
# Functions related to frame transformations

def _assemble_powers(thing, powers):
    return np.array([thing**power for power in powers])

def _wignerD_matrices(q, LMax):
    """
Given a quaternion q with shape (4, N) and some maximum ell value LMax,
computes W[ell, m', m](t_i) for i=0, ..., N-1, for 2 \leq ell \leq LMax,
for -L \leq m', m \leq L.
Returns a list where each entry is a numpy array with shape
((2*ell+1), (2*ell+1), N) corresponding to a given value of ell, taking indices
for m', m, and t_i.

Parts of this function are adapted from GWFrames:
https://github.com/moble/GWFrames
written by Michael Boyle, based on his paper:
http://arxiv.org/abs/1302.2919
    """
    ra = q[0] + 1.j*q[3]
    rb = q[2] + 1.j*q[1]
    ra_small = (abs(ra) < 1.e-12)
    rb_small = (abs(rb) < 1.e-12)
    i1 = np.where((1 - ra_small)*(1 - rb_small))[0]
    i2 = np.where(ra_small)[0]
    i3 = np.where((1 - ra_small)*rb_small)[0]

    n = len(ra)
    lvals = range(2, LMax+1)
    matrices = [0.j*np.zeros((2*ell+1, 2*ell+1, n)) for ell in lvals]

    # Determine res at i2: it's 0 unless mp == -m
    # Determine res at i3: it's 0 unless mp == m
    for i, ell in enumerate(lvals):
        for m in range(-ell, ell+1):
            if (ell+m)%2 == 1:
                matrices[i][ell+m, ell-m, i2] = rb[i2]**(2*m)
            else:
                matrices[i][ell+m, ell-m, i2] = -1*rb[i2]**(2*m)
            matrices[i][ell+m, ell+m, i3] = ra[i3]**(2*m)

    # Determine res at i1, where we can safely divide by ra and rb
    ra = ra[i1]
    rb = rb[i1]
    ra_pows = _assemble_powers(ra, range(-2*LMax, 2*LMax+1))
    rb_pows = _assemble_powers(rb, range(-2*LMax, 2*LMax+1))
    abs_raSqr_pows = _assemble_powers(abs(ra)**2, range(0, 2*LMax+1))
    absRRatioSquared = (abs(rb)/abs(ra))**2
    ratio_pows = _assemble_powers(absRRatioSquared, range(0, 2*LMax+1))

    for i, ell in enumerate(lvals):
        for m in range(-ell, ell+1):
            for mp in range(-ell, ell+1):
                factor = _utils.wigner_coef(ell, mp, m)
                factor *= ra_pows[2*LMax + m+mp]
                factor *= rb_pows[2*LMax + m-mp]
                factor *= abs_raSqr_pows[ell-m]
                rhoMin = max(0, mp-m)
                rhoMax = min(ell+mp, ell-m)
                s = 0.
                for rho in range(rhoMin, rhoMax+1):
                    c = ((-1)**rho)*(_utils.binom(ell+mp, rho)*
                                     _utils.binom(ell-mp, ell-rho-m))
                    s += c * ratio_pows[rho]
                matrices[i][ell+m, ell+mp, i1] = factor*s

    return matrices

def rotateWaveform(quat, h):
    """
Transforms a waveform from the coprecessing frame to the inertial frame.
quat: A quaternion array with shape (4, N) where N is the number of time
      samples describing the coprecessing frame
h: An array of waveform modes with shape (n_modes, N). The modes are ordered
    (2, -2), ..., (2, 2), (3, -3), ...
    and n_modes = 5, 12, or 21 for LMax = 2, 3, or 4.

Returns: h_inertial, a similar array to h containing the inertial frame modes.
    """
    quat = quatInv(quat)

    LMax = {
            5: 2,
            12: 3,
            21: 4,
            32: 5,
            45: 6,
            60: 7,
            77: 8,
            }[len(h)]

    matrices = _wignerD_matrices(quat, LMax)

    res = 0.*h
    i=0
    for ell in range(2, LMax+1):
        for m in range(-ell, ell+1):
            for mp in range(-ell, ell+1):
                res[i+m+ell] += matrices[ell-2][ell+m, ell+mp]*h[i+mp+ell]
        i += 2*ell + 1
    return res

def transformTimeDependentVector(quat, vec):
    """
Given a coprecessing frame quaternion quat, with shape (4, N),
and a vector vec, with shape (3, N), transforms vec from the
coprecessing frame to the inertial frame.
    """
    qInv = quatInv(quat)
    return multiplyQuats(quat, multiplyQuats(np.append(np.array([
            np.zeros(len(vec[0]))]), vec, 0), qInv))[1:]

###############################################################################

class DynamicsSurrogate:
    """
A surrogate intended to reproduce the orbital, precession, and spin dynamics
of numerical relativity BBH waveforms and spins.

This surrogate models:
    -The coprecessing frame
    -The orbital phase in the coprecessing frame, which we can use
     to find the coorbital frame
    -The spin directions in the coprecessing frame

As input, it takes:
    -The quaternion giving the initial coprecessing frame
    -The initial orbital phase in the coprecessing frame
    -The initial spin directions in the coprecessing frame

Using the input, it evolves a system of ODEs.
Given the things this surrogate models at time t, it evaluates a
prediction for:
    -The quaternion derivative in the coprecessing frame: q^{-1}(t)q'(t)
     from which q'(t) is computed
    -The orbital frequency in the coprecessing frame
    -The time derivatives of the spins in the coprecessing frame
These time derivatives are given to the AB4 ODE solver.
    """

    def __init__(self, h5file):
        """h5file is a h5py.File containing the surrogate data"""
        self.t = h5file['t_ds'][()]

        self.fit_data = []
        for i in range(len(self.t)):
            group = h5file['ds_node_%s'%(i)]
            tmp_data = {}
            for key in ['omega', 'omega_orb', 'chiA', 'chiB']:
                tmp_data[key] = {
                        'coefs': group['%s_coefs'%(key)][()],
                        'bfOrders': group['%s_bfOrders'%(key)][()]
                        }
                if key != 'omega':
                    tmp_data[key]['bVecIndices'] = group['%s_bVecIndices'%(key)][()]
            self.fit_data.append(tmp_data)

        self.diff_t = np.diff(self.t)
        self.L = len(self.t)

        # Validate time array
        for i in range(3):
            if not self.diff_t[2*i] == self.diff_t[2*i+1]:
                raise Exception("ab4 needs to do 3 steps of RK4 integration!")

    def get_time_deriv_from_index(self, i0, q, y):
        # Setup fit variables
        x = _utils.get_ds_fit_x(y, q)

        # Evaluate fits
        data = self.fit_data[i0]
        ooxy_coorb = _utils.eval_vector_fit(
                2,
                data['omega_orb']['bfOrders'],
                data['omega_orb']['coefs'],
                data['omega_orb']['bVecIndices'],
                x)
        omega = _utils.eval_fit(
                data['omega']['bfOrders'],
                data['omega']['coefs'],
                x)
        cAdot_coorb = _utils.eval_vector_fit(
                3,
                data['chiA']['bfOrders'],
                data['chiA']['coefs'],
                data['chiA']['bVecIndices'],
                x)
        cBdot_coorb = _utils.eval_vector_fit(
                3,
                data['chiB']['bfOrders'],
                data['chiB']['coefs'],
                data['chiB']['bVecIndices'],
                x)

        # Do rotations to the coprecessing frame, find dqdt, and append
        dydt = _utils.assemble_dydt(y, ooxy_coorb, omega, cAdot_coorb, cBdot_coorb)
        return dydt

    def get_time_deriv(self, t, q, y):
        """
Evaluates dydt at a given time t by interpolating dydt at 4 nearby nodes with
cubic interpolation. Use get_time_deriv_from_index when possible.
        """
        if t < self.t[0] or t > self.t[-1]:
            raise Exception("Cannot extrapolate time derivative!")
        i0 = np.argmin(abs(self.t - t))
        if t > self.t[i0]:
            imin = i0-1
        else:
            imin = i0-2
        imin = min(max(0, imin), len(self.t)-4)
        dydts = np.array([self.get_time_deriv_from_index(imin+i, q, y) for i in range(4)])
        ts = self.t[imin:imin+4]
        dydt = np.array([spline(ts, x)(t) for x in dydts.T])
        return dydt

    def get_omega(self, i0, q, y):
        x = _utils.get_ds_fit_x(y, q)
        data = self.fit_data[i0]
        omega = _utils.eval_fit(
                data['omega']['bfOrders'],
                data['omega']['coefs'],
                x)
        return omega

    def _get_t_ref(self, omega_ref, q, chiA0, chiB0, init_orbphase, init_quat):
        if omega_ref > 0.201:
            raise Exception("Got omega_ref = %0.4f > 0.2, too large!"%(omega_ref))

        y0 = np.append(np.array([1., 0., 0., 0., init_orbphase]), np.append(chiA0, chiB0))
        if init_quat is not None:
            y0[:4] = init_quat

        omega0 = self.get_omega(0, q, y0)
        if omega_ref < omega0:
            raise Exception("Got omega_ref = %0.4f < %0.4f = omega_0, too small!"%(
                    omega_ref, omega0))

        # i0=0 is a lower bound, find the first index where omega > omega_ref
        imax=1
        omega_max = self.get_omega(imax, q, y0)
        omega_min = omega0
        while omega_max <= omega_ref:
            imax += 1
            omega_min = omega_max
            omega_max = self.get_omega(imax, q, y0)

        # Interpolate
        t_ref = (self.t[imax-1] * (omega_max - omega_ref) + self.t[imax] * (omega_ref - omega_min))/(omega_max - omega_min)
        if t_ref < self.t[0] or t_ref > self.t[-1]:
            raise Exception("Somehow, t_ref ended up being outside of the time domain limits!")
        return t_ref

    def __call__(self, q, chiA0, chiB0, init_quat=None, init_orbphase=0.0,
                 t_ref=None, omega_ref=None, allow_extrapolation=False):
        """
Computes the modeled NR dynamics given the initial conditions.

Arguments:
=================
q: The mass ratio
chiA0: The chiA vector at the reference time, given in the coprecessing frame
chiB0: The chiB vector at the reference time, given in the coprecessing frame
init_quat: The quaternion giving the rotation to the coprecessing frame at the
           reference time. By default, this will be the identity quaternion,
           indicating the coprecessing frame is aligned with the inertial frame.
init_orbphase: The orbital phase in the coprecessing frame at the reference time
t_ref: The reference (dimensionless) time, where the peak amplitude occurs at t=0.
       Default: The initial time, t_0/M = -4500.
omega_ref: The dimensionless orbital angular frequency used to determine t_ref,
       which is the time derivative of the orbital phase in the coprecessing frame.
       Specify at most one of t_ref, omega_ref.
allow_extrapolation: Enable arbitrary extrapolation of the surrogate in mass
       ratio and spin magnitude. By default, only allow tiny extrapolations.

Returns:
==================
q_copr: The quaternion representing the coprecessing frame with shape (4, L)
orbphase: The orbital phase in the coprecessing frame with shape (L, )
chiA: The time-dependent chiA in the coprecessing frame with shape (L, 3)
chiB: The time-dependent chiB in the coprecessing frame with shape (L, 3)

L = len(self.t), and these returned arrays are sampled at self.t 
        """

        if t_ref is not None and omega_ref is not None:
            raise Exception("Specify at most one of t_ref, omega_ref.")

        # Sanity checks, allowing tiny extrapolations, and either warning
        # or raising exceptions on large extrapolations.
        if q < 0.99:
            # Black hole A is defined to be the one with a larger mass,
            # and q = mA/mB.
            raise Exception("The mass ratio q should be >= 1")
        if q > 2.01:
            if allow_extrapolation:
                warnings.warn("Extrapolating dynamics to q=%s > 2.0"%(q))
            else:
                raise Exception("Mass ratio %s > 2 outside training range"%(q))
        if chiA0.shape != (3, ) or chiB0.shape != (3, ):
            raise Exception("chiA0 and chiB0 should have shape (3, )")

        normA = np.sqrt(np.sum(chiA0**2))
        normB = np.sqrt(np.sum(chiB0**2))
        maxNorm = max(normA, normB)
        if maxNorm > 1.001:
            raise Exception("Got a spin magnitude of %s > 1.0"%(maxNorm))
        if maxNorm > 0.801:
            if allow_extrapolation:
                warnings.warn("Extrapolating dynamics to |chi|=%s > 0.8"%(
                    maxNorm))
            else:
                raise Exception("Spin magnitude %s outside training range"%(maxNorm))

        if omega_ref is not None:
            t_ref = self._get_t_ref(omega_ref, q, chiA0, chiB0, init_orbphase, init_quat)
        y_of_t, i0 = self._initialize(q, chiA0, chiB0, init_quat, init_orbphase, t_ref,
                                      normA, normB)

        if i0 == 0:
            # Just gonna send it!
            k_ab4, dt_ab4, y_of_t = self._initial_RK4(q, y_of_t, normA, normB)
            y_of_t = self._integrate_forward(q, y_of_t, normA, normB, 3, k_ab4, dt_ab4)
        elif i0 > 2:
            # Initialize by taking 3 steps backwards with RK4
            k_ab4 = [None, None, None]
            for i in range(3):
                y_of_t, tmp_k = self._one_backward_RK4_step(q, y_of_t, normA, normB, i0-i)
                k_ab4[i] = tmp_k
            dt_array = np.append(2 * self.diff_t[:6:2], self.diff_t[6:])
            dt_ab4 = dt_array[i0-3:i0][::-1]
            self._integrate_backward(q, y_of_t, normA, normB, i0-3, k_ab4, dt_ab4)
            tmp_k = self.get_time_deriv_from_index(i0, q, y_of_t[i0-3])
            k_ab4 = [tmp_k, k_ab4[2], k_ab4[1]]
            dt_ab4 = dt_ab4[::-1]
            self._integrate_forward(q, y_of_t, normA, normB, i0, k_ab4, dt_ab4)
        else:
            # Initialize by taking 3 steps forwards with RK4
            k_ab4 = [None, None, None]
            for i in range(3):
                y_of_t, tmp_k = self._one_forward_RK4_step(q, y_of_t, normA, normB, i0+i)
                k_ab4[i] = tmp_k
            dt_array = np.append(2 * self.diff_t[:6:2], self.diff_t[6:])
            dt_ab4 = dt_array[i0:i0+3]
            self._integrate_forward(q, y_of_t, normA, normB, i0+3, k_ab4, dt_ab4)
            tmp_k = self.get_time_deriv_from_index(i0+3, q, y_of_t[i0+3])
            k_ab4 = [tmp_k, k_ab4[2], k_ab4[1]]
            dt_ab4 = dt_ab4[::-1]
            self._integrate_backward(q, y_of_t, normA, normB, i0, k_ab4, dt_ab4)

        quat = y_of_t[:, :4].T
        orbphase = y_of_t[:, 4]
        chiA_copr = y_of_t[:, 5:8]
        chiB_copr = y_of_t[:, 8:]

        return quat, orbphase, chiA_copr, chiB_copr

    def _initialize(self, q, chiA0, chiB0, init_quat, init_orbphase, t_ref, normA, normB):
        """
Initializes an array of data with the initial conditions.
If t_ref does not correspond to a time node, takes one small time step to
the nearest time node.
        """
        # data is [q0, qx, qy, qz, orbphase, chiAx, chiAy, chiAz, chiBx, chiBy, chiBz]
        # We do three steps of RK4, so we have 3 fewer timesteps in the output
        # compared to self.t
        data = np.zeros((self.L-3, 11))

        y0 = np.append(np.array([1., 0., 0., 0., init_orbphase]), np.append(chiA0, chiB0))
        if init_quat is not None:
            y0[:4] = init_quat

        if t_ref is None:
            data[0, :] = y0
            i0 = 0
        else:
            # Step to the closest time node using forward Euler
            times = np.append(self.t[:6:2], self.t[6:])
            i0 = np.argmin(abs(times - t_ref))
            t0 = times[i0]
            dydt0 = self.get_time_deriv(t_ref, q, y0)
            y_node = y0 + (t0 - t_ref) * dydt0
            y_node = _utils.normalize_y(y_node, normA, normB)
            data[i0, :] = y_node

        return data, i0

    def _initial_RK4(self, q, y_of_t, normA, normB):
        """This is used to initialize the AB4 system when t_ref=t_0 (default)"""


        # Three steps of RK4
        k_ab4 = []
        dt_ab4 = []
        for i, dt in enumerate(self.diff_t[:6:2]):
            k1 = self.get_time_deriv_from_index(2*i, q, y_of_t[i])
            k_ab4.append(k1)
            dt_ab4.append(2*dt)
            k2 = self.get_time_deriv_from_index(2*i+1, q, y_of_t[i] + dt*k1)
            k3 = self.get_time_deriv_from_index(2*i+1, q, y_of_t[i] + dt*k2)
            k4 = self.get_time_deriv_from_index(2*i+2, q, y_of_t[i] + 2*dt*k3)
            ynext = y_of_t[i] + (dt/3.)*(k1 + 2*k2 + 2*k3 + k4)
            y_of_t[i+1] = _utils.normalize_y(ynext, normA, normB)

        return k_ab4, dt_ab4, y_of_t

    def _one_forward_RK4_step(self, q, y_of_t, normA, normB, i0):
        """Steps forward one step using RK4"""

        # i0 is on the y_of_t grid, which has 3 fewer samples than the self.t grid
        i_t = i0 + 3
        if i0 < 3:
            i_t = i0*2

        t1 = self.t[i_t]
        t2 = self.t[i_t + 1]
        if i0 < 3:
            t2 = self.t[i_t + 2]
        half_dt = 0.5*(t2 - t1)

        k1 = self.get_time_deriv(t1, q, y_of_t[i0])
        k2 = self.get_time_deriv(t1 + half_dt, q, y_of_t[i0] + half_dt*k1)
        k3 = self.get_time_deriv(t1 + half_dt, q, y_of_t[i0] + half_dt*k2)
        k4 = self.get_time_deriv(t2, q, y_of_t[i0] + 2*half_dt*k3)
        ynext = y_of_t[i0] + (half_dt/3.)*(k1 + 2*k2 + 2*k3 + k4)
        y_of_t[i0+1] = _utils.normalize_y(ynext, normA, normB)
        return y_of_t, k1

    def _one_backward_RK4_step(self, q, y_of_t, normA, normB, i0):
        """Steps backward one step using RK4"""

        # i0 is on the y_of_t grid, which has 3 fewer samples than the self.t grid
        i_t = i0 + 3
        if i0 < 3:
            i_t = i0*2

        t1 = self.t[i_t]
        t2 = self.t[i_t - 1]
        if i0 <= 3:
            t2 = self.t[i_t - 2]
        half_dt = 0.5*(t2 - t1)
        quarter_dt = 0.5*half_dt

        k1 = self.get_time_deriv(t1, q, y_of_t[i0])
        k2 = self.get_time_deriv(t1 + half_dt, q, y_of_t[i0] + half_dt*k1)
        k3 = self.get_time_deriv(t1 + half_dt, q, y_of_t[i0] + half_dt*k2)
        k4 = self.get_time_deriv(t2, q, y_of_t[i0] + 2*half_dt*k3)
        ynext = y_of_t[i0] + (half_dt/3.)*(k1 + 2*k2 + 2*k3 + k4)
        y_of_t[i0-1] = _utils.normalize_y(ynext, normA, normB)
        return y_of_t, k1

    def _integrate_forward(self, q, y_of_t, normA, normB, i0, k_ab4, dt_ab4):
        """
Use AB4 to integrate forward in time, starting at index i0.
i0 refers to the index of y_of_t, which should be the latest index at which
we already have the solution; typically i0=3 after three steps of RK4.
k_ab4 is [dydt(i0 - 3), dydt(i0 - 2), dydt(i0 - 1)]
dt_ab4 is [t(i0 - 2) - t(i0 - 3), t(i0 - 1) - t(i0 - 2), t(i0) - t(i0 - 1)]
where for both k_ab4 and dt_ab4 the indices correspond to y_of_t nodes and skip
fractional nodes.
        """
        if i0 < 3:
            raise Exception("i0 must be at least 3!")

        # Setup AB4
        k1, k2, k3 = k_ab4
        dt1, dt2, dt3 = dt_ab4

        # Run AB4
        for i, dt4 in enumerate(self.diff_t[i0+3:]): #i0+3 due to 3 half time steps
            i_output = i0+i
            k4 = self.get_time_deriv_from_index(i_output+3, q, y_of_t[i_output])
            ynext = y_of_t[i_output] + _utils.ab4_dy(k1, k2, k3, k4, dt1, dt2, dt3, dt4)
            y_of_t[i_output+1] = _utils.normalize_y(ynext, normA, normB)

            # Setup for next iteration
            k1, k2, k3 = k2, k3, k4
            dt1, dt2, dt3 = dt2, dt3, dt4

        return y_of_t

    def _integrate_backward(self, q, y_of_t, normA, normB, i0, k_ab4, dt_ab4):
        """
Use AB4 to integrate backward in time, starting at index i0.
k_ab4 is [dydt(i0 + 3), dydt(i0 + 2), dydt(i0 + 1)]
dt_ab4 is [t(i0 + 3) - t(i0 + 2), t(i0 + 2) - t(i0 + 1), t(i0 + 1) - t(i0)]
        """

        if i0 > len(self.t) - 7:
            raise Exception("i0 must be <= len(self.t) - 7")

        # Setup AB4
        k1, k2, k3 = k_ab4
        dt1, dt2, dt3 = dt_ab4

        # Setup dt array, removing the half steps
        dt_array = np.append(2 * self.diff_t[:6:2], self.diff_t[6:])
        for i_output in range(i0)[::-1]:
            node_index = i_output + 4
            if i_output < 2:
                node_index = 2 + 2*i_output
            dt4 = dt_array[i_output]
            k4 = self.get_time_deriv_from_index(node_index, q, y_of_t[i_output+1])
            ynext = y_of_t[i_output+1] - _utils.ab4_dy(k1, k2, k3, k4, dt1, dt2, dt3, dt4)
            y_of_t[i_output] = _utils.normalize_y(ynext, normA, normB)

            # Setup for next iteration
            k1, k2, k3 = k2, k3, k4
            dt1, dt2, dt3 = dt2, dt3, dt4

        return y_of_t
#########################################################

# Utility functions for the CoorbitalWaveformSurrogate:

def _extract_component_data(h5_group):
    data = {}
    data['EI_basis'] = h5_group['EIBasis'][()]
    data['nodeIndices'] = h5_group['nodeIndices'][()]
    data['coefs'] = [h5_group['nodeModelers']['coefs_%s'%(i)][()]
                     for i in range(len(data['nodeIndices']))]
    data['orders'] = [h5_group['nodeModelers']['bfOrders_%s'%(i)][()]
                      for i in range(len(data['nodeIndices']))]
    return data

def _eval_comp(data, q, chiA, chiB):
    nodes = []
    for orders, coefs, ni in zip(data['orders'], data['coefs'], data['nodeIndices']):
        nodes.append(_utils.eval_fit(
                orders, coefs, np.append(q, np.append(chiA[ni], chiB[ni]))))
    return np.array(nodes).dot(data['EI_basis'])

def _assemble_mode_pair(rep, rem, imp, imm):
    hplus = rep + 1.j*imp
    hminus = rem + 1.j*imm
    # hplus and hminus were built with the (ell, -m) mode as the
    # reference mode:
    #   hplus = 0.5*( h^{ell, -m} + h^{ell, m}* )
    #   hminus = 0.5*(h^{ell, -m} - h^{ell, m}* )
    return (hplus - hminus).conjugate(), hplus + hminus

#########################################################

class CoorbitalWaveformSurrogate:
    """This surrogate models the waveform in the coorbital frame."""

    def __init__(self, h5file):
        self.LMax = 2
        while 'hCoorb_%s_%s_Re+'%(self.LMax+1, self.LMax+1) in h5file.keys():
            self.LMax += 1

        self.t = h5file['t_coorb'][()]

        self.data = {}

        for ell in range(2, self.LMax+1):
            # m=0 is different
            for reim in ['real', 'imag']:
                group = h5file['hCoorb_%s_0_%s'%(ell, reim)]
                self.data['%s_0_%s'%(ell, reim)] = _extract_component_data(group)

            for m in range(1, ell+1):
                for reim in ['Re', 'Im']:
                    for pm in ['+', '-']:
                        group = h5file['hCoorb_%s_%s_%s%s'%(ell, m, reim, pm)]
                        tmp_data = _extract_component_data(group)
                        self.data['%s_%s_%s%s'%(ell, m, reim, pm)] = tmp_data

    def __call__(self, q, chiA, chiB, LMax=4, allow_extrapolation=False):
        """
Evaluates the coorbital waveform modes.
q: The mass ratio
chiA, chiB: The time-dependent spin in the coorbital frame. These should have
            shape (N, 3) where N = len(t_coorb)
LMax: The maximum ell mode to evaluate.
allow_extrapolation: Enable arbitrary extrapolation of the surrogate in mass
       ratio and spin magnitude. By default, only allow tiny extrapolations.
        """

        # Sanity checks, allowing small extrapolation and warning on
        # larger extrapolations
        if q < 0.99:
            # Black hole A is defined to be the one with a larger mass,
            # and q = mA/mB.
            raise Exception("The mass ratio q should be >= 1")
        if q > 2.01:
            if allow_extrapolation:
                warnings.warn("Extrapolating coorbital waveform to q=%s > 2.0"%(q))
            else:
                raise Exception("Mass ratio %s > 2 outside training range"%(q))
        normA = np.sqrt(np.sum(chiA**2, 1))
        normB = np.sqrt(np.sum(chiB**2, 1))
        maxNorm = max(np.max(normA), np.max(normB))
        if maxNorm > 1.001:
            raise Exception("Got a spin magnitude of %s > 1.0"%(maxNorm))
        if maxNorm > 0.801:
            if allow_extrapolation:
                warnings.warn("Extrapolating coorbital waveform to |chi|=%s > 0.8"%(
                    maxNorm))
            else:
                raise Exception("Spin magnitude %s > 0.8 outside training range"%(maxNorm))


        nmodes = LMax*LMax + 2*LMax - 3
        modes = 1.j*np.zeros((nmodes, len(self.t)))

        for ell in range(2, LMax+1):
            # m=0 is different
            re = _eval_comp(self.data['%s_0_real'%(ell)], q, chiA, chiB)
            im = _eval_comp(self.data['%s_0_imag'%(ell)], q, chiA, chiB)
            modes[ell*(ell+1) - 4] = re + 1.j*im

            for m in range(1, ell+1):
                rep = _eval_comp(self.data['%s_%s_Re+'%(ell, m)], q, chiA, chiB)
                rem = _eval_comp(self.data['%s_%s_Re-'%(ell, m)], q, chiA, chiB)
                imp = _eval_comp(self.data['%s_%s_Im+'%(ell, m)], q, chiA, chiB)
                imm = _eval_comp(self.data['%s_%s_Im-'%(ell, m)], q, chiA, chiB)
                h_posm, h_negm = _assemble_mode_pair(rep, rem, imp, imm)
                modes[ell*(ell+1) - 4 + m] = h_posm
                modes[ell*(ell+1) - 4 - m] = h_negm

        return modes

##############################################################################

# Utility functions for the NRSurrogate7dq2 class:

def rotate_spin(chi, phase):
    """For transforming spins between the coprecessing and coorbital frames"""
    v = chi.T
    sp = np.sin(phase)
    cp = np.cos(phase)
    res = 1.*v
    res[0] = v[0]*cp + v[1]*sp
    res[1] = v[1]*cp - v[0]*sp
    return res.T

def coorb_spins_from_copr_spins(chiA_copr, chiB_copr, orbphase):
    chiA_coorb = rotate_spin(chiA_copr, orbphase)
    chiB_coorb = rotate_spin(chiB_copr, orbphase)
    return chiA_coorb, chiB_coorb

def inertial_waveform_modes(t, orbphase, quat, h_coorb):
    q_rot = np.array([np.cos(orbphase / 2.), 0. * orbphase,
                      0. * orbphase, np.sin(orbphase / 2.)])
    qfull = multiplyQuats(quat, q_rot)
    h_inertial = rotateWaveform(qfull, h_coorb)
    return h_inertial

def splinterp_many(t_in, t_out, many_things):
    return np.array([spline(t_in, thing)(t_out) for thing in many_things])

def mode_sum(h_modes, LMax, theta, phi):
    coefs = []
    for ell in range(2, LMax+1):
        for m in range(-ell, ell+1):
            coefs.append(sYlm(-2, ell, m, theta, phi))
    return np.array(coefs).dot(h_modes)

def normalize_spin(chi, chi_norm):
    if chi_norm > 0.:
        tmp_norm = np.sqrt(np.sum(chi**2, 1))
        return (chi.T * chi_norm / tmp_norm).T
    return chi

##############################################################################

class NRSurrogate7dq2:
    """
A class for the NRSur7dq2 surrogate model presented in Blackman et al. 2017,
hence known as THE PAPER:
https://journals.aps.org/prd/abstract/10.1103/PhysRevD.96.024058
https://arxiv.org/abs/1705.07089

This surrogate model evaluates gravitational wave modes up to L=4 from binary
black hole mergers with mass ratios q <=2 and spin magnitudes up to 0.8,
with arbitrary spin directions.

See the __call__ method on how to evaluate waveforms.
    """

    def __init__(self, filename=None):
        """
Loads the NRSur7dq2 surrogate model data.

filename: The hdf5 file containing the surrogate data, probably named
"NRSur7dq2.h5. Default: Look for it in package directory."
        """
        if filename is None:
            filename = os.path.join(os.path.dirname(__file__), "NRSur7dq2.h5")
        h5file = h5py.File(filename, 'r')
        self.dynamics_sur = DynamicsSurrogate(h5file)
        self.coorb_sur = CoorbitalWaveformSurrogate(h5file)
        self.t_coorb = self.coorb_sur.t
        self.tds = np.append(self.dynamics_sur.t[0:6:2], self.dynamics_sur.t[6:])
        self.t_0 = self.t_coorb[0]
        self.t_f = self.t_coorb[-1]

    def get_dynamics(self, q, chiA0, chiB0, init_phase=0.0, init_quat=None,
                     t_ref=None, omega_ref=None, allow_extrapolation=False):
        """
Evaluates only the dynamics surrogate.
q: The mass ratio mA/mB, with 1 <= q <= 2.
chiA0, chiB0: The dimensionless black hole spins, given in the
              coprecessing frame at the reference time
init_phase: The orbital phase $\\varphi(t_ref)$ at the reference time
init_quat: The unit quaternion representing the coprecessing frame at the
           reference time.
           If None, the coprecessing frame and inertial frames will be
           aligned, and the spins can be given in the inertial frame.
t_ref: The reference (dimensionless) time, where the peak amplitude occurs at t=0.
       Default: The initial time, t_0/M = -4500.
omega_ref: The orbital angular frequency in the coprecessing frame, used to
           determine t_ref.
       Specify at most one of t_ref, f_ref.
allow_extrapolation: Enable arbitrary extrapolation of the surrogate in mass
       ratio and spin magnitude. By default, only allow tiny extrapolations.

Returns:
q_copr: The quaternion representing the coprecessing frame with shape (4, L)
orbphase: The orbital phase in the coprecessing frame with shape (L, )
chiA: The time-dependent chiA with shape (L, 3)
chiB: The time-dependent chiB with shape (L, 3)

These are sampled on self.tds which has length L.
        """ 
        return self.dynamics_sur(q, chiA0, chiB0, init_orbphase=init_phase,
                                 init_quat=init_quat, t_ref=t_ref, omega_ref=omega_ref,
                                 allow_extrapolation=allow_extrapolation)

    def get_coorb_waveform(self, q, chiA_coorb, chiB_coorb, LMax=4, allow_extrapolation=False):
        """
Evaluates the coorbital waveform surrogate.
q: The mass ratio mA/mB, with 1 <= q <=2.
chiA_coorb, chiB_coorb: The spins in the coorbital frame, with shape (N, 3)
    where N = len(self.t_coorb).
LMax: The maximum ell mode to evaluate.
allow_extrapolation: Enable arbitrary extrapolation of the surrogate in mass
       ratio and spin magnitude. By default, only allow tiny extrapolations.

Returns a 2d array with shape (n_modes, N) where the modes are ordered:
    (2, -2), ..., (2, 2), (3, -3), ...
with n_modes = 5, 12, or 21 for LMax = 2, 3, or 4 respectively
        """
        return self.coorb_sur(q, chiA_coorb, chiB_coorb, LMax=LMax,
                              allow_extrapolation=allow_extrapolation)

    def get_time_from_freq(self, freq, q, chiA0, chiB0, MTot=None,
                           init_phase=0.0, init_quat=None, t_ref=None,
                           f_ref=None):
        """
Obtain the time at which a particular gravitational wave frequency occurs.
freq: The gravitational wave frequency. If MTot is given, freq is given in Hz.
      Otherwise, freq is dimensionless.
See the __call__ docstring for other parameters.
        """
        # Determine omega_ref if needed
        omega_ref = None
        if f_ref is not None:
            if MTot is None:
                omega_ref = f_ref * np.pi
            else:
                # Get dimensionless omega_ref
                omega_ref = f_ref * MTot * SOLAR_TIME_IN_SECONDS * np.pi
                t_ref = self.dynamics_sur._get_t_ref(omega_ref, q, chiA0, chiB0,
                                                     init_orbphase, init_quat)

        # Get freqs vs time
        quat, orbphase, chiA_copr, chiB_copr = self.get_dynamics(
                q, chiA0, chiB0, init_phase=init_phase, init_quat=init_quat,
                t_ref=t_ref, omega_ref=omega_ref)

        omega = np.gradient(orbphase) / np.gradient(self.tds)
        if MTot is not None:
            freqs = omega / (MTot * SOLAR_TIME_IN_SECONDS * np.pi)
        else:
            freqs = omega / np.pi

        # Find the first time where freqs >= freq, and interpolate to find the time
        if freqs[0] > freq:
            raise Exception("Frequency %s too low: initial frequency is %s"%(
                    freq, freq[0]))
        if np.max(freqs) < freq:
            raise Exception("Frequency %s too high: maximum frequency is %s"%(
                    freq, np.max(freqs)))
        i0 = np.where(freqs >= freq)[0][0]
        t0 = np.interp(freq, freqs, self.tds)
        if MTot is not None:
            t0 = t0 * MTot * SOLAR_TIME_IN_SECONDS

        return t0

    def __call__(self, q, chiA0, chiB0, init_phase=0.0, init_quat=None,
                 return_spins=False, t=None, theta=None, phi=None, LMax=4,
                 t_ref=None, f_ref=None, MTot=None, distance=None,
                 use_lalsimulation_conventions=False, allow_extrapolation=False):
        """
Evaluates the surrogate model, returning either the inertial frame waveform
modes, or the waveform evaluated at some point on the sphere.

Arguments:
    q: The mass ratio mA/mB, with 1 <= q <=2
    chiA0, chiB0:   The initial dimensionless spins given in the coprecessing
                    frame. They should be length 3 lists or numpy arrays.
                    These are $\\vec{\chi_{1,2}^\mathrm{copr}(t_0)$ in THE PAPER.
                    Their norms should be np.sqrt(chi0**2) <= 0.8
    init_phase:     The initial orbital phase in the coprecessing frame.
                    This is $\\varphi(t_0)$ in THE PAPER.
    init_quat:      The initial unit quaternion (length 4 list or numpy array)
                    giving the rotation from the coprecessing frame to the
                    inertial frame.
                    This is $\hat{q}(t_0)$ in THE PAPER.
                    If None (default), uses the identity, in which case the spins
                    in the coprecessing frame are equal to the spins in the
                    inertial frame.
    return_spins:   flag to return the inertial frame time-dependent spins,
                    $\\vec{\chi_{1,2}(t)$.
    t:              The times at which the output should be sampled.
                    The output is interpolated from self.t_coorb using cubic
                    splines. If t=None, returns the results at self.t_coorb.
    theta, phi:     Either specify one or neither. If given, sums up the
                    waveform modes for a gravitational wave propagation
                    direction of (theta, phi) on a sphere centered on the
                    source, where theta is the polar angle and phi is the
                    azimuthal angle.
                    If not given, returns a dictionary of waveform modes h_dict
                    with (ell, m) keys such that (for example) the (2, 2) mode
                    is h_dict[2, 2].
    LMax:           The maximum ell modes to use.
                    The NRSur7dq2 surrogate model contains modes up to L=4.
                    Using LMax=2 or LMax=3 reduces the evaluation time.
    t_ref:
    f_ref:
    t_ref:          The reference (dimensionless) time, where the peak amplitude
                    occurs at t=0.
                    Default: The initial time, t_0/M = -4500.
    f_ref:          A gravitational wave frequency used to determine t_ref,
                    taken to be $\omega / pi$ where $\omega$ is the angular
                    orbital frequency in the coprecessing frame.
                    Specify at most one of t_ref, f_ref.
    MTot:           The total mass of the system, given in solar masses.
                    If given, scales times appropriately, and f_ref should be
                    given in Hz if specified.
                    If t=None, returns the results at
                    self.t_coorb * MTot * SOLAR_TIME_IN_SECONDS
    distance:       The distance to the source, given in megaparsecs.
                    If given, MTot must also be given, and the waveform amplitude
                    will be scaled appropriately.
    use_lalsimulation_conventions: If True, interprets the spin directions and phi
                    using lalsimulation conventions. Specifically, before evaluating
                    the surrogate, the spins will be rotated about the z-axis by
                    init_phase, and pi/2 will be added to phi if it is given.
                    This agrees with lalsimulation's ChooseTDWaveform but not
                    ChooseTDModes; set this to false to agree with ChooseTDModes.
                    This is as of June 2018.
    allow_extrapolation: Enable arbitrary extrapolation of the surrogate in mass
                    ratio and spin magnitude. By default, only allow tiny extrapolations.

Returns:
    h (with return_spins=False)
  or
    h, chiA, chiB (with return_spins=True)

    h: If theta and phi are specified, h is a complex 1d array sampled at times
       t (or self.t_coorb if t=None).
       Otherwise, h is a dictionary with length-2 integer tuples (ell, m) keys,
       and complex 1d arrays giving the (ell, m) mode as values.
    chiA, chiB: The inertial frame spins with shape (N, 3), where N=len(t)
                (or len(self.t_coorb) if t=None).


Examples:

# Load the surrogate
>>> sur = NRSur7dq2.NRSurrogate7dq2('NRSur7dq2.h5')

#Evaluate a waveform with aligned spins
>>> q = 1.5
>>> chiA0 = np.array([0.0, 0.0, 0.5])
>>> chiB0 = np.array([0.0, 0.0, 0.2])
>>> h_mode_dict = sur(q, chiA0, chiB0)

# Plot the results
>>> t = sur.t_coorb
>>> import matplotlib.pyplot as plt
>>> plt.plot(t, np.real(h_mode_dict[2, 2]), label='(2, 2) mode, real part')
>>> plt.plot(t, np.imag(h_mode_dict[2, 2]), label='(2, 2) mode, imag part')
>>> plt.legend()
>>> plt.show()

# Evaluate a precessing waveform, evaluate it on the sphere, and get the spins
>>> t_dense = np.arange(-4500.0, 100.01, 0.1)
>>> chiA0 = np.array([0.8, 0.0, 0.0])
>>> theta, phi = np.pi/3, 0.0
>>> h, chiA, chiB = sur(q, chiA0, chiB0, return_spins=True, t=t_dense,
...                     theta=theta, phi=phi)
>>> plt.plot(t_dense, np.real(h), label='h_+')
>>> plt.plot(t_dense, -1*np.imag(h), label='h_x')
>>> plt.plot(t_dense, chiA[:, 0], label='chiA_x')
>>> plt.legend()
>>> plt.show()
        """
        if use_lalsimulation_conventions:
            # rotate_spin rotates in the -z direction
            chiA0 = rotate_spin(chiA0, -1 * init_phase)
            chiB0 = rotate_spin(chiB0, -1 * init_phase)
            if phi is not None:
                phi += 0.5 * np.pi

        chiA_norm = np.sqrt(np.sum(chiA0**2))
        chiB_norm = np.sqrt(np.sum(chiB0**2))

        # Get dynamics
        if f_ref is None:
            omega_ref = None
        elif MTot is None:
            omega_ref = f_ref * np.pi
        else:
            # Get dimensionless omega_ref
            omega_ref = f_ref * MTot * SOLAR_TIME_IN_SECONDS * np.pi
        quat, orbphase, chiA_copr, chiB_copr = self.get_dynamics(
                q, chiA0, chiB0, init_phase=init_phase, init_quat=init_quat,
                t_ref=t_ref, omega_ref=omega_ref, allow_extrapolation=allow_extrapolation)

        # Interpolate to the coorbital time grid, and transform to coorb frame.
        # Interpolate first since coorbital spins oscillate faster than
        # coprecessing spins
        chiA_copr = splinterp_many(self.tds, self.t_coorb, chiA_copr.T).T
        chiB_copr = splinterp_many(self.tds, self.t_coorb, chiB_copr.T).T
        chiA_copr = normalize_spin(chiA_copr, chiA_norm)
        chiB_copr = normalize_spin(chiB_copr, chiB_norm)
        orbphase = spline(self.tds, orbphase)(self.t_coorb)
        quat = splinterp_many(self.tds, self.t_coorb, quat)
        quat = quat/np.sqrt(np.sum(abs(quat)**2, 0))
        chiA_coorb, chiB_coorb = coorb_spins_from_copr_spins(
                chiA_copr, chiB_copr, orbphase)

        # Evaluate coorbital waveform surrogate
        h_coorb = self.get_coorb_waveform(q, chiA_coorb, chiB_coorb, LMax=LMax,
                                          allow_extrapolation=allow_extrapolation)

        # Transform the sparsely sampled waveform
        h_inertial = inertial_waveform_modes(self.t_coorb, orbphase, quat, h_coorb)

        # Sum up modes if desired
        if ((theta is None) - (phi is None)) != 0:
            raise Exception("Either specify theta and phi, or neither")
        if theta is not None:
            h_inertial = np.array([mode_sum(h_inertial, LMax, theta, phi)])

        # Interpolate to desired times
        if t is not None:
            if MTot is not None:
                t = t / (MTot * SOLAR_TIME_IN_SECONDS)
            # Extrapolating up to 1M in time is not the worst thing in the world
            if t[0] < self.t_0 - 1.0:
                raise Exception("Got t[0]=%s < self.t_0=%s"%(
                        t[0], self.t_0))
            if t[-1] > self.t_f + 1.0:
                raise Exception("Got t[-1]=%s > self.t_f=%s"%(
                        t[-1], self.t_f))
            hre = splinterp_many(self.t_coorb, t, np.real(h_inertial))
            him = splinterp_many(self.t_coorb, t, np.imag(h_inertial))
            h_inertial = hre + 1.j*him

        # Scale waveform if needed
        if distance is not None:
            if MTot is None:
                raise Exception("MTot must be specified if distance is specified")
            h_inertial *= MTot * SOLAR_DISTANCE_IN_MEGAPARSECS / distance

        # Make mode dict if needed
        if theta is not None:
            h = h_inertial[0]
        else:
            h = {}
            i=0
            for ell in range(2, LMax+1):
                for m in range(-ell, ell+1):
                    h[ell, m] = h_inertial[i]
                    i+=1

        #  Transform and interpolate spins if needed
        if return_spins:
            chiA_inertial = transformTimeDependentVector(quat, chiA_copr.T).T
            chiB_inertial = transformTimeDependentVector(quat, chiB_copr.T).T
            if t is not None:
                chiA_inertial = splinterp_many(self.t_coorb, t, chiA_inertial.T).T
                chiB_inertial = splinterp_many(self.t_coorb, t, chiB_inertial.T).T
                chiA_inertial = normalize_spin(chiA_inertial, chiA_norm)
                chiB_inertial = normalize_spin(chiB_inertial, chiB_norm)
            return h, chiA_inertial, chiB_inertial
        return h

#-----------------------------------------------------------------------------
