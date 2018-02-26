#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "NRSur7dq2_utils.h"
#include <math.h>
#include <stdio.h>

/*
 * Created in 2017 by Jonathan Blackman
 *
 * This is a utilities Python module for use by NRSur7dq2.py that performs
 * tasks that would be slow in Python.
 * NRSur7dq2.py evaluates the NRSur7dq2 surrogate model of gravitational waves
 * from numerical relativity simulations.
 * See Blackman et al. 2017 for details:
 * https://arxiv.org/abs/1705.07089
 * https://journals.aps.org/prd/abstract/10.1103/PhysRevD.96.024058
 * 
 * You are free to redistribute and/or modify this software as needed.
 *
 * For help with C extensions for Python including NumPy, see:
 * http://scipy-cookbook.readthedocs.io/items/C_Extensions_NumPy_arrays.html
 *
 * The NumPy C-API can be found at:
 * https://docs.scipy.org/doc/numpy/reference/c-api.html
 */

// These are to rescale the mass ratio fit range from [0.99, 2.01] to [-1, 1].
const double Q_FIT_OFFSET = -2.9411764705882355;
const double Q_FIT_SLOPE = 1.9607843137254901;


/* ==== Setup the python methods table === */
static PyMethodDef _NRSur7dq2_utilsMethods[] = {
    {"eval_fit", eval_fit, METH_VARARGS},
    {"eval_vector_fit", eval_vector_fit, METH_VARARGS},
    {"normalize_y", normalize_y, METH_VARARGS},
    {"get_ds_fit_x", get_ds_fit_x, METH_VARARGS},
    {"assemble_dydt", assemble_dydt, METH_VARARGS},
    {"ab4_dy", ab4_dy, METH_VARARGS},
    {"binom", binom, METH_VARARGS},
    {"wigner_coef", wigner_coef, METH_VARARGS},
    {NULL, NULL} /* Marks the end of this structure */
};

/* Initialize c_test functions */
void init_NRSur7dq2_utils(void) {
    (void) Py_InitModule("_NRSur7dq2_utils", _NRSur7dq2_utilsMethods);
    import_array(); // For numpy
}

double ipow(double base, long exponent) {
    if (exponent == 0) return 1.0;
    double res = base;
    while (exponent > 1) {
        res = res*base;
        exponent -= 1;
    }
    return res;
}

/*
 * This function evaluates a parametric fit.
 * Arguments (with python data types):
 *      bf_orders:  A 2d integer numpy array with shape (n_coefs, 7).
 *                  Gives the basis function orders for each coefficient
 *                  and each parameter.
 *      coefs:      A 1d float numpy array with length n_coefs.
 *                  Contains the fit coefficients.
 *      x: i        A 1d float numpy array with length 7.
 *                  Gives the parameters at which the fit should be evaluated.
 * Computes the fit evaluation by summing up coefficients multiplied by 7 basis
 * functions, each evaluated at one component of x.
 * Returns a python float giving the fit evaluation.
 */
static PyObject *eval_fit(PyObject *self, PyObject *args) {

    PyArrayObject *bf_orders, *coefs, *x;
    int i, j, n;
    long *bf_order_data, *orders;
    double res, prod, *coef_data, *x_data;
    double x_powers[22];

    // Parse tuples
    if (!PyArg_ParseTuple(args, "O!O!O!",
            &PyArray_Type, &bf_orders,
            &PyArray_Type, &coefs,
            &PyArray_Type, &x)) return NULL;

    // Point to numpy array data
    bf_order_data = (long *) PyArray_DATA(bf_orders);
    coef_data = (double *) PyArray_DATA(coefs);
    x_data = (double *) PyArray_DATA(x);
    n = PyArray_DIMS(coefs)[0];
    res = 0.0;

    // Compute all needed powers
    for (i=0; i<22; i++) {
        if (i%7==0) {
            x_powers[i] = ipow(Q_FIT_OFFSET + Q_FIT_SLOPE*x_data[0], i/7);
        } else {
            x_powers[i] = ipow(x_data[i%7], i/7);
        }
    }

    // Sum up result
    for (i=0; i<n; i++) {
        orders = bf_order_data + i*7;
        prod = x_powers[7*orders[0]];
        for (j=1; j<7; j++) {
            prod *= x_powers[7*orders[j] + j];
        }
        res += coef_data[i]*prod;
    }

    return Py_BuildValue("d", res);
}

/* This function is similar to eval_fit, but it is evaluating fits for the
 * components of a vector function.
 * In this case, each term is also multiplied by a unit vector.
 * Arguments (with python data types):
 *      vec_dim:    An integer giving the dimension of the vector function.
 *                  Should be 2 or 3.
 *      bf_orders:  A 2d integer numpy array with shape (n_coefs, 7).
 *                  Gives the basis function orders for each coefficient
 *                  and each parameter.
 *      coefs:      A 1d float numpy array with length n_coefs.
 *                  Contains the fit coefficients.
 *      bvec_indices: A 1d integer numpy array with length n_coefs.
 *                  Each index i has 0 <= i < vec_dim, and specifies the
 *                  component of the vector function to which this term should
 *                  be added.
 *      x: i        A 1d float numpy array with length 7.
 *                  Gives the parameters at which the fit should be evaluated.
 * Computes the fit evaluation by summing up coefficients multiplied by 7 basis
 * functions and a unit vector, each evaluated at one component of x.
 * Returns a 1d float numpy vector with length vec_dim giving the fit evaluation.
 */
static PyObject *eval_vector_fit(PyObject *self, PyObject *args) {

    PyArrayObject *res, *bf_orders, *coefs, *x, *bvec_indices;
    int i, j, n, vec_dim;
    long *bf_order_data, *orders, *bvec_indices_data;
    double *res_data, prod, *coef_data, *x_data;
    npy_intp dims[1];
    double x_powers[22]; // Store up to cubic in mass ratio, quadratic in spins

    // Parse tuples
    if (!PyArg_ParseTuple(args, "iO!O!O!O!",
            &vec_dim,
            &PyArray_Type, &bf_orders,
            &PyArray_Type, &coefs,
            &PyArray_Type, &bvec_indices,
            &PyArray_Type, &x)) return NULL;

    // Initialize output array
    dims[0] = vec_dim;
    res = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    res_data = (double *) PyArray_DATA(res);
    for (i=0; i<vec_dim; i++) {
        res_data[i] = 0.0;
    }

    // Point to numpy array data
    bf_order_data = (long *) PyArray_DATA(bf_orders);
    coef_data = (double *) PyArray_DATA(coefs);
    x_data = (double *) PyArray_DATA(x);
    bvec_indices_data = (long *) PyArray_DATA(bvec_indices);

    // Compute all needed powers
    for (i=0; i<22; i++) {
        if (i%7==0) {
            x_powers[i] = ipow(Q_FIT_OFFSET + Q_FIT_SLOPE*x_data[0], i/7);
        } else {
            x_powers[i] = ipow(x_data[i%7], i/7);
        }
    }

    n = PyArray_DIMS(coefs)[0];

    // Sum up terms
    for (i=0; i<n; i++) {
        orders = bf_order_data + i*7;
        prod = x_powers[7*orders[0]];
        for (j=1; j<7; j++) {
            prod *= x_powers[7*orders[j] + j];
        }
        res_data[bvec_indices_data[i]] += coef_data[i]*prod;
    }

    return PyArray_Return(res);
}

/*
 * This is a helper function to normalize unit quaternions and spin magnitudes
 * at each time step during the ODE integration.
 * Arguments (with python data types):
 *      y:      A 1d float numpy array with length 11, containing:
 *                  y[0:4] is the quaternion
 *                  y[4] is the orbital phase
 *                  y[5:8] is chiA
 *                  y[8:11] is chiB
 *      normA:  A python float giving |chiA|
 *      normB:  A python float giving |chiB|
 * Returns a python float array giving the normalized version of y.
 */
static PyObject *normalize_y(PyObject *self, PyObject *args) {

    PyArrayObject *y, *res;
    int i;
    double *y_data, *res_data, normA, normB, nA, nB, quatNorm, sum;
    npy_intp dims[1];

    // Parse tuples
    if (!PyArg_ParseTuple(args, "O!dd",
            &PyArray_Type, &y,
            &normA,
            &normB)) return NULL;

    // Initialize output array
    dims[0] = 11;
    res = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    res_data = (double *) PyArray_DATA(res);

    y_data = (double *) PyArray_DATA(y);

    // Compute current norms
    sum = 0.0;
    for (i=0; i<4; i++) {
        sum += y_data[i]*y_data[i];
    }
    quatNorm = sqrt(sum);
    sum = 0.0;
    for (i=5; i<8; i++) { // Yes, i=5, not i=4. i=4 is the orbital phase.
        sum += y_data[i]*y_data[i];
    }
    nA = sqrt(sum);
    sum = 0.0;
    for (i=8; i<11; i++) {
        sum += y_data[i]*y_data[i];
    }
    nB = sqrt(sum);

    // Normalize output
    for (i=0; i<4; i++) {
        res_data[i] = y_data[i] / quatNorm;
    }
    res_data[4] = y_data[4];
    for (i=5; i<8; i++) {
        res_data[i] = y_data[i] * normA / nA;
    }
    for (i=8; i<11; i++) {
        res_data[i] = y_data[i] * normB / nB;
    }

    return PyArray_Return(res);
}

/*
 * This function computes the fit input for the dynamics surrogate.
 * Arguments (with python data types):
 *      y:      A 1d float numpy array with length 11, containing:
 *                  y[0:4] is the quaternion
 *                  y[4] is the orbital phase
 *                  y[5:8] is chiA in the coprecessing frame
 *                  y[8:11] is chiB in the coprecessing frame
 *      q:     A python float giving the mass ratio q
 * Returns a python float array x with length 7 containing:
 *                  x[0]: q
 *                  x[1:4]: chiA in the coorbital frame
 *                  x[4:7]: chiB in the coorbital frame
 */

static PyObject *get_ds_fit_x(PyObject *self, PyObject *args) {

    PyArrayObject *y, *x;
    double *y_data, *x_data, q, sp, cp;
    npy_intp dims[1];

    // Parse tuples
    if (!PyArg_ParseTuple(args, "O!d", &PyArray_Type, &y, &q)) return NULL;

    y_data = (double *) PyArray_DATA(y);
    dims[0] = 7;
    x = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    x_data = (double *) PyArray_DATA(x);

    // q
    x_data[0] = q;

    // chiA
    sp = sin(y_data[4]);
    cp = cos(y_data[4]);
    x_data[1] = y_data[5]*cp + y_data[6]*sp;
    x_data[2] = -1*y_data[5]*sp + y_data[6]*cp;
    x_data[3] = y_data[7];

    // chiB
    x_data[4] = y_data[8]*cp + y_data[9]*sp;
    x_data[5] = -1*y_data[8]*sp + y_data[9]*cp;
    x_data[6] = y_data[10];

    return PyArray_Return(x);
}

/*
 * This function assembles the right-hand-side of the dynamics ODE integration
 * Arguments (with python data types):
 *      y:      A 1d float numpy array with length 11, containing:
 *                  y[0:4] is the quaternion
 *                  y[4] is the orbital phase
 *                  y[5:8] is chiA in the coprecessing frame
 *                  y[8:11] is chiB in the coprecessing frame
 *      ooxy:   A length 2 float numpy array with the x and y components of
 *              \Omega^\mathrm{coorb}
 *      omega:  A python float giving the orbital frequency
 *      cAdot:  A length 3 float numpy array with the time derivative of the
 *              coprecessing frame chiA, with components in the coorbital frame.
 *      cBdot:  A length 3 float numpy array with the time derivative of the
 *              coprecessing frame chiB, with components in the coorbital frame.
 * Returns a python float array dydt with length 11 containing the time
 * derivative of y
 */
static PyObject *assemble_dydt(PyObject *self, PyObject *args) {

    PyArrayObject *y, *ooxy, *cAdot, *cBdot, *dydt;
    double *y_data, *ooxy_data, *cAdot_data, *cBdot_data, *dydt_data;
    double omega, sp, cp, ooxy_x, ooxy_y;
    npy_intp dims[1];

    // Parse tuples
    if (!PyArg_ParseTuple(args, "O!O!dO!O!",
            &PyArray_Type, &y,      // length 11, has quat, orbphase, chiA_copr, chiB_copr
            &PyArray_Type, &ooxy,   // length 2 with x and y components of \Omega^{copr}
            &omega,                 // double, orbital frequency \omega
            &PyArray_Type, &cAdot,  // length 3, coprecessing chiA time derivative
            &PyArray_Type, &cBdot)) return NULL;

    // Initialize output array
    dims[0] = 11;
    dydt = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    // Point to numpy array data
    y_data = (double *) PyArray_DATA(y);
    ooxy_data = (double *) PyArray_DATA(ooxy);
    cAdot_data = (double *) PyArray_DATA(cAdot);
    cBdot_data = (double *) PyArray_DATA(cBdot);
    dydt_data = (double *) PyArray_DATA(dydt);

    // Quaternion derivative
    // Omega = 2 * quat^{-1} * dqdt -> dqdt = 0.5 * quat * ooxy_quat where
    // ooxy_quat = [0, ooxy_copr_x, ooxy_copr_y, 0]
    cp = cos(y_data[4]);
    sp = sin(y_data[4]);
    ooxy_x = ooxy_data[0]*cp - ooxy_data[1]*sp;
    ooxy_y = ooxy_data[0]*sp + ooxy_data[1]*cp;
    dydt_data[0] = (-0.5)*y_data[1]*ooxy_x - 0.5*y_data[2]*ooxy_y;
    dydt_data[1] = (-0.5)*y_data[3]*ooxy_y + 0.5*y_data[0]*ooxy_x;
    dydt_data[2] = 0.5*y_data[3]*ooxy_x + 0.5*y_data[0]*ooxy_y;
    dydt_data[3] = 0.5*y_data[1]*ooxy_y - 0.5*y_data[2]*ooxy_x;

    // Orbital phase derivative
    dydt_data[4] = omega;

    // Spin derivatives need to be rotated to the coprecessing frame
    dydt_data[5] = cAdot_data[0]*cp - cAdot_data[1]*sp;
    dydt_data[6] = cAdot_data[0]*sp + cAdot_data[1]*cp;
    dydt_data[7] = cAdot_data[2];
    dydt_data[8] = cBdot_data[0]*cp - cBdot_data[1]*sp;
    dydt_data[9] = cBdot_data[0]*sp + cBdot_data[1]*cp;
    dydt_data[10] = cBdot_data[2];

    return PyArray_Return(dydt);
}

/*
 * A helper function computing the update to y using the Adams-Bashforth
 * 4th-order ODE integration scheme.
 */
static PyObject *ab4_dy(PyObject *self, PyObject *args) {

    PyArrayObject *k1, *k2, *k3, *k4, *res;
    double *k1_data, *k2_data, *k3_data, *k4_data, *res_data;
    double dt1, dt2, dt3, dt4, dt12, dt123, dt23, D1, D2, D3,
            A, B, C, D, B41, B42, B43, B4, C41, C42, C43, C4;
    int i;
    npy_intp dims[1];

    // Parse tuples
    if (!PyArg_ParseTuple(args, "O!O!O!O!dddd",
            &PyArray_Type, &k1,
            &PyArray_Type, &k2,
            &PyArray_Type, &k3,
            &PyArray_Type, &k4,
            &dt1, &dt2, &dt3, &dt4)) return NULL;

    // Initialize output
    dims[0] = 11;
    res = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    // Point to numpy array data
    k1_data = (double *) PyArray_DATA(k1);
    k2_data = (double *) PyArray_DATA(k2);
    k3_data = (double *) PyArray_DATA(k3);
    k4_data = (double *) PyArray_DATA(k4);
    res_data = (double *) PyArray_DATA(res);

    // Various time intervals
    dt12 = dt1 + dt2;
    dt123 = dt12 + dt3;
    dt23 = dt2 + dt3;

    // Denomenators and coefficients
    D1 = dt1 * dt12 * dt123;
    D2 = dt1 * dt2 * dt23;
    D3 = dt2 * dt12 * dt3;

    B41 = dt3 * dt23 / D1;
    B42 = -1 * dt3 * dt123 / D2;
    B43 = dt23 * dt123 / D3;
    B4 = B41 + B42 + B43;

    C41 = (dt23 + dt3) / D1;
    C42 = -1 * (dt123 + dt3) / D2;
    C43 = (dt123 + dt23) / D3;
    C4 = C41 + C42 + C43;

    // Polynomial coefficients and result
    for (i=0; i<11; i++) {
        A = k4_data[i];
        B = k4_data[i]*B4 - k1_data[i]*B41 - k2_data[i]*B42 - k3_data[i]*B43;
        C = k4_data[i]*C4 - k1_data[i]*C41 - k2_data[i]*C42 - k3_data[i]*C43;
        D = (k4_data[i]-k1_data[i])/D1 - (k4_data[i]-k2_data[i])/D2 + (k4_data[i]-k3_data[i])/D3;
        res_data[i] = dt4 * (A + dt4 * (0.5*B + dt4*( C/3.0 + dt4*0.25*D)));
    }

    // Sum up contributions
    return PyArray_Return(res);
}

double factorial(int n) {
    if (n <= 0) return 1.0;
    return factorial(n-1) * n;
}

double factorial_ratio(int n, int k) {
    if (n <= k) return 1.0;
    return factorial_ratio(n-1, k) * n;
}

double _binomial(int n, int k) {
    return factorial_ratio(n, k) / factorial(n-k);
}

double _wigner_coef(int ell, int mp, int m) {
    return sqrt(factorial(ell+m) * factorial(ell-m) / (factorial(ell+mp) * factorial(ell-mp)));
}

static PyObject *binom(PyObject *self, PyObject *args) {
    int n, k;
    double b;

    // Parse tuples
    if (!PyArg_ParseTuple(args, "ii", &n, &k)) return NULL;

    // Compute result
    b = _binomial(n, k);

    // Return result
    return Py_BuildValue("d", b);
}

static PyObject *wigner_coef(PyObject *self, PyObject *args) {
    int ell, mp, m;
    double wc;

    // Parse tuples
    if (!PyArg_ParseTuple(args, "iii", &ell, &mp, &m)) return NULL;

    // Compute result
    wc = _wigner_coef(ell, mp, m);

    // Return result
    return Py_BuildValue("d", wc);
}   
