#!/usr/bin/env python

from setuptools import setup
from setuptools import Extension
import numpy

short_desc = "Package for evaluating the NRSur7dq2 surrogate model"
long_desc = \
"""
NRSur7dq2 is a surrogate model for gravitational waves from numerical
relativity simulations of binary black hole mergers.
It is described in Blackman et al. 2017:
https://arxiv.org/abs/1705.07089
https://journals.aps.org/prd/abstract/10.1103/PhysRevD.96.024058

This package provides a class NRSurrogate7dq2 for evaluating the NRSur7dq2
surrogate model.

See the NRSurrogate7dq2 class docstring for usage.
A tutorial Jupyter notebook can be found at
https://www.black-holes.org/surrogates/
"""

extensions = [
    Extension(
                '_NRSur7dq2_utils',
                sources=['NRSur7dq2_utils/src/NRSur7dq2_utils.c'],
                include_dirs = ['NRSur7dq2_utils/include', numpy.get_include()],
                language='c',
                extra_compile_args = ['-fPIC', '-O3'],
            )
        ]

setup(
        name            = 'NRSur7dq2',
        version         = '1.0.6',
        description     = short_desc,
        long_description = long_desc,
        author          = 'Jonathan Blackman',
        author_email    = 'jonathan.blackman.0@gmail.com',
        url             = 'https://www.black-holes.org/surrogates/',
        packages        = ['NRSur7dq2'],
        package_data    = {'NRSur7dq2': ['NRSur7dq2.h5']},
        ext_modules     = extensions,
        install_requires = ['numpy', 'scipy', 'h5py'],
    )
