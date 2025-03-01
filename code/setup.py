# setup.py

import os
import argparse
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

from pathlib import Path

import numpy
from Cython.Build import cythonize

# Set compiler flags
os.environ['CFLAGS'] = '-O3 -march=native -mveclibabi=svml -flto -fPIC'

cmake_install_prefix = os.environ.get("CMAKE_INSTALL_PREFIX") or "/usr/local"
print(f"Assuming libsbdt.so at {cmake_install_prefix}.")


# Define the extension module
extensions = [
    Extension(
        "sbdt",
        sources=["./sbdt.pyx"],
        extra_link_args=[f"{cmake_install_prefix}/lib/libsbdt.so"],
        library_dirs=[f"{cmake_install_prefix}"],
        include_dirs=[
            f"{cmake_install_prefix}/include/sbdt",
            f"{cmake_install_prefix}/include/sbdt/gbdt",
            numpy.get_include(),
        ],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

# Setup configuration
setup(
    name="sbdt",
    ext_modules=cythonize(extensions, language_level=3),
)