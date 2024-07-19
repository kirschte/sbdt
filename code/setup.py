# setup.py

import os
from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

os.environ['CFLAGS'] = '-O3 -march=native -mveclibabi=svml -flto -fPIC'

setup(
    name="sbdt",
    ext_modules=cythonize(
        [
            Extension(
                "sbdt",
                ["sbdt.pyx"],
                extra_link_args=["./cpp_gbdt/libsbdt.a"],
                library_dirs=["./cpp_gbdt/"],
                include_dirs=[
                    "./cpp_gbdt/include/",
                    "./cpp_gbdt/include/gbdt/",
                    numpy.get_include(),
                ],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            )
        ],
        language_level=3,
        #gdb_debug=True,
    ),
)

# to build the extension, run `python3 setup.py build_ext -i`.
