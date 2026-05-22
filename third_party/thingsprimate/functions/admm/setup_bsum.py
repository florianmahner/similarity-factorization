#!/usr/bin/env python3
"""Simple setup script to compile bsum.pyx"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "bsum",
        sources=["bsum.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': 3}),
    zip_safe=False,
) 