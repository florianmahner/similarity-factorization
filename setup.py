"""Setup script for pysrf with Cython extension compilation."""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy as np


class BuildExtWithFallback(build_ext):
    """Build extension with graceful fallback if compilation fails."""

    def build_extensions(self):
        try:
            super().build_extensions()
        except Exception as e:
            print(f"Warning: Cython extension compilation failed: {e}")
            print("pysrf will use the slower Python fallback implementation.")
            self.extensions = []


def get_extensions():
    """Get Cython extensions if Cython is available."""
    try:
        from Cython.Build import cythonize

        extensions = [
            Extension(
                "pysrf._bsum",
                sources=["pysrf/_bsum.pyx"],
                include_dirs=[np.get_include()],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                extra_compile_args=["-O3", "-march=native", "-ffp-contract=off"],
            ),
        ]
        return cythonize(extensions, compiler_directives={"language_level": "3"})
    except ImportError:
        print("Cython not available, skipping extension compilation")
        return []


setup(
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtWithFallback},
)
