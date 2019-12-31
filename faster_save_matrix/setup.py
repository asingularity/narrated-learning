
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "fast_save_matrix",
        ["fast_save_matrix.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='fast-save-matrix',
    ext_modules=cythonize(ext_modules),
)
