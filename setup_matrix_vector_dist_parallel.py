
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "matrix_vector_dist_parallel",
        ["matrix_vector_dist_parallel.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='matrix-vector-dist-parallel',
    ext_modules=cythonize(ext_modules),
)
