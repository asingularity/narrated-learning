
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "cython_dist_helper",
        ["cython_dist_helper.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='cython-dist-helper',
    ext_modules=cythonize(ext_modules),
)
