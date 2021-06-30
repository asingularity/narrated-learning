
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "cython_tiled_wta",
        ["cython_tiled_wta.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='cython_tiled_wta',
    ext_modules=cythonize(ext_modules),
)

ext_modules = [
    Extension(
        "cython_layers_viz",
        ["cython_layers_viz.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='cython_layers_viz',
    ext_modules=cythonize(ext_modules),
)
