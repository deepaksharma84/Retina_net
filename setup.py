import setuptools
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        'utils.compute_overlap',
        ['utils/compute_overlap.pyx'],
        include_dirs=[np.get_include()]
    ),
]

setuptools.setup(
    ext_modules=cythonize(extensions)
)
