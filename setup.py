import numpy
from Cython.Build import cythonize
from setuptools import Extension
from setuptools import setup

setup(
    ext_modules=cythonize(
        [
            Extension(
                "phyling.decoder.decoder_utils",
                ["phyling/decoder/decoder_utils.pyx"],
                include_dirs=[numpy.get_include()],
            )
        ]
    )
)
