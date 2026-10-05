from Cython.Build import cythonize
from setuptools import Extension
from setuptools import setup

setup(
    ext_modules=cythonize(
        [
            Extension(
                "phyling.decoder.decoder_utils",
                ["phyling/decoder/decoder_utils.pyx"],
            )
        ],
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "language_level": "3",
        },
    )
)
