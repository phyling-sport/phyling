import numpy
from Cython.Build import cythonize
from setuptools import Extension
from setuptools import find_packages
from setuptools import setup


def load_requirements(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="phyling",
    version="6.6.6",
    description="Phyling public package",
    packages=find_packages(),
    install_requires=load_requirements("requirements.txt"),
    ext_modules=cythonize(
        [
            Extension(
                "phyling.decoder.decoder_utils",
                ["phyling/decoder/decoder_utils.pyx"],
                include_dirs=[numpy.get_include()],
            )
        ]
    ),
    url="https://github.com/phyling-sport/phyling",
)
