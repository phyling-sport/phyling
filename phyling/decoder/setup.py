from Cython.Build import cythonize
from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension

# build
# python setup.py build_ext --inplace

ext_modules = [
    Extension(
        "decoder_utils",
        ["decoder_utils.pyx"],
    ),
]

setup(
    name="cython module",
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(ext_modules, compiler_directives={"language_level": "3"}),
)
