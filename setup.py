from setuptools import Extension
from setuptools import find_packages
from setuptools import setup

ext_modules = [
    Extension("phyling.decoder.decoder_utils", ["phyling/decoder/decoder_utils.c"]),
]

setup(
    name="phyling",
    version="6.6.6",
    packages=find_packages(),  # ["phyling", "phyling.decoder", "phyling.api", "phyling.ble"],
    ext_modules=ext_modules,
    install_requires=["numpy", "pandas", "bleak"],
    description="Phyling public package",
    url="https://github.com/phyling-sport/phyling",
)
