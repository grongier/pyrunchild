"""Setup for the pyrunchild package

See:
https://github.com/pypa/sampleproject/blob/master/setup.py
"""

# LICENCE GOES HERE

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyrunchild',
    version='v0.0.1',
    description='Python package for running the landscape evolution model CHILD',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    author='Guillaume Rongier',
    license='CSIRO BSD / MIT licence',
    classifiers=[
        # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3',      
    ],
    keywords='landscape evolution model CHILD',
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=['numpy', 'scipy', 'alphashape', 'rasterio', 'matplotlib'],
)
