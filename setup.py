#!/usr/bin/env python3

from distutils.core import setup

setup(
    name='Ramos',
    version='0.1.0',
    description='Tools for reduced bases',
    author='Eivind Fonn',
    author_email='eivind.fonn@sintef.no',
    license='GPL3',
    url='https://github.com/TheBB/ramos',
    packages=['ramos'],
    install_requires=['click', 'numpy', 'scipy', 'quadpy', 'tqdm', 'vtk'],
    extras_require={
        'IFEM': ['splipy', 'lxml', 'h5py'],
    },
    entry_points={
        'console_scripts': [
            'ramos=ramos.__main__:main',
        ],
    },
)
