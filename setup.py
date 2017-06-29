#!/usr/bin/env python3

from distutils.core import setup

setup(
    name='Ramos',
    version='0.1.0',
    description='Tools for reduced bases',
    author='Eivind Fonn',
    author_email='eivind.fonn@sintef.no',
    license='GPL3',
    url='https://github.com/TheBB/gmesh',
    packages=['gmesh', 'gmesh.utm', 'ramos'],
    install_requires=['click', 'numpy', 'quadpy'],
    entry_points={
        'console_scripts': [
            'gmesh=gmesh.__main__:main',
            'ramos=ramos.__main__:main',
        ],
    },
)
