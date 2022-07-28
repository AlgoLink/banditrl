import os
from setuptools import setup

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
# read the docs could not compile numpy and c extensions
if on_rtd:
    setup_requires = []
    install_requires = []
else:
    setup_requires = [
        'nose',
        'coverage',
    ]
    install_requires = [
        'six',
        'numpy',
        'scipy',
        'matplotlib',
    ]

long_description = ("See `github <https://github.com/algolink/banditrl>`_ "
                    "for more information.")

setup(
    name='banditrl',
    version='0.0.1',
    description='Contextual bandit in python',
    long_description=long_description,
    author='lee',
    author_email='lee@163.com',
    url='https://github.com/algolink/banditrl',
    setup_requires=setup_requires,
    install_requires=install_requires,
    classifiers=[
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
    test_suite='nose.collector',
    packages=[
        'banditrl',
        'banditrl.bandit',
        'banditrl.preprocessing',
        'banditrl.storage',
        'banditrl.utils',
    ],
    package_dir={
        'banditrl': 'banditrl',
        'banditrl.bandit': 'banditrl/bandit',
        'banditrl.storage': 'banditrl/storage',
        'banditrl.utils': 'banditrl/utils',
        'banditrl.preprocessing':'banditrl/preprocessing'
    },
)