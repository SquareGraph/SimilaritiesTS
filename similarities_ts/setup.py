#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['torch','faiss-cpu','faiss-gpu']

test_requirements = [ ]

setup(
    author="Marcin Dąbrowski",
    author_email='mrcndabrowski@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.10',
    ],
    description="A python package for reducing dimensionalities and fininding similarities in latent representations of multiple multivariate time series",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='similarities_ts',
    name='similarities_ts',
    packages=find_packages(include=['similarities_ts', 'similarities_ts.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/SquareGraph9045/similarities_ts',
    version='0.1.0',
    zip_safe=False,
)
