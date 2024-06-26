"""  Created on 26/06/2024::
------------- setup.py -------------
 
**Authors**: L. Mingarelli
"""
import os
import setuptools


with open("README.md", "r") as f:
    long_description = f.read()

about = {}
with open("compnet/__about__.py") as f:
    exec(f.read(), about)


with open("compnet/requirements.txt") as f:
    install_requirements = f.read().splitlines()

setuptools.setup(
    name="compnet",
    version=about['__version__'],
    author=about['__author__'],
    author_email=about['__email__'],
    description=about['__about__'],
    url=about['__url__'],
    license='MIT',
    long_description=long_description,
    long_description_content_type="markdown",
    packages=setuptools.find_packages() + ['compnet.res'],
    include_package_data=True,
    package_data={'':  []},
    install_requires=install_requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

