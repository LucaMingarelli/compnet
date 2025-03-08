"""  Created on 26/05/2023::
------------- setup.py -------------

**Authors**: L. Mingarelli
"""

import setuptools
from compnet.__about__ import (__about__, __author__, __email__, __version__, __url__)

with open("README.md", 'r') as f:
    long_description = f.read()

with open("compnet/requirements.txt") as f:
    install_requirements = f.read().splitlines()


setuptools.setup(
    name="compnet",
    version=__version__,
    author=__author__,
    author_email=__email__,
    description=__about__,
    url=__url__,
    license='CC BY-NC-SA 4.0',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['compnet', 'compnet.tests'],
    package_data={'':  ['../compnet/res/*']},
    install_requires=install_requirements,
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent"],
    python_requires='>=3.6',
)




