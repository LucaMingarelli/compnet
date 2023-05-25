"""  Created on 26/05/2023::
------------- setup.py -------------

**Authors**: L. Mingarelli
"""

import setuptools
import compnet as cn

with open("README.md", 'r') as f:
    long_description = f.read()

with open("compnet/requirements.txt") as f:
    install_requirements = f.read().splitlines()


setuptools.setup(
    name="compnet",
    version=cn.__version__,
    author=cn.__author__,
    author_email=cn.__email__,
    description=cn.__about__,
    url=cn.__url__,
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['compnet', 'compnet.tests'],
    package_data={'':  ['../bindata/res/*']},
    install_requires=install_requirements,
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent"],
    python_requires='>=3.6',
)





