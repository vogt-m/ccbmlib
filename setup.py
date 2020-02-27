#   Copyright 2020 Martin Vogt
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
#  associated documentation files (the "Software"), to deal in the Software without restriction,
#  including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do
#  so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial
#  portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#  PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
#  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
#  WITH  THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from setuptools import setup

import re
version = ""
with open("ccbmlib/__init__.py") as fin:
    for line in fin:
        version_match = re.search(r"^__version__ *= *['\"]([^'\"]*)['\"]",line)
        if version_match:
            version = version_match.group(1)

with open("README.md") as f:
    long_description = f.read()

setup(
    name='ccbmlib',
    version=version,
    packages=['ccbmlib'],
    url='',
    license='MIT License',
    author='Martin Vogt',
    author_email='martin.vogt@bit.uni-bonn.de',
    description='Tanimoto distribution models for RDKit',
    long_description=long_description
)
