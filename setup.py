# -*- coding: utf-8 -*-
#author: Xuan Zhou 

import os
from setuptools import setup,find_packages
from palframe import about
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8').read()

py_modules = [
    f'palframe.{name.replace(".py","")}' for name in os.listdir('palframe')
    if name.endswith('.py') and name not in ['__init__.py']
]

setup(
    name='palframe',
    version=about.__version__,
    author=about.__author__,
    author_email=about.__email__,
    description='palframe',

    packages=[f'palframe.{p}' for p in find_packages('palframe')],
    package_dir={'palframe': 'palframe'},
    py_modules=py_modules,
    include_package_data=True,

    long_description=read('README.md'),
    install_requires=open('requirements.txt').readlines(),
    license='MIT',
    python_requires='>=3.7',
)


if __name__ == '__main__':
    pass
