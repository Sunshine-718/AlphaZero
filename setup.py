from setuptools import setup
from Cython.Build import cythonize
import numpy
import shutil
import os

setup(
    ext_modules=cythonize(
        ["env_cython.pyx"],
        compiler_directives={"language_level": "3"}
    ),
    include_dirs=[numpy.get_include()],
)

for filename in os.listdir():
    if 'env_cython' in filename and ('.pyd' in filename or '.so' in filename):
        shutil.move(filename, f'./environments/Connect4/{filename}')
    elif '.cpp' in filename:
        os.remove(filename)
shutil.rmtree('./build')
