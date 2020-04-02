import io
from setuptools import setup

setup(
    name='pwlf',
    version='2.0.0',
    author='Charles Jekel',
    author_email='cjekel@gmail.com',
    packages=['pwlf'],
    url='https://github.com/cjekel/piecewise_linear_fit_py',
    license='MIT License',
    description='fit piecewise linear functions to data',
    long_description=io.open('README.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    platforms=['any'],
    install_requires=[
        "numpy >= 1.14.0",
        "scipy >= 1.2.0",
        "pyDOE >= 0.3.8",
        "setuptools >= 38.6.0",
    ],
)
