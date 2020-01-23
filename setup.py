from setuptools import setup

setup(
    name='pwlf',
    version='1.1.6',
    author='Charles Jekel',
    author_email='cjekel@gmail.com',
    packages=['pwlf'],
    url='https://github.com/cjekel/piecewise_linear_fit_py',
    license='MIT License',
    description='fit piecewise linear functions to data',
    long_description=open('README.rst').read(),
    platforms=['any'],
    install_requires=[
        "numpy >= 1.14.0",
        "scipy >= 0.19.0",
        "pyDOE >= 0.3.8",
        "setuptools >= 38.6.0",
    ],
    extras_require={
        'PiecewiseLinFitTF':  ["tensorflow < 2.0.0"]
    }
)
