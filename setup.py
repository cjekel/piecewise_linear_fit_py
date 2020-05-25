import io
from distutils.core import setup

setup(
    name='pwlf',
    version='2.0.1',
    author='Charles Jekel',
    author_email='cjekel@gmail.com',
    packages=['pwlf'],
    url='https://github.com/cjekel/piecewise_linear_fit_py',
    license='MIT License',
    description='fit piecewise linear functions to data',
    long_description=io.open('README.rst').read(),
    # long_description_content_type='text/markdown',
    platforms=['any'],
    install_requires=[
        "numpy >= 1.14.0",
        "scipy >= 1.2.0",
        "pyDOE >= 0.3.8",
        'importlib-metadata ~= 1.0 ; python_version < "3.8"',
    ],
)
