from distutils.core import setup
setup(
    name='pwlf',
    version=open('pwlf/VERSION').read().strip(),
    author='Charles Jekel',
    author_email='cjekel@gmail.com',
    packages=['pwlf'],
    package_data={'pwlf': ['VERSION']},
    url='https://github.com/cjekel/piecewise_linear_fit_py',
    license='MIT License',
    description='fit piece-wise linear function to data',
    long_description=open('README.rst').read(),
    platforms=['any'],
    install_requires=[
        "numpy >= 1.14.0",
        "scipy >= 0.19.0",
        "pyDOE >= 0.3.8",
    ],
)
