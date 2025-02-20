import io
from distutils.core import setup

# load the version from version.py
version = {}
with open("pwlf/version.py") as fp:
    exec(fp.read(), version)

setup(
    name="pwlf",
    version=version["__version__"],
    author="Charles Jekel",
    author_email="cjekel@gmail.com",
    packages=["pwlf"],
    url="https://github.com/cjekel/piecewise_linear_fit_py",
    license="MIT License",
    description="fit piecewise linear functions to data",
    long_description=io.open("README.rst", encoding="utf-8").read(),
    # long_description_content_type='text/markdown',
    platforms=["any"],
    install_requires=[
        "numpy >= 1.14.0",
        "scipy >= 1.8.0",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">3.5",
)
