from setuptools import setup
from src.neurotorch.version import __version__
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.readlines()

setup(
    name='NeuroTorch',
    version=__version__,
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/JeremieGince/NeuroTorch',
    author='Jérémie Gince',
    author_email='gincejeremie@gmail.com',
    license='Apache 2.0',
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
)


# build library
#  setup.py sdist bdist_wheel

# publish on PyPI
#   twine check dist/*
#   twine upload --repository-url https://test.pypi.org/legacy/ dist/*
#   twine upload dist/*

