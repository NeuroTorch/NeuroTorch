from setuptools import setup
# from neurotorch import __author__, __url__, __email__, __version__, __license__
import setuptools
# from src import neurotorch

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()
#
# with open("requirements.txt", "r", encoding="utf-8") as fh:
#     install_requires = fh.readlines()

setup(
    name='NeuroTorch',
    # version=__version__,
    # description=neurotorch.__doc__,
    long_description='file: README.md',
    long_description_content_type="text/markdown",
    # url=__url__,
    # author=__author__,
    # author_email=__email__,
    # license=__license__,
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
    ],
    # install_requires=install_requires,
    project_urls={
        'Homepage': 'https://github.com/NeuroTorch/NeuroTorch',
        'Source': 'https://github.com/NeuroTorch/NeuroTorch',
        'Documentation': 'https://neurotorch.github.io/NeuroTorch',
    },
)


# build library
#  setup.py sdist bdist_wheel
# With pyproject.toml
# python -m pip install --upgrade build
# python -m build

# publish on PyPI
#   twine check dist/*
#   twine upload --repository-url https://test.pypi.org/legacy/ dist/*
#   twine upload dist/*

