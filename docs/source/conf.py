# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, basedir)


def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NeuroTorch'
copyright = '2022, Jeremie Gince'
author = 'Jeremie Gince'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
