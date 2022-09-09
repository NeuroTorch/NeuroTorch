# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import pprint
import sys
import shutil
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, basedir)
import neurotorch


_html_folders_formatted = {}
_allowed_special_methods = ["__init__", "__call__"]


def skip(app, what, name, obj, would_skip, options):
    if name in _allowed_special_methods:
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)
    app.connect('html-page-context', change_pathto)
    app.connect('build-finished', move_private_folders)


def change_pathto(app, pagename, templatename, context, doctree):
    """
    Replace pathto helper to change paths to folders with a leading underscore.
    """
    pathto = context.get('pathto')
    
    def gh_pathto(otheruri, *args, **kw):
        if otheruri.startswith('_'):
            otheruri_fmt = otheruri[1:]
            _html_folders_formatted[os.path.dirname(otheruri)] = os.path.dirname(otheruri_fmt)
        else:
            otheruri_fmt = otheruri
        return pathto(otheruri_fmt, *args, **kw)
    
    context['pathto'] = gh_pathto


def move_private_folders(app, e):
    """
    Remove leading underscore from folders in in the output folder.
    """
    
    def join(dir):
        return os.path.join(app.builder.outdir, dir)
    
    for item in os.listdir(app.builder.outdir):
        if item.startswith('_') and os.path.isdir(join(item)) and item in _html_folders_formatted:
            shutil.move(join(item), join(item[1:]))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NeuroTorch'
copyright = neurotorch.__copyright__.replace("Copyright ", "")
author = neurotorch.__author__
version = neurotorch.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    # 'sphinx_rtd_theme',
    'sphinxcontrib.bibtex',
    'sphinx_mdinclude',
    # 'nbsphinx',
    'sphinx.ext.mathjax',
    # 'sphinx.ext.mathbase',
    'sphinx.ext.todo',
]

bibtex_bibfiles = ['references.bib']

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'karma_sphinx_theme'
html_static_path = ['_static']
# html_css_files = [
#     'css/float_right.css',
# ]
# mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax_path = "https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
latex_engine = 'xelatex'
latex_elements = {
    'preamble': r'\usepackage{physics}'
                r'\usepackage{mathtools}'
                r'\usepackage{amsmath}'
                r'\usepackage{nicefrac}'
}
