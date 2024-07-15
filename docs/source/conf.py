# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))


project = 'SynapticSync'
copyright = '2024, Alan Sun'
author = 'Alan Sun'
release = '0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
]

templates_path = ['_templates']
exclude_patterns = []

autodoc_mock_imports = ["torch", "transformer_lens"]

# html_theme = 'alabaster'
html_theme = "nature"
html_static_path = ['_static']
html_title = f"{project} Documentation"
napoleon_google_docstring = True
html_favicon = "_static/brain-icon.png"
numfig = True
numfig_format = {
    'code-block': 'Block %s',
    'figure': 'Fig. %s',
    'section': 'Section',
    'table': 'Table %s',
}
