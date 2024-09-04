# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sorn

version = sorn.__version__

# sys.path.insert(0, os.path.abspath("../.."))
# SORN root folder to sphinx path
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, os.path.abspath("N:/sorn/sorn/"))

# -- Project information -----------------------------------------------------

project = "Self-Organizing Recurrent Neural Network (SORN)"
copyright = "2021, Saranraj Nambusubramaniyan"
author = "Saranraj Nambusubramaniyan"

autosummary_generate = True

# Napoleon settings
napoleon_google_docstring = True

# autodoc_mock_imports = ["sorn"]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# source_suffix
# The file extensions of source files. Sphinx considers the files with this suffix as sources. The value can be a dictionary mapping file extensions to file types. For example:

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "restructuredtext",
    ".md": "markdown",
}
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"
html_theme_options = {"body_max_width": "auto"}
# html_theme_options = {"rightsidebar": "true", "relbarbgcolor": "black"}


# Tell sphinx what the primary language being documented is.
primary_domain = "py"

# Tell sphinx what the pygments highlight language should be.
highlight_language = "py"

autodoc_default_flags = ["members", "undoc-members", "special-members"]
