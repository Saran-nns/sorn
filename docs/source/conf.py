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
from sphinx.ext.autosummary import Autosummary
from sphinx.ext.autosummary import get_documenter
from docutils.parsers.rst import directives
from sphinx.util.inspect import safe_getattr

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("N:/sorn/sorn/"))

# -- Project information -----------------------------------------------------

project = "Self-Organizing Recurrent Neural Network (SORN)"
copyright = "2021, Saranraj Nambusubramaniyan"
author = "Saranraj Nambusubramaniyan"

# The full version, including alpha/beta/rc tags
release = version = sorn.__version__

autosummary_generate = True

# Sphinx gallery configuration
from sphinx_gallery.sorting import FileNameSortKey

sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    #'filename_pattern': '^((?!sgskip).)*$',
    "gallery_dirs": ["gallery"],
    "doc_module": ("sorn",),
    "reference_url": {
        "numpy": "http://docs.scipy.org/doc/numpy",
        "geopandas": "https://geopandas.readthedocs.io/en/latest/",
    },
    "sphinx_gallery": None,
    "backreferences_dir": "reference",
    "within_subsection_order": FileNameSortKey,
}

# Napoleon settings
napoleon_google_docstring = True

autodoc_mock_imports = ["sorn"]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
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

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Tell sphinx what the primary language being documented is.
primary_domain = "py"

# Tell sphinx what the pygments highlight language should be.
highlight_language = "py"

# def setup(app):
#     app.add_css_file("_theme.css")

autodoc_default_flags = ["members", "undoc-members", "special-members"]
autodoc_member_order = "bysource"
autodoc_mock_imports = ["rpy2"]


def setup(app):
    """
    Enable documenting 'special methods' using the autodoc_ extension.
    :param app: The Sphinx application object.
    This function connects the :func:`special_methods_callback()` function to
    ``autodoc-skip-member`` events.
    .. _autodoc: http://www.sphinx-doc.org/en/stable/ext/autodoc.html
    """
    app.connect("autodoc-skip-member", special_methods_callback)


def special_methods_callback(app, what, name, obj, skip, options):
    """
    Enable documenting 'special methods' using the autodoc_ extension.
    Refer to :func:`enable_special_methods()` to enable the use of this
    function (you probably don't want to call
    :func:`special_methods_callback()` directly).
    This function implements a callback for ``autodoc-skip-member`` events to
    include documented 'special methods' (method names with two leading and two
    trailing underscores) in your documentation. The result is similar to the
    use of the ``special-members`` flag with one big difference: Special
    methods are included but other types of members are ignored. This means
    that attributes like ``__weakref__`` will always be ignored (this was my
    main annoyance with the ``special-members`` flag).
    The parameters expected by this function are those defined for Sphinx event
    callback functions (i.e. I'm not going to document them here :-).
    """
    import types

    if getattr(obj, "__doc__", None) and isinstance(
        obj, (types.FunctionType, types.MethodType)
    ):
        return False
    else:
        return skip
