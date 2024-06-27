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
sys.path.insert(0, os.path.abspath('../../lcpom'))  # Source code dir relative to this file


# -- Project information -----------------------------------------------------

project = "LC-POM"
copyright = "2023-present, LCPOM developers, i.e. Chuqiao Chen (Elise), Viviana Palacio-Betancur, Pablo Zubieta"
author = (
    "LCPOM developers: Chuqiao Chen (Elise), Viviana Palacio-Betancur, Pablo Zubieta"
)

# The full version, including alpha/beta/rc tags
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon"
]

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '_templates']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_title = "LCPOM documentation"
html_logo = "_static/logo.svg"

html_theme = "furo"
html_show_sphinx = False

html_theme_options = {
    "light_css_variables": {
        "color-foreground-primary": "#3c3c3c",
        "color-background-secondary": "var(--color-background-primary)",
        "color-brand-primary": "#34818a",
        "color-brand-content": "#34818a",
        "color-api-name": "#76a02c",
        "color-api-pre-name": "#76a02c",
        "font-stack": "Atkinson Hyperlegible, system-ui, -apple-system, BlinkMacSystemFont, "
        "Segoe UI, Helvetica, Arial, sans-serif, Apple Color Emoji, Segoe UI Emoji",
    },
    "dark_css_variables": {
        "color-background-primary": "var(--color-background-secondary)",
        "color-brand-primary": "#45acb8",
        "color-brand-content": "#45acb8",
        "color-api-name": "#9fd620",
        "color-api-pre-name": "#9fd620",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/depablogroup/lc-pom",
            "html": "",
            "class": "fa-brands fa-solid fa-github fa-lg",
        },
    ],
    "sidebar_hide_name": False,
}


# # Add any paths that contain custom static files (such as style sheets) here,
# # relative to this directory. They are copied after the builtin static files,
# # so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for EPUB output
epub_show_urls = "footnote"

# Add options for the spell checking.
spelling_lang = "en_US"
tokenizer_lang = "en_US"
spelling_show_suggestions = True
spelling_warning = True