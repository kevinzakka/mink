# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pathlib import Path

import toml

project = "mink"
copyright = "2026, Kevin Zakka"
author = "Kevin Zakka"

# The short X.Y version
version: str = toml.load(Path(__file__).absolute().parent.parent / "pyproject.toml")[
    "project"
]["version"]
if not version.isalpha():
    version = "v" + version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx-mathjax-offline",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_favicon",
]

autodoc_typehints = "both"
autodoc_class_signature = "separated"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "inherited-members": False,
    "exclude-members": "__init__, __post_init__, __new__",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {".rst": "restructuredtext"}

autodoc_type_aliases = {
    "npt.ArrayLike": "ArrayLike",
}

# Keep the in-page "On this page" TOC to section headings (one entry per task /
# limit) instead of listing every class and method, which overflows the pane.
toc_object_entries = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_theme_options = {
    "logo": {
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
    },
    # Expand the in-page "On this page" TOC so individual tasks/limits (H3
    # sections nested under H2 groups) appear in the right pane.
    "show_toc_level": 2,
}

favicons = [
    {
        "href": "favicon.png",
    }
]

htmlhelp_basename = "minkdoc"
