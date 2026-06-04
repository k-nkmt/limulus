# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "limulus"
copyright = "2026, k-nkmt"
author = "k-nkmt"
release = "0.5.0"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "ja", "README_ja.md"]

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_title = "limulus - Data Step for your Workspace"

html_theme_options = {
    "logo": {
        "image_light": "_static/limulus_light.svg",
        "image_dark": "_static/limulus_dark.svg",
    }
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]

nb_execution_mode = "off"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
}

autodoc_default_options = {
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_mock_imports = [
    "pyarrow",
    "pyarrow.compute",
    "pyarrow.ipc",
    "pyarrow.parquet",
    "polars",
    "pandas",
    "limulus_native",
]
napoleon_google_docstring = True
napoleon_numpy_docstring = False
