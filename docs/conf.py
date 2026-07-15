import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "Distributed Inference"
author = "Distributed Inference contributors"
copyright = "2026, Distributed Inference contributors"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "furo"
pygments_style = "friendly"
pygments_dark_style = "monokai"
autodoc_typehints = "description"
