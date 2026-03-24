..
   DO NOT DELETE THIS FILE! It contains the all-important `.. autosummary::` directive with `:recursive:` option, without
   which API documentation wouldn't get extracted from docstrings by the `sphinx.ext.autosummary` engine. It is hidden 
   (not declared in any toctree) to remove an unnecessary intermediate page; index.md instead points directly to the 
   submodule page. For .pl., we have a separate page to allow extra docs there and since it has very few top-level functions
   to declare.


:orphan:

.. currentmodule:: crested
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   pp
   tl
   pl
   utils
