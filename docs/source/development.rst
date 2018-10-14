Development
===========

If you're interested in contributing to NetCDF-SCM, we'd love to have you on board!
This section of the docs details how to get setup to contribute and how best to communicate.


Contributing
------------

[To be written]


Getting setup
-------------

[To be written]


Getting help
~~~~~~~~~~~~

Whilst developing, unexpected things can go wrong (that's why it's called 'developing', if we knew what we were doing, it would already be 'developed').
Normally, the fastest way to solve an issue is to contact us via the `issue tracker <https://github.com/znicholls/netcdf-scm/issues>`_ (if your issue is a feature request or a bug, please use the templates available, otherwise, simply open a normal issue).
The other option is to debug yourself.
For this purpose, we provide a list of the tools we use during our development as starting points for your search to find what has gone wrong.


Development tools
+++++++++++++++++

This list of development tools is what we rely on to develop NetCDF-SCM reliably and reproducibly.
It gives you a few starting points in case things do go inexplicably wrong and you want to work out why.
We include links with each of these tools to starting points that we think are useful, in case you want to learn more.

- `Git <http://swcarpentry.github.io/git-novice/>`_
- `Make <https://swcarpentry.github.io/make-novice/>`_
- `Conda virtual environments <https://medium.freecodecamp.org/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c>`_
    - note the common gotcha that ``source activate`` has now changed to ``conda activate``
    - we use conda instead of pure pip environments because they help us deal with Iris' dependencies: if you want to learn more about pip and pip virtual environments, check out `this introduction <https://www.dabapps.com/blog/introduction-to-pip-and-virtualenv-python/>`_
- `Tests <https://semaphoreci.com/community/tutorials/testing-python-applications-with-pytest>`_
    - we use a blend of `pytest <https://docs.pytest.org/en/latest/>`_ and the inbuilt Python testing capabilities for our tests so checkout what we've already done in ``tests`` to get a feel for how it works
- `Continuous integration (CI) <https://docs.travis-ci.com/user/for-beginners/>`_
    - we use `Travis CI <https://travis-ci.com/>`_ for our CI but there are a number of good providers
- `Jupyter Notebooks <https://medium.com/codingthesmartway-com-blog/getting-started-with-jupyter-notebook-for-python-4e7082bd5d46>`_
    - we'd recommend simply installing via pip in your virtual environment
- Sphinx_


Buiding the docs
----------------

[To be written]

For documentation we use Sphinx_.
To get ourselves started with Sphinx, we started with `this example <https://pythonhosted.org/an_example_pypi_project/sphinx.html>`_ then used `Sphinx's getting started guide <http://www.sphinx-doc.org/en/master/usage/quickstart.html>`_.


Gotchas
~~~~~~~

To get Sphinx to work completely, you require `Latexmk <https://mg.readthedocs.io/latexmk.html>`_.
On a Mac this can be installed with ``sudo tlmgr install latexmk``.
You will most likely also need to install some other packages (if you don't have the full distribution).
You can check which package contains any missing files with ``tlmgr search --global --file [filename]``.
You can then install the packages with ``sudo tlmgr install [package]``.


Docstring style
~~~~~~~~~~~~~~~

For our docstrings we use numpy style docstrings.
For more information on these, `here is the full guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_ and `the quick reference we also use <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_.


Releasing
---------

[To be written]


Why is there a ``Makefile`` in a pure Python repository?
--------------------------------------------------------

Whilst it may not be standard practice, a ``Makefile`` is a simple way to automate general setup (environment setup in particular).
Hence we have one here which basically acts as a notes file for how to do all those little jobs which we often forget e.g. setting up environments, running tests (and making sure we're in the right environment), building docs, setting up auxillary bits and pieces.


Why did we choose a BSD 2-Clause License?
-----------------------------------------

We want to ensure that our code can be used and shared as easily as possible.
Whilst we love transparency, we didn't want to **force** all future users to also comply with a stronger license such as AGPL.
Hence the choice we made.

We recommend [Morin et al. 2012]_ for more information for scientists about open-source software licenses.


.. _Sphinx: http://www.sphinx-doc.org/en/master/
