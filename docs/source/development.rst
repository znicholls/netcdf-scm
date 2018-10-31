Development
===========

If you're interested in contributing to NetCDF-SCM, we'd love to have you on board!
This section of the docs details how to get setup to contribute and how best to communicate.


Contributing
------------

All contributions are welcome, some possible suggestions include:

- tutorials (or support questions which, once solved, result in a new tutorial :D)
- blog posts
- improving the documentation
- bug reports
- feature requests
- pull requests

Please report issues or discuss feature requests in the `NetCDF-SCM issue tracker`_.
If your issue is a feature request or a bug, please use the templates available, otherwise, simply open a normal issue :)

As a contributor, please follow a couple of conventions:

- Create issues in the `NetCDF-SCM issue tracker`_ for changes and enhancements, this ensures that everyone in the community has a chance to comment
- Be welcoming to newcomers and encourage diverse new contributors from all backgrounds: see the `Python Community Code of Conduct <https://www.python.org/psf/codeofconduct/>`_


Getting setup
-------------

To get setup as a developer, we recommend the following steps (if any of these tools are unfamiliar, please see the resources we recommend in `Development tools`_):

#. Install conda and make
#. Run ``make conda_env``, if that fails you can try doing it manually

    #. Create a conda virtual environment to use with NetCDF-SCM
    #. Activate your virtual environment
    #. Install the conda minimal dependencies with ``conda install --file conda-environment-minimal.yaml -n your-environment-name``
    #. Install the conda development dependencies with ``conda install --file conda-environment-dev.yaml -n your-environment-name``
    #. Upgrade pip ``pip install --upgrade pip``
    #. Install pip minimal dependencies ``pip install -Ur pip-requirements-minimal.txt``
    #. Change your current directory to NetCDF-SCM's root directory (i.e. the one which contains ``README.rst``), ``cd netcdf-scm``
    #. Install an editable version of NetCDF-SCM along with development dependencies, ``pip install -e .[test,docs,deploy]``

#. Make sure the tests pass by running ``make test_all``, if that files the commands are

    #. Activate your virtual environment
    #. Run the unit and integration tests ``pytest --cov -rfsxEX --cov-report term-missing``
    #. Test the notebooks ``pytest -rfsxEX --nbval ./notebooks --sanitize ./notebooks/tests_sanitize.cfg``


Getting help
~~~~~~~~~~~~

Whilst developing, unexpected things can go wrong (that's why it's called 'developing', if we knew what we were doing, it would already be 'developed').
Normally, the fastest way to solve an issue is to contact us via the `issue tracker <https://github.com/znicholls/netcdf-scm/issues>`_.
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
    - we'd recommend simply ``jupyter`` installing via pip in your virtual environment
- Sphinx_


Other tools
+++++++++++

We also use some other tools which aren't necessarily the most familiar.
Here we provide a list of these along with useful resources.

- `Regular expressions <https://www.oreilly.com/ideas/an-introduction-to-regular-expressions>`_
    - we use `regex101.com <regex101.com>`_ to help us write and check our regular expressions, make sure the language is set to Python to make your life easy!


Formatting
----------

To help us focus on what the code does, not how it looks, we use a couple of automatic formatting tools.
These automatically format the code for us and tell use where the errors are.
To use them, after setting yourself up (see `Getting setup`_), simply run ``make black`` and ``make flake8``.
Note that ``make black`` can only be run if you have committed all your work i.e. your working directory is 'clean'.
This restriction is made to ensure that you don't format code without being able to undo it, just in case something goes wrong.


Buiding the docs
----------------

After setting yourself up (see `Getting setup`_), building the docs is as simple as running ``make docs`` (note, run ``make -B docs`` to force the docs to rebuild and ignore make when it says '... index.html is up to date').
This will build the docs for you.
You can preview them by opening ``docs/build/html/index.html`` in a browser.

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

[To be written, once I've done it]


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
.. _NetCDF-SCM issue tracker: https://github.com/znicholls/netcdf-scm/issues
