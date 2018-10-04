<h1 align="center">
  <a href="https://github.com/znicholls/netcdf-scm" style="display: block; margin: 0 auto;">
   <img src="https://raw.githubusercontent.com/znicholls/netcdf-scm/master/docs_to_update/imgs/magicc_logo.png"
        style="max-width: 5%;" alt="MAGICC logo"></a><br>
</h1>

# NetCDF SCM

**Warning: `netcdf_scm` is still under heavy development so things are likely to change fast**

<p align="center">
<a href="https://codecov.io/gh/znicholls/netcdf-scm">
<img src="https://img.shields.io/codecov/c/github/znicholls/netcdf-scm.svg"
     alt="Code coverage"/></a>
<a href="https://pypi.org/project/netcdf-scm/">
<img src=https://img.shields.io/pypi/v/netcdf-scm.svg"
     alt="PyPI"/></a>
<a href="https://pypi.org/project/netcdf-scm/">
<img src="https://img.shields.io/pypi/pyversions/netcdf-scm.svg"
     alt="Python versions"/></a>
<!-- JOSS paper -->
<!-- <a href="https://joss.theoj.org/papers/85eb9a9401fe968073bb429ea361924e/status.svg">
<img src="https://joss.theoj.org/papers/85eb9a9401fe968073bb429ea361924e"
     alt="JOSS paper"/></a> -->
<!-- Stickler CI badge, not sure yet -->
<br>




<!-- https://shields.io/ is a good source of these -->
<!-- conda shields, for the future -->
<!-- <a href="https://anaconda.org/conda-forge/iris">
<img src="https://img.shields.io/conda/dn/conda-forge/iris.svg"
     alt="conda-forge downloads" /></a> -->
<a href="https://github.com/znicholls/netcdf-scm/releases">
<img src="https://img.shields.io/github/tag/znicholls/netcdf-scm.svg"
     alt="Latest version"/></a>
<a href="https://github.com/znicholls/netcdf-scm/commits/master">
<img src="https://img.shields.io/github/last-commit/znicholls/netcdf-scm.svg"
     alt="Last commit"/></a>
<a href="https://github.com/znicholls/netcdf-scm/commits/master">
<img src="https://img.shields.io/github/commits-since/znicholls/netcdf-scm/latest.svg"
     alt="Commits since last release" /></a>
<a href="https://github.com/znicholls/netcdf-scm/graphs/contributors">
<img src="https://img.shields.io/github/contributors/znicholls/netcdf-scm.svg"
     alt="# contributors" /></a>
<a href="https://travis-ci.org/znicholls/netcdf-scm/branches">
<img src="https://travis-ci.org/znicholls/netcdf-scm.svg?branch=master"
     alt="Travis-CI" /></a>
<!-- DOI -->
<!-- <a href="https://zenodo.org/badge/latestdoi/5312648">
<img src="https://zenodo.org/badge/5312648.svg"
     alt="zenodo" /></a> -->
<a href="https://github.com/znicholls/netcdf-scm/blob/master/LICENSE">
<img src="https://img.shields.io/pypi/l/netcdf-scm.svg"
     alt="license" /></a>
</p>
<br>

## Why is there a `Makefile` in a pure Python repository?

Whilst it's probably not good practice, a `Makefile` is a simple way to automate general setup (environment setup in particular).
Hence we have one here which basically acts as a notes file for how to do all those little jobs which we often forget e.g. setting up environments, running tests (and making sure we're in the right environment), building docs, setting up auxillary bits and pieces.
