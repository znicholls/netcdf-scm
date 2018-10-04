---
title: 'NetCDF-SCM: A Python wrapper for processing netCDF files for use with simple climate models'
tags:
  - climate change
  - simple climate models
  - python-wrapper
  - netCDF
authors:
  - name: Zebedee Nicholls
    orcid: 0000-0002-4767-2723
    affiliation: 1, 2
  - name: Robert Gieseke
    orcid: 0000-0002-1236-5109
    affiliation: 2
affiliations:
  - name: Australian-German Climate & Energy College, School of Earth Sciences, University of Melbourne, Melbourne, Australia
    index: 2
  - name: Potsdam Institute for Climate Impact Research, 14473 Potsdam, Germany
    index: 2
date: TBD
bibliography: paper.bib
output: pdf_document
---

# Summary

NetCDF-SCM is a simple package for processing netCDF files for use in simple climate models. It provides some basic functionality to read, average and plot such files and is built on top of the `iris` library [add iris reference].

It is designed to be used as part of data processing for simple climate models such as MAGICC [@Meinshausen2011], OSCAR^[https://github.com/tgasser/OSCAR] (@Gasser2017), Pyhector^[https://github.com/openclimatedata/pyhector] (@Willner17, @Hartin2015), and FAIR^[https://github.com/OMS-NetZero/FAIR] (@Millar2017.).

[Make this true and fix details]
It can be installed using `pip` from the Python Package Index ^[<https://pypi.python.org/pypi/netcdf-scm>].


Source code, documentation and issue tracker are available in NetCDF-SCM's GitHub
repository^[<https://github.com/znicholls/netcdf-scm>].


## Acknowledgements

We thank the authors of iris for all of their efforts in handling netCDF files.
Without their efforts this project would not have been possible.

# References
