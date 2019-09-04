---
title: 'NetCDF-SCM: Processing NetCDF-SCM files for use with simple climate models'
tags:
  - climate change
  - simple climate models
  - netcdf
  - data processing
authors:
  - name: Zebedee Nicholls
    orcid: 0000-0002-4767-2723
    affiliation: 1, 2
  - name: Jared Lewis
    orcid: 0000-0002-8155-8924
    affiliation: 1
affiliations:
  - name: Australian-German Climate & Energy College, University of Melbourne, Parkville, Victoria, Australia
    index: 1
  - name: School of Earth Sciences, University of Melbourne, Parkville, Victoria, Australia
    index: 2
date: 30 August 2019
bibliography: paper.bib
output: pdf_document
---

# Summary

Climate model output is among the world's largest datasets.
For example, the Sixth Coupled Model Intercomparison Project archive is expected to be approximately 18PB [@balaji2018requirements].
To minimise the on-disk data volume, this data is always provided in the self-describing netCDF binary format.
In addition, it is provided in a highly regularised way, based on a clearly-defined data reference syntax.

Working with such large datasets requires a specialised set of techincal skills.
However, many of the users of climate data do not have such skills.
In particular, the writers of computationally efficient, global-mean focussed climate models (also known as 'reduced complexity' or 'simple' climate models aka SCMs) rarely, if ever, base their tools on the netCDF format [@Meinshausen2011,@Hartin2015,@smith2018fair].

NetCDF-SCM makes it easy to process the raw climate model output into formats which of interest to SCM developers yet do not require the technical expertise of netCDF (e.g. plain csvs or marked up ASCII files).
In addition, SCMs do not require the full detail of the more complex model output.
Instead, they typically focus on land/ocean and hemispheric mean timeseries, greatly reducing the size of the data of interest.
NetCDF-SCM allows such area averages to be calculated using model specific area and land surface type information.
It does this thanks to its inbuilt knowledge of different data reference syntaxes used in climate data processing.

In practice, NetCDF-SCM is simply a wrapper around the Iris package [@Iris].
Without Iris' climate data handling capabilities, in partciular its lazy data handling (via dask [@Dask]) and grid awareness, NetCDF-SCM would not be possible.

In order to process files, NetCDF-SCM procedes in two steps.
The first step, 'crunching', processes the source data into netCDF files containing only the average timeseries of interest.
From these intermediate files, the data can be 'wrangled' into other formats.
The steps are split because crunching is generally much slower, requiring reading GB to TB of data, while the crunched files are much smaller and hence can be re-processed many orders of magnitude faster.

The source data files vary in size by several orders of magnitude.
The smallest data files are in the region of 10MB, with the largest being approximately a thousand times bigger (on disk).
Crunching such data efficiently hence requires different strategies, depending on the number of cores and memory available.

To allow user control of this, NetCDF-SCM's crunching command line tool allows the user to define how they want to use their resources.
This control extends to choosing different strategies for different sized files.
For example, the user can choose to crunch all files with fewer than 100 million data points in memory and in parallel with 30 cores whilst simulataneously specifying that any file with over 1 billion data points should be crunched lazily and any such files should be crunched serially (allowing Dask to handle efficient use of memory during the lazy crunching).

On top of data processing, NetCDF-SCM also retains all relevant metadata.
Specifically, it keeps all metadata from the source data files as well as information about how the files was processed (including the version of NetCDF-SCM) to ensure full traceability and reproducibility.
Additionally, extensive logs are written during the crunching step and a database is developed.
This database can be used to ensure that data is only re-crunched if it does not already exist, providing a significant increase in speed.

NetCDF-SCM was designed to be used by developers of SCMs who need these processed for their own tools.
It may also find an audience with climate analyses who are only interested in aggregate timeseries.
To ensure that its outputs are quality, NetCDF-SCM has been validated against the output of the KNMI climate explorer, the most well-known public database of such timeseries.
Given its relatively simple installation (via the Conda package manager), NetCDF-SCM enables researchers to spend more time analysing and developing their models and less time writing their own processing tools.

## Acknowledgements

We thank the developers of Iris for their tireless efforts and the brilliant tool they have developed.
Without it, NetCDF-SCM would not be possible.
We would also like to thank Robert Gieseke for ongoing advice, in particular with initial decisions about how to setup the repository.

# References
