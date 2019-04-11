# Geospatial Modelling of Australia's National Electricity Market - Dataset

This dataset contains information relating to the topology of Australia's largest electricity transmission network, along with details pertaining to the technical and economic characteristics of generators operating within this grid. Information has been compiled from publicly available datasets released by the Australian Energy Market Operator (AEMO) [1, 2] and Geoscience Australia (GA) [3, 4, 5]. Potential applications include the development of economic dispatch, power-flow, and unit commitment models.

The network is comprised of 912 nodes, 1406 AC edges, and three HVDC links. Information regarding forward and reverse power-flow limits for two AC interconnectors is also provided. Latitude and longitude coordinates are given for each node, with the network based off of geospatial datasets obtained from GA [3, 4, 5]. Signals for electricity demand at each node were derived using regional load profiles in combination with population data obtained from the Australian Bureau of Statistics (ABS) [6]. Allocation methods outlined in [7, 8] were used to disaggregate regional load profiles based on the geospatial distribution of Australia's population. Construction of the generator dataset involved compiling information obtained from AEMO's Market Management System Data Model (MMSDM) [1] and National Transmission Network Development Plan (NTNDP) datasets [2]. Historic generator dispatch signals were also obtained from AEMO [1], allowing the output of market models to be compared with realised outcomes.

For further information regarding the contents of each csv file please refer to "dataset_summary.pdf". Jupyter Notebooks at [9] contain the Python code necessary to reproduce these datasets.

## References:
[1] - Australian Energy Markets Operator. Data Archive (2018). at [http://www.nemweb.com.au/#mms-data-model](http://www.nemweb.com.au/#mms-data-model)

[2] - Australian Energy Markets Operator. NTNDP Database. (2018). at [https://www.aemo.com.au/Electricity/National-Electricity-Market-NEM/Planning-and-forecasting/National-Transmission-Network-Development-Plan/NTNDP-database](https://www.aemo.com.au/Electricity/National-Electricity-Market-NEM/Planning-and-forecasting/National-Transmission-Network-Development-Plan/NTNDP-database)

[3] - Commonwealth of Australia (Geoscience Australia), Electricity Transmission Lines (2017), at [http://pid.geoscience.gov.au/dataset/ga/83105](http://pid.geoscience.gov.au/dataset/ga/83105)

[4] - Commonwealth of Australia (Geoscience Australia), Electricity Transmission Substations (2017), at [http://pid.geoscience.gov.au/dataset/ga/83173](http://pid.geoscience.gov.au/dataset/ga/83173)

[5] - Commonwealth of Australia (Geoscience Australia), Power Stations (2017), at [http://pid.geoscience.gov.au/dataset/ga/82326](http://pid.geoscience.gov.au/dataset/ga/82326)

[6] - Australian Bureau of Statistics. Regional Population Growth, Australia, 2014-15. (2016). at [http://www.abs.gov.au/AUSSTATS/abs@.nsf/DetailsPage/3218.02014-15?OpenDocument](http://www.abs.gov.au/AUSSTATS/abs@.nsf/DetailsPage/3218.02014-15?OpenDocument)

[7] - Zhou, Q. & Bialek, J. W. Approximate model of european interconnected system as a benchmark system to study eﬀects of cross-border trades. IEEE Trans. Power Syst. 20, 782–788 (2005).

[8] - Jensen, T. V. & Pinson, P. RE-Europe, a large-scale dataset for modeling a highly renewable European electricity system. Sci. Data 4, 170175 (2017).

[9] Xenophon, A. K. Geospatial Modelling of Australia's National Electricity Market. (2018). at https://github.com/akxen/egrimod-nem

## Creative Commons Attributions
Geospatial datasets obtained from Geoscience Australia are made available under the following license:

[![License: CC BY 4.0](https://licensebuttons.net/l/by/4.0/80x15.png)](https://creativecommons.org/licenses/by/4.0/) © Commonwealth of Australia (Geoscience Australia) 2017. With the exception of the Commonwealth Coat of Arms, and where otherwise noted, this product is provided under a Creative Commons Attribution 4.0 International Licence. [http://creativecommons.org/licenses/by/4.0/legalcode](http://creativecommons.org/licenses/by/4.0/legalcode)

Population datasets obtained from the Australian Bureau of Statistics are made available under the following license:

[![License: CC BY 4.0](https://licensebuttons.net/l/by/4.0/80x15.png)](https://creativecommons.org/licenses/by/4.0/)
[http://creativecommons.org/licenses/by/4.0/legalcode](http://creativecommons.org/licenses/by/4.0/legalcode)
