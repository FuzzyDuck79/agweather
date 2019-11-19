Notes on NOAA climate indices
===================================

Mutahar Chalmers, January 2018
-----------------------------------
Indices obtained from the NOAA website: https://www.esrl.noaa.gov/psd/data/climateindices/list/
A shell script, getNOAA.sh, can be run from the command line to download all the various indices automatically.
The script automatically creates a new directory with today's date in YYYYMMDD format, and downloads all indices
in separate text files.

The format for all time series is:

 year1 yearN
 year1 janval febval marval aprval mayval junval julval augval sepval octval novval decval
 year2 janval febval marval aprval mayval junval julval augval sepval octval novval decval
 ...
 yearN janval febval marval aprval mayval junval julval augval sepval octval novval decval
   missing_value

