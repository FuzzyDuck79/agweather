#!/bin/bash

# Paths to input and output directories
basepath='/mnt/gpfs/backup/agri/data/weather/global/AgMERRA/'
outpath=${basepath}

# Setting up the output directories
if [ ! -e $outpath ]; then mkdir -p $outpath; fi

for year in {1980..2010}

do
    wget https://data.giss.nasa.gov/impacts/agmipcf/agmerra/AgMERRA_${year}_prate.nc4
    wget https://data.giss.nasa.gov/impacts/agmipcf/agmerra/AgMERRA_${year}_srad.nc4
    wget https://data.giss.nasa.gov/impacts/agmipcf/agmerra/AgMERRA_${year}_tmax.nc4
    wget https://data.giss.nasa.gov/impacts/agmipcf/agmerra/AgMERRA_${year}_tmin.nc4
    wget https://data.giss.nasa.gov/impacts/agmipcf/agmerra/AgMERRA_${year}_wndspd.nc4
    wget https://data.giss.nasa.gov/impacts/agmipcf/agmerra/AgMERRA_${year}_rhstmax.nc4
done

echo 'Complete!'
