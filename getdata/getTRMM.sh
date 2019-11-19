#!/bin/bash

# Script to be used from terminal to download TRMM precipitation for specific date

if [ $# != '3' ]; then
        echo -e '\nArguments: <yyyy> <mm> <dd>'
        exit 3
fi

year=$1
month=$2
day=$3

url='https://disc2.gesdisc.eosdis.nasa.gov/data/TRMM_RT/TRMM_3B42RT_Daily.7/'${year}'/'${month}'/3B42RT_Daily.'${year}${month}${day}'.7.nc4'

# Download file
curl -n -c ~/.urs_cookies -b ~/.urs_cookies -LJO --url ${url}

