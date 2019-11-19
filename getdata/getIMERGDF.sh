#!/bin/bash

# Script to be used from terminal to download final IMERG precipitation for specific date

if [ $# != '3' ]; then
        echo -e '\nArguments: <yyyy> <mm> <dd>'
        exit 3
fi

year=$1
month=$2
day=$3

url='https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGDF.06/'${year}'/'${month}'/3B-DAY.MS.MRG.3IMERG.'${year}${month}${day}'-S000000-E235959.V06.nc4.nc4?HQprecipitation[0:0][0:3599][299:1499],precipitationCal[0:0][0:3599][299:1499],time,lon[0:3599],lat[299:1499]'

# Download file
curl -n -c ~/.urs_cookies -b ~/.urs_cookies -gLJO --url ${url}

