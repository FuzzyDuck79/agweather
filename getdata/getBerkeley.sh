#!/bin/bash

# Script to download Berkeley Earth temperature data

if [ $# != '2' ]; then
    echo -e '\nArguments: <decade_start> <decade_end>'
    exit 3
fi

decade_start=$1
decade_end=$2

for dec in `seq ${decade_start} 10 ${decade_end}`;
do
    wget http://berkeleyearth.lbl.gov/auto/Global/Gridded/Complete_TMAX_Daily_LatLong1_${dec}.nc
    wget http://berkeleyearth.lbl.gov/auto/Global/Gridded/Complete_TMIN_Daily_LatLong1_${dec}.nc
done

echo -e 'Complete!'

