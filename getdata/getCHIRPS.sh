#!/bin/bash

# Script to download CHIRPS2.0 precipitation data

if [ $# != '2' ]; then
    echo -e '\nArguments: <year_start> <year_end>'
    exit 3
fi

year_start=$1
year_end=$2

for i in `seq ${year_start} ${year_end}`;
do
    wget ftp://ftp.chg.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/global_daily/netcdf/p25/chirps-v2.0.${i}.days_p25.nc
done

echo -e 'Complete.'

