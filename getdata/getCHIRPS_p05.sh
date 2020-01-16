#!/bin/bash

# Script to download CHIRPS2.0 precipitation data at 0.05deg spatial resolution
# and at monthly temporal resolution

if [ $# != '2' ]; then
    echo -e '\nArguments: <year_start> <year_end>'
    exit 3
fi

year_start=$1
year_end=$2

url='ftp://ftp.chg.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/global_daily/netcdf/p05/by_month/'

for y in `seq ${year_start} ${year_end}`;
do
    for m in `seq -w 1 12`;
    do
        wget ${url}chirps-v2.0.${y}.${m}.days_p05.nc -P /mnt/gpfs/backup/agri/data/weather/global/CHIRPS2.0/p05/
    done
done

echo -e 'Complete.'

