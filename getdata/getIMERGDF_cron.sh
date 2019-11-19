#!/bin/bash

# Script to be used with cron job, to download final IMERG precipitation on a daily basis
# with a lag of 150 days (to ensure that it has been produced).

today_minus150="$(date -d '-150 day' '+%Y%m%d')"
year="$(date -d '-150 day' '+%Y')"
month="$(date -d '-150 day' '+%m')"

url='https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGDF.06/'${year}'/'${month}'/3B-DAY.MS.MRG.3IMERG.'${today_minus150}'-S000000-E235959.V06.nc4.nc4?HQprecipitation[0:0][0:3599][299:1499],precipitationCal[0:0][0:3599][299:1499],time,lon[0:3599],lat[299:1499]'

# Download file
outpath='/mnt/gpfs/backup/agri/data/weather/global/GPM/GPM_3IMERGDF/'
cd ${outpath} 
curl -n -c ~/.urs_cookies -b ~/.urs_cookies -gLJO --url ${url}

