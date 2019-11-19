#!/bin/bash

# Script to be used with cron job, to download late IMERG precipitation on a daily basis

today_minus2="$(date -d '-2 day' '+%Y%m%d')"
year="$(date -d '-2 day' '+%Y')"
month="$(date -d '-2 day' '+%m')"

url='https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGDL.06/'${year}'/'${month}'/3B-DAY-L.MS.MRG.3IMERG.'${today_minus2}'-S000000-E235959.V06.nc4.nc4?HQprecipitation[0:0][0:3599][299:1499],precipitationCal[0:0][0:3599][299:1499],time,lon[0:3599],lat[299:1499]'

# Download file
outpath='/mnt/gpfs/backup/agri/data/weather/global/GPM/GPM_3IMERGDL/'
cd ${outpath} 
curl -n -c ~/.urs_cookies -b ~/.urs_cookies -gLJO --url ${url}

