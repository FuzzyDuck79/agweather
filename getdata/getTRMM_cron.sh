#!/bin/bash

# Script to be used with cron job, to download TRMM precipitation on a daily basis

yesterday="$(date -d '-1 day' '+%Y%m%d')"
year="$(date -d '-1 day' '+%Y')"
month="$(date -d '-1 day' '+%m')"


url='https://disc2.gesdisc.eosdis.nasa.gov/data/TRMM_RT/TRMM_3B42RT_Daily.7/'${year}'/'${month}'/3B42RT_Daily.'${yesterday}'.7.nc4'
fname='/3B42RT_Daily.'${yesterday}'.7.nc4'

# Download file
outpath='/mnt/gpfs/backup/agri/data/weather/global/TRMM/3B42RT/'
cd ${outpath} 
curl -n -c ~/.urs_cookies -b ~/.urs_cookies -LJO --url ${url}

