#!/bin/sh

./qsub.sh data/20190802/ country 0.1 /mnt/gpfs/backup/agri/data/geospatial/global/masks/
./qsub.sh data/20190802/ country 0.25 /mnt/gpfs/backup/agri/data/geospatial/global/masks/
./qsub.sh data/20190802/ country 0.5 /mnt/gpfs/backup/agri/data/geospatial/global/masks/
./qsub.sh data/20190802/ country 1.0 /mnt/gpfs/backup/agri/data/geospatial/global/masks/

./qsub.sh data/20190802/ subregion 0.1 /mnt/gpfs/backup/agri/data/geospatial/global/masks/
./qsub.sh data/20190802/ subregion 0.25 /mnt/gpfs/backup/agri/data/geospatial/global/masks/
./qsub.sh data/20190802/ subregion 0.5 /mnt/gpfs/backup/agri/data/geospatial/global/masks/
./qsub.sh data/20190802/ subregion 1.0 /mnt/gpfs/backup/agri/data/geospatial/global/masks/

./qsub.sh data/20190802/ region 0.1 /mnt/gpfs/backup/agri/data/geospatial/global/masks/
./qsub.sh data/20190802/ region 0.25 /mnt/gpfs/backup/agri/data/geospatial/global/masks/
./qsub.sh data/20190802/ region 0.5 /mnt/gpfs/backup/agri/data/geospatial/global/masks/
./qsub.sh data/20190802/ region 1.0 /mnt/gpfs/backup/agri/data/geospatial/global/masks/

