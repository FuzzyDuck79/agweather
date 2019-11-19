#!/bin/bash

nowdir="$(date '+%Y%m%d')"

echo 'Files being downloaded to '${nowdir}

# Paths to input and output directories
basepath='/mnt/gpfs/backup/agri/data/climateindices/NOAA/'
outpath=${basepath}${nowdir}'/'


# Setting up the output directories
if [ ! -e $outpath ]; then mkdir -p $outpath; fi

# Pacific North American Index (PNA)
curl https://www.esrl.noaa.gov/psd/data/correlation/pna.data > ${outpath}'pna.txt'

# East Pacific/North Pacific Oscillation (EPO)
curl https://www.esrl.noaa.gov/psd/data/correlation/epo.data > ${outpath}'epo.txt'

# Western Pacific Index (WP)
curl https://www.esrl.noaa.gov/psd/data/correlation/wp.data > ${outpath}'wp.txt'

# Eastern Asia/Western Russia Index (EAWR)
curl https://www.esrl.noaa.gov/psd/data/correlation/ea.data > ${outpath}'eawr.txt'

# North Atlantic Oscillation (NAO) 
curl https://www.esrl.noaa.gov/psd/data/correlation/nao.data > ${outpath}'nao.txt'

# North Atlantic Oscillation, Jones (JonesNAO) 
curl https://www.esrl.noaa.gov/psd/data/correlation/jonesnao.data > ${outpath}'jonesnao.txt'

# Southern Oscillation Index (SOI)
curl https://www.esrl.noaa.gov/psd/data/correlation/soi.data > ${outpath}'soi.txt'

# Eastern Tropical Pacific SST (Nino3) 
curl https://www.esrl.noaa.gov/psd/data/correlation/nina3.data > ${outpath}'nino3.txt'

# Bivariate ENSO Timeseries (CENSO)
curl https://www.esrl.noaa.gov/psd/data/correlation/censo.data > ${outpath}'censo.txt'

# Tropical Northern Atlantic Index (TNA) 
curl https://www.esrl.noaa.gov/psd/data/correlation/tna.data > ${outpath}'tna.txt'

# Tropical Southern Atlantic Index (TSA) 
curl https://www.esrl.noaa.gov/psd/data/correlation/tsa.data > ${outpath}'tsa.txt'

# Western Hemisphere Warm Pool (WHWP)
curl https://www.esrl.noaa.gov/psd/data/correlation/whwp.data > ${outpath}'whwp.txt'

# Oceanic Nino Index (ONI)
curl https://www.esrl.noaa.gov/psd/data/correlation/oni.data > ${outpath}'oni.txt'

# Multivariate Nino Index (MEI) versions 1 and 2 (v1 time series ends Nov 2018)
curl https://www.esrl.noaa.gov/psd/data/correlation/mei.data > ${outpath}'mei.txt'
curl https://www.esrl.noaa.gov/psd/data/correlation/meiv2.data > ${outpath}'meiv2.txt'

# Nino 1+2 (Nino12)
curl https://www.esrl.noaa.gov/psd/data/correlation/nina1.data > ${outpath}'nino12.txt'

# Nino 4 (Nino4)
curl https://www.esrl.noaa.gov/psd/data/correlation/nina4.data > ${outpath}'nino4.txt'

# Nino 3.4 (Nino34)
curl https://www.esrl.noaa.gov/psd/data/correlation/nina34.data > ${outpath}'nino34.txt'

# Pacific Decadal Oscillation (PDO)
curl https://www.esrl.noaa.gov/psd/data/correlation/pdo.data > ${outpath}'pdo.txt'

# Tripole Index for the Interdecadal Pacific Oscillation (IPOTPI) 
curl https://www.esrl.noaa.gov/psd/data/timeseries/IPOTPI/ipotpi.hadisst2.data > ${outpath}'ipotpi.txt'

# Northern Oscillation Index (NOI) 
curl https://www.esrl.noaa.gov/psd/data/correlation/noi.data > ${outpath}'noi.txt'

# Northern Pacific Pattern (NP) 
curl https://www.esrl.noaa.gov/psd/data/correlation/np.data > ${outpath}'np.txt'

# Trans-Nino Index (TNI)
curl https://www.esrl.noaa.gov/psd/data/correlation/tni.data > ${outpath}'tni.txt'

# Arctic Oscillation (AO) 
curl https://www.esrl.noaa.gov/psd/data/correlation/ao.data > ${outpath}'ao.txt'

# Antarctic Oscillation (AAO) 
curl https://www.esrl.noaa.gov/psd/data/correlation/aao.data > ${outpath}'aao.txt'

# Pacific Warmpool (PW) 
curl https://www.esrl.noaa.gov/psd/data/correlation/pacwarm.data > ${outpath}'pacwarm.txt'

# Tropical Pacific SST EOF (EOFPAC)
curl https://www.esrl.noaa.gov/psd/data/correlation/eofpac.data > ${outpath}'eofpac.txt'

# Atlantic Tripole SST EOF (ATLTRI) 
curl https://www.esrl.noaa.gov/psd/data/correlation/atltri.data > ${outpath}'atltri.txt'

# Atlantic Multidecadal Oscillation, unsmoothed (AMO, unsmoothed) 
curl https://www.esrl.noaa.gov/psd/data/correlation/amon.us.data > ${outpath}'amo_unsmoothed.txt'

# Atlantic Multidecadal Oscillation, smoothed (AMO, smoothed) 
curl https://www.esrl.noaa.gov/psd/data/correlation/amon.sm.data > ${outpath}'amo_smoothed.txt'

# Atlantic Meridional Mode (AMM) 
curl https://www.esrl.noaa.gov/psd/data/timeseries/monthly/AMM/ammsst.data > ${outpath}'amm.txt'

# Northern Tropical Atlantic Index (NTA) 
curl https://www.esrl.noaa.gov/psd/data/correlation/NTA_ersst.data > ${outpath}'nta.txt'

# Caribbean Index (CAR) 
curl https://www.esrl.noaa.gov/psd/data/correlation/CAR_ersst.data > ${outpath}'car.txt'

# Quasi-Biennial Oscillation (QBO) 
curl https://www.esrl.noaa.gov/psd/data/correlation/qbo.data > ${outpath}'qbo.txt'

# ENSO Precipitation Index (ESPI)
curl https://www.esrl.noaa.gov/psd/data/correlation/espi.data > ${outpath}'espi.txt'

# Central Indian Precipitation, monsoon region (IndiaMon) 
curl https://www.esrl.noaa.gov/psd/data/correlation/indiamon.data > ${outpath}'indiamon.txt'

# Sahel rainfall (SahelRain) 
curl https://www.esrl.noaa.gov/psd/data/correlation/sahelrain.data > ${outpath}'sahelrain.txt'

# SW monsoon region rainfall (SWMonsoon) 
curl https://www.esrl.noaa.gov/psd/data/correlation/swmonsoon.data > ${outpath}'swmonsoon.txt'

# Northeast Brazil Rainfall Anomaly (BrazilRain) 
curl https://www.esrl.noaa.gov/psd/data/correlation/brazilrain.data > ${outpath}'brazilrain.txt'

# Solar Flux (Solar) 
curl https://www.esrl.noaa.gov/psd/data/correlation/solar.data > ${outpath}'solar.txt'

# Global Mean Land/Ocean Temperature Index (GMSST) 
curl https://www.esrl.noaa.gov/psd/data/correlation/gmsst.data > ${outpath}'gmsst.txt'

# Dipole Mode Index (DMI) - Index characterising the Indian Ocean Dipole
curl https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/Data/dmi.long.data > ${outpath}'dmi.txt'

echo 'Complete!'
