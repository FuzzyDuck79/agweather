#!/bin/sh

# Submission script to generate raster masks
# Mutahar Chalmers, October 2019 

if [ $# != '4' ]; then
	echo -e '\nArguments: <inpath> [country | subregion | region] <gridsize> <outpath>' 
	exit 3
fi


inpath=$1
resolution=$2
gridsize=$3
outpath=$4

# Define location of location file
locationfile=${inpath}'/'${resolution}'/'${resolution}'.txt'

n_locations=`wc -l < ${locationfile}`
echo 'Processing '${n_locations}' locations at '${gridsize}' resolution...'

############################################################################
# INPUT SECTION
############################################################################

# Paths to input and output directories
logpath=${outpath}'/SGE_LOGS/'

# Setting up the output and log directories
logpath=${logpath}
if [ ! -e ${logpath} ]; then mkdir -p ${logpath}; fi
if [ ! -e ${outpath}'code/' ]; then mkdir -p ${outpath}'code/'; fi
if [ ! -e ${outpath}'region/' ]; then mkdir -p ${outpath}'region/'; fi
if [ ! -e ${outpath}'subregion/' ]; then mkdir -p ${outpath}'subregion/'; fi
if [ ! -e ${outpath}'country/' ]; then mkdir -p ${outpath}'country/'; fi

# Copy scripts to output directory for reference
cp *.py ${outpath}'code/'
cp *.sh ${outpath}'code/'
############################################################################

# Run the job
job='geomask'
jid=`qsub -q long.q -P AGRI -o ${logpath}geomask.o -e ${logpath}geomask.e -l mf=2G -t 1-${n_locations} -N ${job} ./exec_geomask.sh ${inpath} ${resolution} ${gridsize} ${outpath}`

exit 0

