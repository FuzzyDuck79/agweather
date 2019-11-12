#!/bin/sh

# Generate analytics for my stochastic weather generator
if [ $# != '4' ]; then echo -e '\nUsage: ./qsub.sh <inpath> <qidfile> <elevationfile> <outpath>'; exit 3; fi 

# Hard-coded file format - Parquet preferred
format='parquet'

inpath=$1
qidfile=$2
qid_level=$3
qid_offset=$4
elevfile=$5
fformat=$6
outpath=$7

# Paths to input and output directories
logpath=${outpath}'SGE_LOGS/'

# Setting up the output and log directories
if [ ! -e $outpath ]; then mkdir -p $outpath; fi
if [ ! -e $logpath ]; then mkdir -p $logpath; fi

n_qids=`wc -l < ${qids}`
echo 'Processing '${n_qids}' qids...'

job='et0'
echo 'Running '$job
jid1=`qsub -q long.q -P AGRI -o $logpath -e $logpath -l mf=1G -t 1-${n_qids} -N ${job} ./exec_eto.sh ${inpath} ${qidfile} ${qid_level} ${qid_offset} ${elevfile} ${fformat} ${outpath}`

exit 0

