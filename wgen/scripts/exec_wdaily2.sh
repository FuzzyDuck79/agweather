#!/bin/sh

# *** Grid engine formalities ***
#$ -S /bin/sh  -cwd 
# *** Grid engine formalities ***

# *** Grid engine echos and important shell variables*** 
echo 'TMPDIR' ${TMPDIR}
echo 'QUEUE' ${QUEUE}
echo 'SGE_TASK_ID' ${SGE_TASK_ID}
echo 'SGE_O_WORKDIR' ${SGE_O_WORKDIR}
echo 'SGE_O_HOME' ${SGE_O_HOME}
echo 'HOST' ${HOST}
echo 'SLOTS' ${NSLOTS}
# *** Grid engine echos and important shell variables*** 

# Collect inputs
sttgsfile=$1
qidfile=$2
outpath=$3

# Get file to process
filenum=${SGE_TASK_ID}
qid=$(sed ${filenum}'q;d' ${qidfile})

# *** modules required by the script
module purge
module load python3-anaconda/5.0.1 # CentOS7 queue only
source activate anaconda3.6 # Load VirtualEnv with all my modules
# *** modules required by the script

echo python wdaily2.py ${sttgsfile} ${outpath}'hist/' ${outpath}'stoc/' ${qid}
python wdaily2.py ${sttgsfile} ${outpath}'hist/' ${outpath}'stoc/' ${qid}

