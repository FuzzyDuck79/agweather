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
inpath=$1
qidfile=$2
qid_level=$3
qid_offset=$4
elevfile=$5
fformat=$6
outpath=$7

# Get geoid and corresponding elevation to process
qid_ix=${SGE_TASK_ID}
qid=$(sed ${qid_ix}'q;d' ${qidfile})
elev=$(sed '/'${qid}'/!d;s/.*,//' ${elevfile})


# *** modules required by the script
module purge
module load python3-anaconda/5.0.1 # CentOS7 queue
source activate anaconda3.6 # Load VirtualEnv with all my modules
# *** modules required by the script

python eto.py ${qid} ${qid_level} ${qid_offset} ${elev} ${fformat} ${inpath} ${outpath}

