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
inpath=$2

# *** modules required by the script
module purge
module load python3-anaconda/5.0.1 # CentOS7 queue only
source activate anaconda3.6 # Load VirtualEnv with all my modules
# *** modules required by the script

echo python wplot2.py ${sttgsfile} ${inpath}
python wplot2.py ${sttgsfile} ${inpath}

