#!/bin/sh

# Submission script for wgen, stochastic weather generator for agriculture models
# Mutahar Chalmers, 2018-9

if [ $# != '4' ]; then
	echo -e '\nArguments: <model e.g. BRAG> <settings> <description> <N_qids>' 
	exit 3
fi

model=$1
sttgsfile=$2
desc=$3
N_qids=$4

#########################################################################################
# Define paths and create directories
#########################################################################################

# Paths to input, log and output directories
outpath='/mnt/gpfs/nobackup/agri/models/'${model}'/weather/stochastic/'${desc}'/'
histpath=${outpath}'hist/'
stocpath=${outpath}'stoc/'
analpath=${outpath}'analytics/'
codepath=${outpath}'code/'
plotpath=${outpath}'plots/'
logpath=${outpath}'SGE_LOGS/'


if [ ! -e ${outpath} ]; then mkdir -p ${outpath}; fi
if [ ! -e ${histpath} ]; then mkdir -p ${histpath}; fi
if [ ! -e ${stocpath} ]; then mkdir -p ${stocpath}; fi
if [ ! -e ${stocpath}'daily/' ]; then mkdir -p ${stocpath}'daily/'; fi
if [ ! -e ${stocpath}'ssfs/' ]; then mkdir -p ${stocpath}'ssfs/'; fi
if [ ! -e ${stocpath}'histdaily/' ]; then mkdir -p ${stocpath}'histdaily/'; fi

if [ ! -e ${analpath} ]; then mkdir -p ${analpath}; fi
if [ ! -e ${analpath}'raw/' ]; then mkdir -p ${analpath}'raw/'; fi
if [ ! -e ${codepath} ]; then mkdir -p ${codepath}; fi
if [ ! -e ${plotpath} ]; then mkdir -p ${plotpath}; fi

if [ ! -e ${e_logpath} ]; then mkdir -p ${e_logpath}; fi
if [ ! -e ${o_logpath} ]; then mkdir -p ${o_logpath}; fi
#########################################################################################

qidfile=${histpath}'qids.txt'

# Copy Python and shell scripts, as well as settings file to output directory
cp wgen/*.py ${codepath}
cp scripts/*.sh ${codepath}
cp ${sttgsfile} ${outpath}

# Seasonal simulation
job1='wgen'
echo 'Running '${job1}
jid1=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=20G -N ${job1} ./scripts/exec_wgen.sh ${sttgsfile} ${outpath}`

# Daily downscaling, ET0 and SSF calculation
job2='wdaily2'
echo 'Running '${job2}
#jid2=`qsub -q long.q -P AGRI -o ${logpath}wdaily.o -e ${logpath}wdaily.e -l mf=5G -t 1-${N_qids} -N ${job2} -hold_jid ${job1} ./exec_wdaily2.sh ${sttgsfile} ${qidfile} ${outpath}`

# Analytics - Reassemble individual qid files into DataFrames by statistic
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_hist_quant_d' -hold_jid ${job2} ./exec_wcombine2.sh 'hist_m_day_quant' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_hist_clima_m' -hold_jid ${job2} ./exec_wcombine2.sh 'hist_m_agg_clima' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_histmax_clima_m' -hold_jid ${job2} ./exec_wcombine2.sh 'hist_m_max_clima' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_histmin_clima_m' -hold_jid ${job2} ./exec_wcombine2.sh 'hist_m_min_clima' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_histvar_clima_m' -hold_jid ${job2} ./exec_wcombine2.sh 'hist_m_var_clima' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_hist_quant_m' -hold_jid ${job2} ./exec_wcombine2.sh 'hist_m_agg_quant' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_histmax_quant_m' -hold_jid ${job2} ./exec_wcombine2.sh 'hist_m_max_quant' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_histmin_quant_m' -hold_jid ${job2} ./exec_wcombine2.sh 'hist_m_min_quant' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_histvar_quant_m' -hold_jid ${job2} ./exec_wcombine2.sh 'hist_m_var_quant' ${outpath}`

#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_stoc_quant_d' -hold_jid ${job2} ./exec_wcombine2.sh 'stoc_m_day_quant' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_stoc_clima_m' -hold_jid ${job2} ./exec_wcombine2.sh 'stoc_m_agg_clima' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_stocmax_clima_m' -hold_jid ${job2} ./exec_wcombine2.sh 'stoc_m_max_clima' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_stocmin_clima_m' -hold_jid ${job2} ./exec_wcombine2.sh 'stoc_m_min_clima' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_stocvar_clima_m' -hold_jid ${job2} ./exec_wcombine2.sh 'stoc_m_var_clima' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_stoc_quant_m' -hold_jid ${job2} ./exec_wcombine2.sh 'stoc_m_agg_quant' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_stocmax_quant_m' -hold_jid ${job2} ./exec_wcombine2.sh 'stoc_m_max_quant' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_stocmin_quant_m' -hold_jid ${job2} ./exec_wcombine2.sh 'stoc_m_min_quant' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_stocvar_quant_m' -hold_jid ${job2} ./exec_wcombine2.sh 'stoc_m_var_quant' ${outpath}`

#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_hist_xcorr_m' -hold_jid ${job2} ./exec_wcombine2.sh 'hist_m_agg_xcorr' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_histmax_xcorr_m' -hold_jid ${job2} ./exec_wcombine2.sh 'hist_m_max_xcorr' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_histmin_xcorr_m' -hold_jid ${job2} ./exec_wcombine2.sh 'hist_m_min_xcorr' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_histvar_xcorr_m' -hold_jid ${job2} ./exec_wcombine2.sh 'hist_m_var_xcorr' ${outpath}`

#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_stoc_xcorr_m' -hold_jid ${job2} ./exec_wcombine2.sh 'stoc_m_agg_xcorr' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_stocmax_xcorr_m' -hold_jid ${job2} ./exec_wcombine2.sh 'stoc_m_max_xcorr' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_stocmin_xcorr_m' -hold_jid ${job2} ./exec_wcombine2.sh 'stoc_m_min_xcorr' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_stocvar_xcorr_m' -hold_jid ${job2} ./exec_wcombine2.sh 'stoc_m_var_xcorr' ${outpath}`

#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_ks_m' -hold_jid ${job2} ./exec_wcombine2.sh 'm_agg_ks' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_ksmax_m' -hold_jid ${job2} ./exec_wcombine2.sh 'm_max_ks' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_ksmin_m' -hold_jid ${job2} ./exec_wcombine2.sh 'm_min_ks' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_ksvar_m' -hold_jid ${job2} ./exec_wcombine2.sh 'm_var_ks' ${outpath}`

#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_hist_scorr_m' -hold_jid ${job2} ./exec_wcombine2.sh 'hist_scorr' ${outpath}`
#jid3=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N 'anal_stoc_scorr_m' -hold_jid ${job2} ./exec_wcombine2.sh 'stoc_scorr' ${outpath}`
    
# Plot analytics
job4='wplot2'
echo 'Running '${job4}
#jid4=`qsub -q long.q -P AGRI -o ${logpath} -e ${logpath} -l mf=5G -N ${job4} -hold_jid 'anal_*' ./exec_wplot2.sh ${sttgsfile} ${outpath}`

exit 0

