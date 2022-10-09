#!/bin/bash

#SBATCH --array=0-47%24
#SBATCH --job-name=naive_sbatch
#SBATCH --mem=1G
#SBATCH --time=0-0:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --mail-user=rlwillia@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=Array_test.%A.out
#SBATCH --error=Array_test.%A.error

muN_values=( 2 )
burden_values=( 99 90 70 50 30 10 )
n_cassettes_values=( 1 2 )
K_values=( 1000000000 )
Vmax_x_values=( 0 1 5 )
abx_in_values=( 100 )
kPL_values=( 0.0001 )
selection_values=( additive )
muP_single_cassette_values=( 0 1 )
max_growth_values=( 20 )
n_sim_values=( 8 )


trial=${SLURM_ARRAY_TASK_ID}
muN=${muN_values[$(( trial % ${#muN_values[@]} ))]}
trial=$(( trial / ${#muN_values[@]} ))
burden=${burden_values[$(( trial % ${#burden_values[@]} ))]}
trial=$(( trial / ${#burden_values[@]} ))
n_cassettes=${n_cassettes_values[$(( trial % ${#n_cassettes_values[@]} ))]}
trial=$(( trial / ${#n_cassettes_values[@]} ))
K=${K_values[$(( trial % ${#K_values[@]} ))]}
trial=$(( trial / ${#K_values[@]} ))
Vmax_x=${Vmax_x_values[$(( trial % ${#Vmax_x_values[@]} ))]}
trial=$(( trial / ${#Vmax_x_values[@]} ))
abx_in=${abx_in_values[$(( trial % ${#abx_in_values[@]} ))]}
trial=$(( trial / ${#abx_in_values[@]} ))
kPL=${kPL_values[$(( trial % ${#kPL_values[@]} ))]}
trial=$(( trial / ${#kPL_values[@]} ))
selection=${selection_values[$(( trial % ${#selection_values[@]} ))]}
trial=$(( trial / ${#selection_values[@]} ))
muP_single_cassette=${muP_single_cassette_values[$(( trial % ${#muP_single_cassette_values[@]} ))]}
trial=$(( trial / ${#muP_single_cassette_values[@]} ))
max_growths=${max_growths_values[$(( trial % ${#max_growths_values[@]} ))]}
trial=$(( trial / ${#max_growths_values[@]} ))
n_sim=${n_sim_values[$(( trial % ${#n_sim_values[@]} ))]}

## source ../../prep.sh
## export OMP_NUM_THREADS=16
## export OMP_PROC_BIND=spread

## use ${burden}, ${K}, ${D}, ${n_cassettes}, ${selection}, ${n_sim} below
python -u naive_stoch_hpc_args.py $muN $burden $n_cassettes $K $Vmax_x $abx_in $kPL $selection $muP_single_cassette $max_growths $n_sim