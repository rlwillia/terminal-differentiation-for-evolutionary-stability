#!/bin/bash

#SBATCH --array=0-575%48
#SBATCH --job-name=diff_sbatch
#SBATCH --mem=2G
#SBATCH --time=0-0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --mail-user=rlwillia@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=Array_test.%A.out
#SBATCH --error=Array_test.%A.error

muN_values=( 2 )
burden_values=( 99 90 70 50 30 10 )
n_cassettes_values=( 1 2 )
n_int_cassettes_values=( 2 )
cassettes_equal_values=( 0 )
K_values=( 1000000000 )
Vmax_x_values=( 0 )
abx_in_values=( 0 )
kPL_values=( 0.0001 0.000001 0.00000001 )
selection_values=( additive )
kdiff_values=( 0.2 0.4 0.6 0.8 1 1.2 )
split_cassettes_values=( 1 )
CIIE_values=( 0 )
max_growths_values=( 20 )
n_div_values=( 2 4 8 )
n_sim_values=( 8 )


trial=${SLURM_ARRAY_TASK_ID}
muN=${muN_values[$(( trial % ${#muN_values[@]} ))]}
trial=$(( trial / ${#muN_values[@]} ))
burden=${burden_values[$(( trial % ${#burden_values[@]} ))]}
trial=$(( trial / ${#burden_values[@]} ))
n_cassettes=${n_cassettes_values[$(( trial % ${#n_cassettes_values[@]} ))]}
trial=$(( trial / ${#n_cassettes_values[@]} ))
n_int_cassettes=${n_int_cassettes_values[$(( trial % ${#n_int_cassettes_values[@]} ))]}
trial=$(( trial / ${#n_int_cassettes_values[@]} ))
cassettes_equal=${cassettes_equal_values[$(( trial % ${#cassettes_equal_values[@]} ))]}
trial=$(( trial / ${#cassettes_equal_values[@]} ))
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
kdiff=${kdiff_values[$(( trial % ${#kdiff_values[@]} ))]}
trial=$(( trial / ${#kdiff_values[@]} ))
split_cassettes=${split_cassettes_values[$(( trial % ${#split_cassettes_values[@]} ))]}
trial=$(( trial / ${#split_cassettes_values[@]} ))
CIIE=${CIIE_values[$(( trial % ${#CIIE_values[@]} ))]}
trial=$(( trial / ${#CIIE_values[@]} ))
max_growths=${max_growths_values[$(( trial % ${#max_growths_values[@]} ))]}
trial=$(( trial / ${#max_growths_values[@]} ))
n_div=${n_div_values[$(( trial % ${#n_div_values[@]} ))]}
trial=$(( trial / ${#n_div_values[@]} ))
n_sim=${n_sim_values[$(( trial % ${#n_sim_values[@]} ))]}

## source ../../prep.sh
## export OMP_NUM_THREADS=16
## export OMP_PROC_BIND=spread

## use ${burden}, ${K}, ${D}, ${n_cassettes}, ${selection}, ${n_sim} below
python -u diff_select_stoch_hpc_args.py $muN $burden $n_cassettes $n_int_cassettes $cassettes_equal $K $Vmax_x $abx_in $kPL $selection $kdiff $split_cassettes $CIIE $max_growths $n_div $n_sim 