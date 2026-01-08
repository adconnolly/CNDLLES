#PBS -q casper
#PBS -l select=1:ncpus=1:ngpus=1:mem=10GB:gpu_type=l40

source ~/env_escnn
conda activate escnn

python baseline_noDataAug_extrap_test270.py
