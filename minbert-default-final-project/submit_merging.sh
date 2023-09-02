#!/bin/bash
#SBATCH --job-name=MergingIntegration
#SBATCH -t 00:15:00
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=6G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=all
#SBATCH --mail-user=moataz.dawor@stud.uni-goettingen.de,lukas.niegsch@stud.uni-goettingen.de
#SBATCH --output=slurm_files/%x-slurm-%j.out
#SBATCH --error=slurm_files/%x-slurm-%j.err

# Load modules and environment.
module load anaconda3
module load cuda
source activate dl-gpu

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env
nvcc -V

# Execute the script.
python -B multitask_classifier.py --use_gpu --epochs=3 --option=finetune \
    --num_batches_para=3 \
    --num_batches_sst=3 \
    --num_batches_sts=3
