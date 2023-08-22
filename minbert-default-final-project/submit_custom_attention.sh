#!/bin/bash
#SBATCH --job-name=BertSelfAttention
#SBATCH -t 03:00:00
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=6G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=all
#SBATCH --mail-user=moataz.dawor@stud.uni-goettingen.de,lukas.niegsch@stud.uni-goettingen.de
#SBATCH --output=slurm_files/slurm-%x-%j.out
#SBATCH --error=slurm_files/slurm-%x-%j.err

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
CUSTOM_ATTENTION="BertSelfAttention"
python -B multitask_classifier.py --use_gpu --epochs=10 --lr=1e-5 --option=finetune --logdir=$CUSTOM_ATTENTION --save=False --custom_attention=$CUSTOM_ATTENTION \
    --sst_dev_out="predictions/$CUSTOM_ATTENTION-sst-dev-output.csv" \
    --sst_test_out="predictions/$CUSTOM_ATTENTION-sst-test-output.csv" \
    --para_dev_out="predictions/$CUSTOM_ATTENTION-para-dev-output.csv" \
    --para_test_out="predictions/$CUSTOM_ATTENTION-para-test-output.csv" \
    --sts_dev_out="predictions/$CUSTOM_ATTENTION-sts-dev-output.csv" \
    --sts_test_out="predictions/$CUSTOM_ATTENTION-sts-test-output.csv" \
