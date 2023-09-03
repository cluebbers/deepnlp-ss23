#!/bin/bash
#SBATCH --job-name=final
#SBATCH -t 12:00:00                  # estimated time # TODO: adapt to your needs
#SBATCH -p grete:shared              # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G A100:1                    # take 1 GPU, see https://www.hlrn.de/doc/display/PUB/GPU+Usage for more options
#SBATCH --mem-per-gpu=5G             # setting the right constraints for the splitted gpu partitions
#SBATCH --nodes=1                    # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --cpus-per-task=2            # number cores per task
#SBATCH --mail-type=all              # send mail when job begins and ends
#SBATCH --mail-user=finn.schmidt@stud.uni-goettingen.de # TODO: change this to your mailaddress!
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error

module load anaconda3
module load cuda
source activate dl-gpu # Or whatever you called your environment.

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env
nvcc -V

# Run the script:
python -u multitask_classifier.py --use_gpu --option finetune  --optimizer "adamw" --epochs 10 --one_embed True  --add_layers True --comment adam_add_layers_one_embed_smart --batch_size 64 --lr 1e-5

# Run the script with logger:
#python -u train_with_logger.py -l ~/${SLURM_JOB_NAME}_${SLURM_JOB_ID}  -t True -p True -d True -s True -f True
