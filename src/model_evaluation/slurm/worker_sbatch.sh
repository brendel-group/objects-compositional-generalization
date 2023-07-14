#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --mem=16000               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --time=1-0:00            # Runtime in D-HH:MM
#SBATCH --output=logs/slog-%j.out  # File to which STDOUT will be written
#SBATCH --error=logs/slog-%j.err   # File to which STDERR will be written
#SBATCH --gres=gpu:1            # Request one GPU
#SBATCH --cpus-per-task=8
#SBATCH --job-name=objects_identifiability_evaluation
#SBATCH --exclude=slurm-bm-[08,09,17,19,26,39,49,50,18,27]

echo "$SLURM_ARRAY_TASK_ID"

COMMAND=$(cat <<-END
PYTHONPATH=$PYTHONPATH:$(pwd)/.. python -u evaluate.py \
  --inferred-latents=$1 \
  --ground-truth-latents=$2 \
  --n-slots=$3 \
  --evaluation-frequency=$4 \
  --categorical-dimensions $5
END
)

echo "Executing: "$COMMAND

SCRATCH_DIRECTORY="/scratch_local/$SLURM_JOB_USER-$SLURM_JOBID"
singularity exec -p --nv --bind /scratch_local/ --bind /mnt/qb/ $HOME/sifs/bfv.sif /bin/bash -c "$COMMAND"

echo "Done"