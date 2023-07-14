basedir="/home/bethge/rzimmermann/work/objects_data/model_metrics/spriteworld_4"

mkdir logs -p

for fn in $basedir/additive/*.npy; do
  echo "Starting evaluation for $fn"
  sbatch slurm/worker_sbatch.sh $fn $fn 4 5 "4"
done

sbatch slurm/worker_sbatch.sh $fn $fn 4 5 "4"