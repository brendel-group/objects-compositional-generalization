basedir="/home/bethge/rzimmermann/work/objects_data/model_metrics/clevr_6"

mkdir logs -p

for fn in $basedir/additive/*_inf_latents.npy; do
  echo "Starting evaluation for $fn"
  sbatch slurm/worker_sbatch.sh $fn "$basedir/clevr_gt_latents.npy" 6 5 "2 3 4 5"
done
