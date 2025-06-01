cd ~/repeng/slurm
sbatch --error=logs/repeng_$1.err --output=logs/repeng_$1.out --job-name=repeng job.slurm $1 $2 $3