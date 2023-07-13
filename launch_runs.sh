#!/bin/bash
# ======= SLURM OPTIONS ======= (user input required)
### See inline comments for what each option means
#SBATCH --partition=GPU
### Set the job name
#SBATCH --job-name=socratic_exp
### Specify the # of cpus for your job.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=30gb
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
### pass the full environment
#SBATCH --export=ALL
#SBATCH --output=%j.o
#SBATCH --error=%j.e
# ===== END SLURM OPTIONS =====
### IMPORTANT: load Python 3 environment with Pytorch and cuda enabled
#module load pytorch/1.5.1-anaconda3-cuda10.2
source activate socratic_env
#module load pytorch/1.6.0-cuda10.2 
echo "loaded module"
### Go to the directory of the sample.sh file
cd $SLURM_SUBMIT_DIR
### Make a folder for job_logs if one doesn't exist
mkdir -p job_logs
### Run the python file
echo "running code"


# multiple responses (all possible socratic utterances) with instruction mode
python -u run_socratic_benchmark_metrics.py --generation_mode multiple
# instruction mode with a single response 
# python run_socratic_benchmark_metrics.py
echo "finished running"
cd $SLURM_SUBMIT_DIR
### move the log files inside the folder
mv $SLURM_JOB_ID.o job_logs/$SLURM_JOB_ID.o
mv $SLURM_JOB_ID.e job_logs/$SLURM_JOB_ID.e