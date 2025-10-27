#!/bin/bash
#SBATCH --account=vjgo8416-asp-llm
#SBATCH --qos=turing
#SBATCH --time=5:0

set -e
module purge
module load baskerville
echo "Job ID: ${SLURM_JOB_ID}"

python -m venv --system-site-packages venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .

python examples/debate_sandwich.py --model-name mistralai/Mistral-7B-Instruct-v0.2 --num-agents 2 --num-rounds 3
