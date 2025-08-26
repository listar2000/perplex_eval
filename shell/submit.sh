#!/bin/bash
#SBATCH --job-name=falcon-7b-instruct
#SBATCH --output=/to/your/own/path/slurm/Falcon-7B-Instruct-redpajama-50k/%j-%x.out
#SBATCH --error=/to/your/own/path/slurm/Falcon-7B-Instruct-redpajama-50k/%j-%x.err
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64000
#SBATCH --partition=general

# this number should usually agree with the number of A100 above
NUM_GPUS=2
OUTPUT_FOLDER=/to/your/own/path/output/
DATASET_FILE=/to/your/own/path/data/redpajama-subset-50k.parquet
VLLM_CONFIG_PATH=vllm_config.yaml

nvidia-smi

cd /to/your/own/path

source shell/your_setup_shell.sh
# if you are using a virtual environment, you can use the following command to activate it
source .venv/bin/activate

python -m src.evaluate.eval_vllm \
    --model Qwen/Qwen3-0.6B \
    --tp_size $NUM_GPUS \
    --dataset_path $DATASET_FILE \
    --vllm_config_path $VLLM_CONFIG_PATH \
    --save_folder $OUTPUT_FOLDER