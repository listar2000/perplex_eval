#!/bin/bash
#SBATCH --job-name=redpajama
#SBATCH --output=/net/scratch2/listar2000/perplex_eval/slurm/Llama-3.1-8B-Instruct-redpajama-50k/%j-%x.out
#SBATCH --error=/net/scratch2/listar2000/perplex_eval/slurm/Llama-3.1-8B-Instruct-redpajama-50k/%j-%x.err
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64000
#SBATCH --partition=general

# this number should usually agree with the number of A100 above
NUM_GPUS=2
OUTPUT_FOLDER=output/
DATASET_FILE=data/redpajama-subset-50k.parquet
VLLM_CONFIG_PATH=vllm_config.yaml

nvidia-smi

cd /net/scratch2/listar2000/perplex_eval

source shell/star_setup.sh
source .venv/bin/activate

python -m src.evaluate.eval_document \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tp_size $NUM_GPUS \
    --dataset_path $DATASET_FILE \
    --vllm_config_path $VLLM_CONFIG_PATH \
    --save_folder $OUTPUT_FOLDER