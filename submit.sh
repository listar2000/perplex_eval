#!/bin/bash
#SBATCH --job-name=redpajama
#SBATCH --output=/net/scratch2/listar2000/perplex_eval/slurm/wiki-10M.out
#SBATCH --error=/net/scratch2/listar2000/perplex_eval/slurm/wiki-10M.err
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64000
#SBATCH --partition=general

# this number should usually agree with the number of A100 above
NUM_GPUS=2
OUTPUT_FILE=output/wiki-10M.parquet
DATASET_FILE=data/wiki/chunked_texts_df.parquet

nvidia-smi

cd /net/scratch2/listar2000/perplex_eval

source setup.sh
source .venv/bin/activate

python wiki_eval.py \
    --model Qwen/Qwen3-0.6B \
    --dp-size $NUM_GPUS \
    --tp-size 1 \
    --dataset $DATASET_FILE \
    --prompt-column chunk_text \
    --output-file $OUTPUT_FILE \

# inspect the output parquet file
python inspect_parquet.py --parquet-file $OUTPUT_FILE