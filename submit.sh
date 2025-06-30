#!/bin/bash
#SBATCH --job-name=redpajama
#SBATCH --output=/net/scratch2/listar2000/perplex_eval/slurm/redpajama-1b-inference.out
#SBATCH --error=/net/scratch2/listar2000/perplex_eval/slurm/redpajama-1b-inference.err
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64000
#SBATCH --partition=general

NUM_GPUS=2

cd /net/scratch2/listar2000/perplex_eval

source .venv/bin/activate
source setup.sh

# Print diagnostic info
echo "Running on host $(hostname)"
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-default}"  # may be empty if not set by cluster
nvidia-smi || echo "nvidia-smi not available"

echo "Using ${NUM_GPUS} GPUs"

# -----------------------------------------------------------------------------
# Config — adjust paths to your data and output location
# -----------------------------------------------------------------------------
MODEL_NAME="Qwen/Qwen3-0.6B"
DATA_DIR="data/redpajama-1b"
FILE_PATTERN="0.parquet"
OUTPUT_PARQUET="output/results.parquet"
BATCH_SIZE=8
SERVER_PORT=8000
SERVER_URL="http://localhost:${SERVER_PORT}/v1"

# -----------------------------------------------------------------------------
# 1) Launch the vLLM OpenAI server *in the background*
# -----------------------------------------------------------------------------

echo "Starting vLLM server …"
python -m vllm.entrypoints.openai.api_server \
  --model ${MODEL_NAME} \
  --tensor-parallel-size ${NUM_GPUS} \
  --enable-chunked-prefill \
  --long-prefill-token-threshold 2048 \
  --max-num-partial-prefills 2 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.9 \
  --port ${SERVER_PORT} \
  > ./slurm/vllm_server.log 2>&1 &
SERVER_PID=$!

echo "vLLM server PID: ${SERVER_PID}"

# Give the server some time to load the model weights before hitting it.
# Poll the /models endpoint until it responds.
echo "Waiting for vLLM server to be ready..."
while true; do
    if curl -s "${SERVER_URL}/models" > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    echo "Server not ready yet, waiting 5 seconds..."
    sleep 5
done

# -----------------------------------------------------------------------------
# 2) Run the client script – this sends prompts to the server we just started.
# -----------------------------------------------------------------------------

echo "Launching log‑prob extraction client …"
python run_redpajama_1b_inference.py \
  --model ${MODEL_NAME} \
  --data-dir ${DATA_DIR} \
  --file-pattern "${FILE_PATTERN}" \
  --output ${OUTPUT_PARQUET} \
  --batch-size ${BATCH_SIZE} \
  --server-url ${SERVER_URL} \
  --api-key EMPTY \
  --limit-docs 20

# -----------------------------------------------------------------------------
# 3) Clean‑up – terminate the server gracefully (optional)
# -----------------------------------------------------------------------------

echo "Client finished; stopping server (PID ${SERVER_PID}) …"
kill ${SERVER_PID}
wait ${SERVER_PID} || true

echo "Job completed successfully."