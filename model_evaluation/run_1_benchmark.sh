#!/bin/bash
#SBATCH --job-name=eval_all_models
#SBATCH --gres=gpu:a100:2
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=general

set +e  # continue even on errors

export HF_HOME=/net/scratch2/siyangwu/huggingface_model/.cache/hf_home
export HF_HUB_CACHE=/net/scratch2/siyangwu/huggingface_model/.cache/hf_hub_cache
export HF_XET_CACHE=/net/scratch2/siyangwu/huggingface_model/.cache/hf_xet_cache

export HF_ALLOW_CODE_EVAL=1

# Task to evaluate (pass as argument)
TASK="wikiqa"

if [ -z "$TASK" ]; then
  echo "❌ No task provided. Usage: sbatch run_all_models.sh <task_name>"
  exit 1
fi

MODELS=(
  meta-llama/Llama-3.2-1B-Instruct
  meta-llama/Llama-3.2-3B-Instruct
  meta-llama/Llama-3.1-8B-Instruct
  google/gemma-3-4b-it
  google/gemma-3-12b-it
  google/gemma-3-27b-it
  mistralai/Mistral-7B-Instruct-v0.3
  deepseek-ai/deepseek-llm-7b-chat
  Qwen/Qwen3-0.6B
  Qwen/Qwen3-8B
  tiiuae/falcon-rw-1b
  EleutherAI/pythia-1b
  EleutherAI/pythia-6.9b-v0
  EleutherAI/pythia-12b-deduped
  mosaicml/mpt-7b-instruct
  microsoft/Phi-4-mini-instruct
  microsoft/phi-4
  microsoft/Phi-3-mini-4k-instruct
  01-ai/Yi-1.5-9B-Chat
  01-ai/Yi-1.5-6B-Chat
)

for MODEL in "${MODELS[@]}"; do
  echo -e "\n▶▶ Running $TASK on $MODEL"
  model_args_base="pretrained=$MODEL,tokenizer_mode=auto,\
tensor_parallel_size=2,data_parallel_size=1,\
gpu_memory_utilization=0.7,enforce_eager=False"

  lm_eval --model vllm \
    --model_args "$model_args_base" \
    --tasks "$TASK" \
    --batch_size auto \
    --confirm_run_unsafe_code \
    --verbosity INFO \
    --output_path "output/${TASK}/$(basename "$MODEL")" || \
      echo "⚠️  $MODEL failed on $TASK (exit $?). Continuing."
done
