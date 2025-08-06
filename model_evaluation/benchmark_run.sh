#!/bin/bash
#SBATCH --job-name=gemma_eval
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

MODEL="google/gemma-3-27b-it"
TASKS_ALL=(acp_bench anli blimp coqa eq_bench fda glue gsm8k ifeval mbpp mmlu humaneval truthfulqa)
HARD_TASKS=(c4 wikitext)

model_args_base="pretrained=$MODEL,tokenizer_mode=auto,\
tensor_parallel_size=2,data_parallel_size=1,\
gpu_memory_utilization=0.7,enforce_eager=False"

# loop over easy tasks
for task in "${TASKS_ALL[@]}"; do
  echo -e "\n▶▶ Running $task"
  lm_eval --model vllm \
    --model_args "$model_args_base" \
    --tasks "$task" \
    --batch_size auto \
    --confirm_run_unsafe_code \
    --verbosity INFO \
    --output_path "output/$task" || \
      echo "⚠️  $task failed (exit $?). Continuing."
done

# special handling for OOM-prone tasks
# reduce batch size and context length
model_args_hard="${model_args_base}"
for task in "${HARD_TASKS[@]}"; do
  echo -e "\n▶▶ Running (OOM‑safe) $task"
  lm_eval --model vllm \
    --model_args "$model_args_hard" \
    --tasks "$task" \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --verbosity INFO \
    --output_path "output/$task" || \
      echo "⚠️  $task (safe config) failed. Continuing."
done

