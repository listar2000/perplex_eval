# 🤖 Perplexity Evaluation with vLLM

A high-performance perplexity evaluation framework using vLLM for efficient LLM inference with data and tensor parallelism support.

## 📦 Installation

### Option 1: Using `uv` (Recommended) 🚀

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Option 2: Using pip 🐍

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Running Inference with SLURM

### Prerequisites

Make sure you have:
- Access to a SLURM cluster with GPU nodes
- Your dataset in Parquet format
- The model you want to evaluate available (locally or via HuggingFace)

### Customizing the Submit Script

The `submit.sh` script needs to be customized for your environment. Here are the key variables to modify:

```bash
# Update these paths to match your setup
#SBATCH --output=/path/to/your/output/wiki-10M.out
#SBATCH --error=/path/to/your/output/wiki-10M.err

# Adjust resource requirements based on your needs
#SBATCH --gres=gpu:a100:2        # Number and type of GPUs
#SBATCH --cpus-per-task=32       # CPU cores per task
#SBATCH --mem=64000              # Memory in MB
#SBATCH --time=2:00:00           # Time limit

# Update the working directory
cd /path/to/your/project

# Update dataset and output file paths
OUTPUT_FILE=output/wiki-10M.parquet
DATASET_FILE=path/to/your/dataset.parquet
```

### Submitting the Job

```bash
# Make the script executable
chmod +x submit.sh

# Submit the job
sbatch submit.sh
```

### Key Script Parameters

The `wiki_eval.py` script accepts several important parameters:

- `--model`: Model name or path (default: `Qwen/Qwen3-0.6B`)
- `--dp-size`: Data parallel size (should match number of GPUs)
- `--tp-size`: Tensor parallel size (usually 1 for single-node)
- `--dataset`: Path to your Parquet dataset
- `--prompt-column`: Column name containing the text prompts
- `--output-file`: Output Parquet file path
- `--debug-limit`: Limit number of prompts for testing purpose. Set it to `0` to run on the full dataset.

## ⚙️ Configuration

### vLLM Configuration (`vllm_config.yaml`)

The configuration file contains key vLLM settings:

```yaml
enforce_eager: true           # Forces eager execution mode
trust_remote_code: false      # Disables loading custom model code
max_num_seqs: 512            # Maximum concurrent sequences
gpu_memory_utilization: 0.9  # GPU memory usage limit (90%)
dtype: "bfloat16"            # Model precision (bfloat16 for efficiency)
```

This file also supports customization of other vLLM offline inference settings. See the [vLLM documentation](https://docs.vllm.ai/en/v0.7.2/api/offline_inference/llm.html) for more details.

**Key Settings Explained:**
- `enforce_eager`: Ensures deterministic execution for evaluation
- `gpu_memory_utilization`: Controls memory usage (0.9 = 90% of available GPU memory)
- `dtype`: Uses bfloat16 for faster inference with minimal precision loss
- `max_num_seqs`: Limits concurrent sequences to prevent OOM errors

## 🔄 Parallelism in LLM Inference

This project supports two types of parallelism for efficient large-scale inference:

### Data Parallelism (DP) 📊
- **What it does**: Distributes different data samples across multiple GPUs
- **When to use**: When your model fits on a single GPU but you have multiple GPUs
- **Benefits**: Linear scaling with number of GPUs, no model splitting overhead
- **Configuration**: Set `--dp-size` to number of available GPUs

### Tensor Parallelism (TP) 🧩
- **What it does**: Splits the model weights across multiple GPUs
- **When to use**: When your model is too large to fit on a single GPU
- **Benefits**: Enables inference with models larger than single GPU memory
- **Configuration**: Set `--tp-size` to number of GPUs needed for model

### Hybrid Approach 🔀
You can combine both approaches:
- Use TP to split large models across GPUs
- Use DP to process multiple batches in parallel

### Useful Resources 📚
- [vLLM Data Parallel Documentation](https://docs.vllm.ai/en/latest/examples/offline_inference/data_parallel.html)
- [vLLM Tensor Parallel Guide](https://docs.vllm.ai/en/latest/serving/distributed_serving.html#running-vllm-on-multiple-nodes)
- [Understanding Different Paradigms of Parallelism in LLMs](https://colossalai.org/docs/concepts/paradigms_of_parallelism/)

## 📊 Output Format

The script outputs a Parquet file with the following schema:
- `text_id`: Integer identifier for the source text
- `chunk_id`: Integer identifier for the text chunk
- `logprobs`: List of log probabilities for each token in the chunk

## 🐛 Debugging

For testing, use the `--debug-limit` parameter to process only a subset of your data:

```bash
python wiki_eval.py --debug-limit 100 --output-file output/debug.parquet
```
