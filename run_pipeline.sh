#!/bin/bash
set -e
cd /home/whiteowl/last_shot
export PYTHONPATH="/home/whiteowl/last_shot:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========================================="
echo "  INVISIBLE LOGIC PIPELINE"
echo "  Resume-safe: skips completed stages"
echo "========================================="

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llm

# Stage 0: Base model benchmark
if [ ! -f benchmarks/results/base_llama3b/.benchmark_complete ]; then
    echo "=== STAGE 0: Base Model Benchmark ==="
    python scripts/run_benchmarks.py \
        --model-path meta-llama/Llama-3.2-3B-Instruct \
        --output-dir benchmarks/results/base_llama3b
fi

# Stage 1: Download datasets
if [ ! -f data/raw/.download_complete ]; then
    echo "=== STAGE 1: Download Datasets ==="
    python scripts/download_datasets.py
fi

# Stage 2: Prepare data
echo "=== STAGE 2: Prepare Data ==="
python scripts/prepare_dapt_data.py
python scripts/prepare_baseline_sft_data.py
python scripts/prepare_logic_sft_data.py

# Stage 3: DAPT
if [ ! -d output/dapt/final_adapter ]; then
    echo "=== STAGE 3: DAPT ==="
    accelerate launch --config_file configs/accelerate_2gpu.yaml scripts/run_dapt.py
fi

# Stage 3m: Merge DAPT
if [ ! -d output/dapt/merged_model ]; then
    echo "=== STAGE 3m: Merge DAPT ==="
    python scripts/merge_adapter.py \
        --base-model meta-llama/Llama-3.2-3B-Instruct \
        --adapter output/dapt/final_adapter \
        --output output/dapt/merged_model
fi

# Stage 4a: Baseline SFT
if [ ! -d output/baseline_sft/final_adapter ]; then
    echo "=== STAGE 4a: Baseline SFT ==="
    accelerate launch --config_file configs/accelerate_2gpu.yaml \
        scripts/run_sft.py --config baseline_sft_config.yaml
fi

# Stage 4am: Merge Baseline SFT
if [ ! -d output/baseline_sft/merged_model ]; then
    echo "=== STAGE 4am: Merge Baseline SFT ==="
    python scripts/merge_adapter.py \
        --base-model output/dapt/merged_model \
        --adapter output/baseline_sft/final_adapter \
        --output output/baseline_sft/merged_model
fi

# Stage 4b: Logic SFT
if [ ! -d output/logic_sft/final_adapter ]; then
    echo "=== STAGE 4b: Logic SFT ==="
    accelerate launch --config_file configs/accelerate_2gpu.yaml \
        scripts/run_sft.py --config logic_sft_config.yaml
fi

# Stage 4bm: Merge Logic SFT
if [ ! -d output/logic_sft/merged_model ]; then
    echo "=== STAGE 4bm: Merge Logic SFT ==="
    python scripts/merge_adapter.py \
        --base-model output/dapt/merged_model \
        --adapter output/logic_sft/final_adapter \
        --output output/logic_sft/merged_model
fi

# Stage 5a: Baseline DPO data
python scripts/prepare_baseline_dpo_data.py

# Stage 5b: Logic DPO data
python scripts/prepare_logic_dpo_data.py

# Stage 6a: Baseline DPO
if [ ! -d output/baseline_dpo/final_adapter ]; then
    echo "=== STAGE 6a: Baseline DPO ==="
    accelerate launch --config_file configs/accelerate_2gpu.yaml \
        scripts/run_dpo.py --config baseline_dpo_config.yaml
fi

# Stage 6am: Merge Baseline DPO
if [ ! -d output/baseline_dpo/merged_model ]; then
    echo "=== STAGE 6am: Merge Baseline DPO ==="
    python scripts/merge_adapter.py \
        --base-model output/baseline_sft/merged_model \
        --adapter output/baseline_dpo/final_adapter \
        --output output/baseline_dpo/merged_model
fi

# Stage 6b: Logic DPO
if [ ! -d output/logic_dpo/final_adapter ]; then
    echo "=== STAGE 6b: Logic DPO ==="
    accelerate launch --config_file configs/accelerate_2gpu.yaml \
        scripts/run_dpo.py --config logic_dpo_config.yaml
fi

# Stage 6bm: Merge Logic DPO
if [ ! -d output/logic_dpo/merged_model ]; then
    echo "=== STAGE 6bm: Merge Logic DPO ==="
    python scripts/merge_adapter.py \
        --base-model output/logic_sft/merged_model \
        --adapter output/logic_dpo/final_adapter \
        --output output/logic_dpo/merged_model
fi

# Stage 7: Benchmarks (pre H-Neurons)
if [ ! -f benchmarks/results/.benchmark_pre_hn_complete ]; then
    echo "=== STAGE 7: Benchmarks (pre H-Neurons) ==="
    python scripts/run_benchmarks.py \
        --model-path output/baseline_dpo/merged_model \
        --output-dir benchmarks/results/baseline_dpo
    python scripts/run_benchmarks.py \
        --model-path output/logic_dpo/merged_model \
        --output-dir benchmarks/results/logic_dpo
    touch benchmarks/results/.benchmark_pre_hn_complete
fi

# Stage 8a: H-Neurons on Baseline
if [ ! -f output/h_neurons_baseline/h_neurons.json ]; then
    echo "=== STAGE 8a: H-Neurons Baseline ==="
    python scripts/run_h_neurons.py \
        --model-path output/baseline_dpo/merged_model \
        --output-dir output/h_neurons_baseline
fi

# Stage 8b: H-Neurons on Logic
if [ ! -f output/h_neurons_logic/h_neurons.json ]; then
    echo "=== STAGE 8b: H-Neurons Logic ==="
    python scripts/run_h_neurons.py \
        --model-path output/logic_dpo/merged_model \
        --output-dir output/h_neurons_logic
fi

# Stage 9: Final Benchmarks with H-Neurons
if [ ! -f benchmarks/results/.benchmark_hn_complete ]; then
    echo "=== STAGE 9: Final Benchmarks (with H-Neurons) ==="
    python scripts/run_benchmarks.py \
        --model-path output/baseline_dpo/merged_model \
        --h-neurons-path output/h_neurons_baseline/h_neurons.json \
        --output-dir benchmarks/results/baseline_dpo_hn
    python scripts/run_benchmarks.py \
        --model-path output/logic_dpo/merged_model \
        --h-neurons-path output/h_neurons_logic/h_neurons.json \
        --output-dir benchmarks/results/logic_dpo_hn
    touch benchmarks/results/.benchmark_hn_complete
fi

echo "========================================="
echo "  PIPELINE COMPLETE!"
echo "  Results in benchmarks/results/"
echo "========================================="
