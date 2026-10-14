#!/bin/bash
set -e
cd /home/whiteowl/last_shot/v2
export PYTHONPATH="/home/whiteowl/last_shot/v2:/home/whiteowl/last_shot:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========================================="
echo "  INVISIBLE LOGIC PIPELINE V2"
echo "  Resume-safe: skips completed stages"
echo "========================================="

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llm

# Stage 0: Base model new benchmarks (FalseQA, FaithEval, Sycophancy only)
if [ ! -f benchmarks/results/base_llama3b/.benchmark_complete ]; then
    echo "=== STAGE 0: Base Model New Benchmarks (FalseQA, FaithEval, Sycophancy) ==="
    python scripts/run_benchmarks.py \
        --model-path meta-llama/Llama-3.2-3B-Instruct \
        --output-dir benchmarks/results/base_llama3b \
        --benchmarks falseqa,faitheval,sycophancy
fi

# Stage 1: Download datasets
if [ ! -f data/raw/.download_v2_complete ]; then
    echo "=== STAGE 1: Download Datasets ==="
    python scripts/download_datasets.py
fi

# Stage 2: Prepare data (each script has its own internal marker)
echo "=== STAGE 2: Prepare Data ==="
python scripts/prepare_baseline_sft_data.py
python scripts/prepare_logic_sft_data.py

# Stage 3: DAPT — SKIP, reuse V1 model
if [ ! -f ../output/dapt/merged_model/config.json ]; then
    echo "ERROR: V1 DAPT merged model not found at ../output/dapt/merged_model/config.json"
    echo "Please run the V1 pipeline first or ensure the DAPT model is available."
    exit 1
else
    echo "=== STAGE 3: DAPT — Reusing V1 model ==="
fi

# Stage 4a: Baseline SFT
if [ ! -f output/baseline_sft/final_adapter/adapter_config.json ]; then
    echo "=== STAGE 4a: Baseline SFT ==="
    accelerate launch --config_file configs/accelerate_2gpu.yaml \
        scripts/run_sft.py --config baseline_sft_v2.yaml
fi

# Stage 4am: Merge Baseline SFT
if [ ! -f output/baseline_sft/merged_model/config.json ]; then
    echo "=== STAGE 4am: Merge Baseline SFT ==="
    python scripts/merge_adapter.py \
        --base-model /home/whiteowl/last_shot/output/dapt/merged_model \
        --adapter output/baseline_sft/final_adapter \
        --output output/baseline_sft/merged_model
fi

# Stage 4b: Logic SFT
if [ ! -f output/logic_sft/final_adapter/adapter_config.json ]; then
    echo "=== STAGE 4b: Logic SFT ==="
    accelerate launch --config_file configs/accelerate_2gpu.yaml \
        scripts/run_sft.py --config logic_sft_v2.yaml
fi

# Stage 4bm: Merge Logic SFT
if [ ! -f output/logic_sft/merged_model/config.json ]; then
    echo "=== STAGE 4bm: Merge Logic SFT ==="
    python scripts/merge_adapter.py \
        --base-model /home/whiteowl/last_shot/output/dapt/merged_model \
        --adapter output/logic_sft/final_adapter \
        --output output/logic_sft/merged_model
fi

# Stage 5a: Baseline DPO data prep
if [ ! -f data/processed/baseline_dpo/.prep_complete ]; then
    echo "=== STAGE 5a: Baseline DPO Data Prep ==="
    python scripts/prepare_dpo_data.py \
        --pipeline baseline \
        --model-path output/baseline_sft/merged_model
fi

# Stage 5b: Logic DPO data prep
if [ ! -f data/processed/logic_dpo/.prep_complete ]; then
    echo "=== STAGE 5b: Logic DPO Data Prep ==="
    python scripts/prepare_dpo_data.py \
        --pipeline logic \
        --model-path output/logic_sft/merged_model
fi

# Stage 6a: Baseline DPO
if [ ! -f output/baseline_dpo/final_adapter/adapter_config.json ]; then
    echo "=== STAGE 6a: Baseline DPO ==="
    accelerate launch --config_file configs/accelerate_2gpu.yaml \
        scripts/run_dpo.py --config baseline_dpo_v2.yaml
fi

# Stage 6am: Merge Baseline DPO
if [ ! -f output/baseline_dpo/merged_model/config.json ]; then
    echo "=== STAGE 6am: Merge Baseline DPO ==="
    python scripts/merge_adapter.py \
        --base-model output/baseline_sft/merged_model \
        --adapter output/baseline_dpo/final_adapter \
        --output output/baseline_dpo/merged_model
fi

# Stage 6b: Logic DPO
if [ ! -f output/logic_dpo/final_adapter/adapter_config.json ]; then
    echo "=== STAGE 6b: Logic DPO ==="
    accelerate launch --config_file configs/accelerate_2gpu.yaml \
        scripts/run_dpo.py --config logic_dpo_v2.yaml
fi

# Stage 6bm: Merge Logic DPO
if [ ! -f output/logic_dpo/merged_model/config.json ]; then
    echo "=== STAGE 6bm: Merge Logic DPO ==="
    python scripts/merge_adapter.py \
        --base-model output/logic_sft/merged_model \
        --adapter output/logic_dpo/final_adapter \
        --output output/logic_dpo/merged_model
fi

# Stage 7: Pre-HN Benchmarks (all 7 benchmarks on both models)
if [ ! -f benchmarks/results/.benchmark_pre_hn_v2_complete ]; then
    echo "=== STAGE 7: Benchmarks (pre H-Neurons) ==="
    python scripts/run_benchmarks.py \
        --model-path output/baseline_dpo/merged_model \
        --output-dir benchmarks/results/baseline_dpo
    python scripts/run_benchmarks.py \
        --model-path output/logic_dpo/merged_model \
        --output-dir benchmarks/results/logic_dpo
    touch benchmarks/results/.benchmark_pre_hn_v2_complete
fi

# Stage 8a: H-Neurons Baseline
if [ ! -f output/h_neurons_baseline/h_neurons.json ]; then
    echo "=== STAGE 8a: H-Neurons Baseline ==="
    python scripts/run_h_neurons.py \
        --model-path output/baseline_dpo/merged_model \
        --output-dir output/h_neurons_baseline
fi

# Stage 8b: H-Neurons Logic
if [ ! -f output/h_neurons_logic/h_neurons.json ]; then
    echo "=== STAGE 8b: H-Neurons Logic ==="
    python scripts/run_h_neurons.py \
        --model-path output/logic_dpo/merged_model \
        --output-dir output/h_neurons_logic
fi

# Stage 9: Post-HN Benchmarks (all 7 benchmarks on both models with H-Neurons)
if [ ! -f benchmarks/results/.benchmark_hn_v2_complete ]; then
    echo "=== STAGE 9: Final Benchmarks (with H-Neurons) ==="
    python scripts/run_benchmarks.py \
        --model-path output/baseline_dpo/merged_model \
        --h-neurons-path output/h_neurons_baseline/h_neurons.json \
        --output-dir benchmarks/results/baseline_dpo_hn
    python scripts/run_benchmarks.py \
        --model-path output/logic_dpo/merged_model \
        --h-neurons-path output/h_neurons_logic/h_neurons.json \
        --output-dir benchmarks/results/logic_dpo_hn
    touch benchmarks/results/.benchmark_hn_v2_complete
fi

# Stage 10: Over-Compliance Analysis
if [ ! -f benchmarks/results/.overcompliance_analysis_complete ]; then
    echo "=== STAGE 10: Over-Compliance Analysis ==="
    python scripts/run_overcompliance.py
fi

echo "========================================="
echo "  PIPELINE V2 COMPLETE!"
echo "  Results in benchmarks/results/"
echo "========================================="
