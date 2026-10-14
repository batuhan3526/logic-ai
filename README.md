# Logic Data Helps, Neuron Suppression Does Not

Code and raw artifacts for:

> S. Topal and B. Bayraktar, "Logic Data Helps, Neuron Suppression Does Not: A Controlled Ablation of Hallucination Mitigation During Domain Adaptation of a 3B Language Model," submitted to *IEEE Transactions on Audio, Speech and Language Processing*, 2026.

The study adapts Llama-3.2-3B-Instruct to finance and cryptocurrency through DAPT, SFT and DPO in two arms. The arms differ only in the SFT recipe: domain QA versus natural-language logical-inference data (LogiQA 2.0, ReClor, FOLIO). Inference-time suppression of hallucination-associated neurons (H-Neurons) is then applied to the final checkpoint of each arm.

## Contents

| Path | What it is |
|---|---|
| `v2/` | **The pipeline reported in the paper (V2).** |
| `v2/scripts/` | Data preparation, SFT, preference-data generation and DPO, H-Neuron discovery and suppression, benchmarks |
| `v2/scripts/run_benchmarks.py` | TruthfulQA, HaluEval, DMind runners and the in-house sycophancy probe (`SYCOPHANCY_QUESTIONS`, 175 items) |
| `v2/configs/` | Training configurations for both arms |
| `v2/run_pipeline.sh` | Resume-safe end-to-end driver |
| `v2/benchmarks/results/*/results.json` | Raw benchmark outputs for every configuration |
| `v2/output/h_neurons_{baseline,logic}/` | H-Neuron discovery artifacts: selected neurons, probe and gate statistics, α-sweep checkpoints |
| `scripts/`, `configs/`, `run_pipeline.sh`, `benchmarks/`, `output/` | First attempt (V1), described in Section III-B. V2 reuses its DAPT checkpoint |
| `finance_qwen3/benchmarks/` | Raw outputs of the exploratory finance experiment on Qwen3-1.7B |
| `analysis/gen_tables.py` | Recomputes the result tables, Wilson intervals and z-tests from the raw outputs above |

## Reproducing the tables

```bash
python3 analysis/gen_tables.py   # writes analysis/TABLES_FROM_RAW.md
```

The script uses only the Python standard library and the JSON files in this repository.

## Re-running the pipeline

The scripts were run from `/home/whiteowl/last_shot` (V1) and `/home/whiteowl/last_shot/v2` (V2). These absolute paths are hard-coded in `run_pipeline.sh` and `scripts/utils.py`, so change them before running. Hardware: two RTX 3090 (24 GB) GPUs. V2 takes about 18 hours.

```bash
pip install -r requirements.txt
bash run_pipeline.sh        # V1, for the DAPT checkpoint that V2 reuses
bash v2/run_pipeline.sh     # V2
```

Datasets are **not** redistributed here. `scripts/download_datasets.py` and `v2/scripts/download_datasets.py` download them from their original sources, under their own licences. Model weights are not distributed either. Data preparation uses seed 42, and training seeds were not fixed.

## Known limitations

See Section IX of the paper. In short: TruthfulQA was part of both arms' training data, every configuration was trained once, and per-item outputs were not retained.

## Licence

Required Notice: Copyright (c) 2026 Selçuk Topal and Batuhan Bayraktar

- Code is released under the [PolyForm Noncommercial License 1.0.0](LICENSE).
- Text, figures, tables and data artifacts (benchmark outputs, H-Neuron artifacts) are released under [CC BY-NC 4.0](LICENSE-CC-BY-NC-4.0).

## Contact

Selçuk Topal (corresponding author), selcuktopal19832@gmail.com
Batuhan Bayraktar, batubayraktar26@gmail.com
