# Benchmark Results: output/baseline_dpo/merged_model

**Timestamp:** 2026-03-21T01:40:22.148950

**H-Neurons:** output/h_neurons_baseline/h_neurons.json

| Benchmark | Score |
|-----------|-------|
| TruthfulQA Truthful | 34.6% |
| TruthfulQA Informative | 70.1% |
| HaluEval Overall | 53.0% |
| DMind MCQ | 36.0% |
| DMind Subjective | 70.1% |

## HaluEval Breakdown

| Subset | H.Detect | R.Accuracy | Overall |
|--------|----------|------------|---------|
| qa | 468/1000 | 771/1000 | 62.0% |
| dialogue | 930/1000 | 10/1000 | 47.0% |
| summarization | 43/1000 | 958/1000 | 50.0% |

## DMind MCQ Breakdown

| Dataset | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| Blockchain_Fundamentals | 121 | 341 | 35.5% |
| Security | 162 | 372 | 43.5% |
| DAO | 117 | 372 | 31.5% |
| SmartContract | 109 | 340 | 32.1% |
| Defi | 139 | 340 | 40.9% |
| MEME | 70 | 349 | 20.1% |
| Infra | 145 | 333 | 43.5% |
| Tokenomist | 116 | 355 | 32.7% |
| NFT | 155 | 352 | 44.0% |

## DMind Subjective Breakdown

| Dataset | Score | Max | Accuracy |
|---------|-------|-----|----------|
| Blockchain_Fundamentals | 306.4 | 484 | 63.3% |
| DAO | 381.4 | 514 | 74.2% |
| Defi | 408.5 | 519 | 78.7% |
| Infra | 346.7 | 500 | 69.3% |
| MEME | 376.4 | 412 | 91.4% |
| NFT | 338.2 | 412 | 82.1% |
| Tokenomist | 380.1 | 410 | 92.7% |
| Security | 333.3 | 450 | 74.1% |
| SmartContract | 85.5 | 516 | 16.6% |
