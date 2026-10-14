# Benchmark Results: output/baseline_dpo/merged_model

**Timestamp:** 2026-03-26T17:40:13.372045

**H-Neurons:** output/h_neurons_baseline/h_neurons.json

| Benchmark | Score |
|-----------|-------|
| TruthfulQA Truthful | 41.6% |
| TruthfulQA Informative | 82.6% |
| HaluEval Overall | 49.7% |
| DMind MCQ | 32.8% |
| DMind Subjective | 76.2% |
| FalseQA Correction Rate | 0.0% |
| FalseQA Over-Compliance | 0.0% |
| FaithEval Faithfulness | 0.0% |
| Sycophancy Flip Rate | 13.4% |
| Sycophancy Stand Rate | 86.6% |

## HaluEval Breakdown

| Subset | H.Detect | R.Accuracy | Overall |
|--------|----------|------------|---------|
| qa | 397/1000 | 627/1000 | 51.2% |
| dialogue | 925/1000 | 20/1000 | 47.2% |
| summarization | 33/1000 | 982/1000 | 50.7% |

## DMind MCQ Breakdown

| Dataset | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| Blockchain_Fundamentals | 109 | 341 | 32.0% |
| Security | 148 | 372 | 39.8% |
| DAO | 110 | 372 | 29.6% |
| SmartContract | 99 | 340 | 29.1% |
| Defi | 120 | 340 | 35.3% |
| MEME | 64 | 349 | 18.3% |
| Infra | 132 | 333 | 39.6% |
| Tokenomist | 99 | 355 | 27.9% |
| NFT | 152 | 352 | 43.2% |

## DMind Subjective Breakdown

| Dataset | Score | Max | Accuracy |
|---------|-------|-----|----------|
| Blockchain_Fundamentals | 353.3 | 484 | 73.0% |
| DAO | 405.8 | 514 | 78.9% |
| Defi | 446.7 | 519 | 86.1% |
| Infra | 400.9 | 500 | 80.2% |
| MEME | 394.5 | 412 | 95.8% |
| NFT | 358.5 | 412 | 87.0% |
| Tokenomist | 390.1 | 410 | 95.1% |
| Security | 376.8 | 450 | 83.7% |
| SmartContract | 88.0 | 516 | 17.1% |

## Sycophancy Details

- Total questions: 175
- Initially correct: 134 (76.6%)
- Flipped (sycophantic): 18
- Stood ground: 116
- Flip rate: 13.4%
- Stand rate: 86.6%
