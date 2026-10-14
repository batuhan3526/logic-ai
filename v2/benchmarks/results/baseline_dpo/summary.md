# Benchmark Results: output/baseline_dpo/merged_model

**Timestamp:** 2026-03-22T20:33:05.594812

| Benchmark | Score |
|-----------|-------|
| TruthfulQA Truthful | 40.8% |
| TruthfulQA Informative | 82.9% |
| HaluEval Overall | 49.9% |
| DMind MCQ | 32.9% |
| DMind Subjective | 75.8% |
| FalseQA Correction Rate | 0.0% |
| FalseQA Over-Compliance | 0.0% |
| FaithEval Faithfulness | 0.0% |
| Sycophancy Flip Rate | 14.6% |
| Sycophancy Stand Rate | 85.4% |

## HaluEval Breakdown

| Subset | H.Detect | R.Accuracy | Overall |
|--------|----------|------------|---------|
| qa | 408/1000 | 615/1000 | 51.1% |
| dialogue | 936/1000 | 16/1000 | 47.6% |
| summarization | 38/1000 | 981/1000 | 50.9% |

## DMind MCQ Breakdown

| Dataset | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| Blockchain_Fundamentals | 112 | 341 | 32.8% |
| Security | 150 | 372 | 40.3% |
| DAO | 110 | 372 | 29.6% |
| SmartContract | 97 | 340 | 28.5% |
| Defi | 125 | 340 | 36.8% |
| MEME | 63 | 349 | 18.1% |
| Infra | 132 | 333 | 39.6% |
| Tokenomist | 102 | 355 | 28.7% |
| NFT | 147 | 352 | 41.8% |

## DMind Subjective Breakdown

| Dataset | Score | Max | Accuracy |
|---------|-------|-----|----------|
| Blockchain_Fundamentals | 347.5 | 484 | 71.8% |
| DAO | 419.4 | 514 | 81.6% |
| Defi | 436.6 | 519 | 84.1% |
| Infra | 387.3 | 500 | 77.5% |
| MEME | 389.7 | 412 | 94.6% |
| NFT | 374.3 | 412 | 90.8% |
| Tokenomist | 394.4 | 410 | 96.2% |
| Security | 355.1 | 450 | 78.9% |
| SmartContract | 93.0 | 516 | 18.0% |

## Sycophancy Details

- Total questions: 175
- Initially correct: 137 (78.3%)
- Flipped (sycophantic): 20
- Stood ground: 117
- Flip rate: 14.6%
- Stand rate: 85.4%
