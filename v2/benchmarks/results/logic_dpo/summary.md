# Benchmark Results: output/logic_dpo/merged_model

**Timestamp:** 2026-03-22T21:18:57.604496

| Benchmark | Score |
|-----------|-------|
| TruthfulQA Truthful | 40.8% |
| TruthfulQA Informative | 89.2% |
| HaluEval Overall | 48.9% |
| DMind MCQ | 34.7% |
| DMind Subjective | 76.3% |
| FalseQA Correction Rate | 0.0% |
| FalseQA Over-Compliance | 0.0% |
| FaithEval Faithfulness | 0.0% |
| Sycophancy Flip Rate | 7.0% |
| Sycophancy Stand Rate | 93.0% |

## HaluEval Breakdown

| Subset | H.Detect | R.Accuracy | Overall |
|--------|----------|------------|---------|
| qa | 19/1000 | 986/1000 | 50.2% |
| dialogue | 619/1000 | 312/1000 | 46.6% |
| summarization | 1/1000 | 999/1000 | 50.0% |

## DMind MCQ Breakdown

| Dataset | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| Blockchain_Fundamentals | 120 | 341 | 35.2% |
| Security | 164 | 372 | 44.1% |
| DAO | 115 | 372 | 30.9% |
| SmartContract | 101 | 340 | 29.7% |
| Defi | 132 | 340 | 38.8% |
| MEME | 67 | 349 | 19.2% |
| Infra | 146 | 333 | 43.8% |
| Tokenomist | 118 | 355 | 33.2% |
| NFT | 130 | 352 | 36.9% |

## DMind Subjective Breakdown

| Dataset | Score | Max | Accuracy |
|---------|-------|-----|----------|
| Blockchain_Fundamentals | 365.0 | 484 | 75.4% |
| DAO | 423.6 | 514 | 82.4% |
| Defi | 432.4 | 519 | 83.3% |
| Infra | 369.3 | 500 | 73.9% |
| MEME | 393.2 | 412 | 95.4% |
| NFT | 375.0 | 412 | 91.0% |
| Tokenomist | 393.2 | 410 | 95.9% |
| Security | 372.6 | 450 | 82.8% |
| SmartContract | 93.5 | 516 | 18.1% |

## Sycophancy Details

- Total questions: 175
- Initially correct: 143 (81.7%)
- Flipped (sycophantic): 10
- Stood ground: 133
- Flip rate: 7.0%
- Stand rate: 93.0%
