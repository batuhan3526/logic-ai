# Benchmark Results: output/logic_dpo/merged_model

**Timestamp:** 2026-03-20T14:18:46.385643

| Benchmark | Score |
|-----------|-------|
| TruthfulQA Truthful | 33.7% |
| TruthfulQA Informative | 70.3% |
| HaluEval Overall | 52.8% |
| DMind MCQ | 35.3% |
| DMind Subjective | 69.3% |

## HaluEval Breakdown

| Subset | H.Detect | R.Accuracy | Overall |
|--------|----------|------------|---------|
| qa | 663/1000 | 433/1000 | 54.8% |
| dialogue | 1000/1000 | 0/1000 | 50.0% |
| summarization | 203/1000 | 870/1000 | 53.6% |

## DMind MCQ Breakdown

| Dataset | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| Blockchain_Fundamentals | 121 | 341 | 35.5% |
| Security | 163 | 372 | 43.8% |
| DAO | 112 | 372 | 30.1% |
| SmartContract | 108 | 340 | 31.8% |
| Defi | 138 | 340 | 40.6% |
| MEME | 61 | 349 | 17.5% |
| Infra | 137 | 333 | 41.1% |
| Tokenomist | 120 | 355 | 33.8% |
| NFT | 152 | 352 | 43.2% |

## DMind Subjective Breakdown

| Dataset | Score | Max | Accuracy |
|---------|-------|-----|----------|
| Blockchain_Fundamentals | 322.7 | 484 | 66.7% |
| DAO | 376.8 | 514 | 73.3% |
| Defi | 404.6 | 519 | 78.0% |
| Infra | 350.8 | 500 | 70.2% |
| MEME | 365.5 | 412 | 88.7% |
| NFT | 333.5 | 412 | 80.9% |
| Tokenomist | 363.0 | 410 | 88.5% |
| Security | 318.8 | 450 | 70.8% |
| SmartContract | 87.5 | 516 | 17.0% |
