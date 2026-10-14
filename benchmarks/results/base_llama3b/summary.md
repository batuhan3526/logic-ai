# Benchmark Results: meta-llama/Llama-3.2-3B-Instruct

**Timestamp:** 2026-03-19T18:29:06.204028

| Benchmark | Score |
|-----------|-------|
| TruthfulQA Truthful | 39.5% |
| TruthfulQA Informative | 98.0% |
| HaluEval Overall | 46.1% |
| DMind MCQ | 33.9% |
| DMind Subjective | 76.3% |

## HaluEval Breakdown

| Subset | H.Detect | R.Accuracy | Overall |
|--------|----------|------------|---------|
| qa | 203/1000 | 594/1000 | 39.9% |
| dialogue | 981/1000 | 1/1000 | 49.1% |
| summarization | 54/1000 | 935/1000 | 49.5% |

## DMind MCQ Breakdown

| Dataset | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| Blockchain_Fundamentals | 119 | 341 | 34.9% |
| Security | 150 | 372 | 40.3% |
| DAO | 123 | 372 | 33.1% |
| SmartContract | 100 | 340 | 29.4% |
| Defi | 130 | 340 | 38.2% |
| MEME | 74 | 349 | 21.2% |
| Infra | 136 | 333 | 40.8% |
| Tokenomist | 103 | 355 | 29.0% |
| NFT | 133 | 352 | 37.8% |

## DMind Subjective Breakdown

| Dataset | Score | Max | Accuracy |
|---------|-------|-----|----------|
| Blockchain_Fundamentals | 354.3 | 484 | 73.2% |
| DAO | 402.3 | 514 | 78.3% |
| Defi | 449.0 | 519 | 86.5% |
| Infra | 379.1 | 500 | 75.8% |
| MEME | 394.6 | 412 | 95.8% |
| NFT | 378.6 | 412 | 91.9% |
| Tokenomist | 395.0 | 410 | 96.3% |
| Security | 373.4 | 450 | 83.0% |
| SmartContract | 93.0 | 516 | 18.0% |
