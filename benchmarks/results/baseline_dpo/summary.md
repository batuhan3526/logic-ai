# Benchmark Results: output/baseline_dpo/merged_model

**Timestamp:** 2026-03-20T13:35:41.998475

| Benchmark | Score |
|-----------|-------|
| TruthfulQA Truthful | 34.8% |
| TruthfulQA Informative | 70.3% |
| HaluEval Overall | 53.0% |
| DMind MCQ | 35.5% |
| DMind Subjective | 69.3% |

## HaluEval Breakdown

| Subset | H.Detect | R.Accuracy | Overall |
|--------|----------|------------|---------|
| qa | 452/1000 | 755/1000 | 60.4% |
| dialogue | 954/1000 | 13/1000 | 48.4% |
| summarization | 66/1000 | 940/1000 | 50.3% |

## DMind MCQ Breakdown

| Dataset | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| Blockchain_Fundamentals | 122 | 341 | 35.8% |
| Security | 163 | 372 | 43.8% |
| DAO | 116 | 372 | 31.2% |
| SmartContract | 104 | 340 | 30.6% |
| Defi | 142 | 340 | 41.8% |
| MEME | 68 | 349 | 19.5% |
| Infra | 142 | 333 | 42.6% |
| Tokenomist | 107 | 355 | 30.1% |
| NFT | 157 | 352 | 44.6% |

## DMind Subjective Breakdown

| Dataset | Score | Max | Accuracy |
|---------|-------|-----|----------|
| Blockchain_Fundamentals | 334.3 | 484 | 69.1% |
| DAO | 378.9 | 514 | 73.7% |
| Defi | 410.4 | 519 | 79.1% |
| Infra | 350.0 | 500 | 70.0% |
| MEME | 360.2 | 412 | 87.4% |
| NFT | 329.2 | 412 | 79.9% |
| Tokenomist | 366.3 | 410 | 89.3% |
| Security | 309.4 | 450 | 68.8% |
| SmartContract | 85.5 | 516 | 16.6% |
