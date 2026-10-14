# Benchmark Results: output/logic_dpo/merged_model

**Timestamp:** 2026-03-21T02:26:16.824709

**H-Neurons:** output/h_neurons_logic/h_neurons.json

| Benchmark | Score |
|-----------|-------|
| TruthfulQA Truthful | 32.7% |
| TruthfulQA Informative | 68.3% |
| HaluEval Overall | 51.9% |
| DMind MCQ | 34.9% |
| DMind Subjective | 68.8% |

## HaluEval Breakdown

| Subset | H.Detect | R.Accuracy | Overall |
|--------|----------|------------|---------|
| qa | 627/1000 | 431/1000 | 52.9% |
| dialogue | 1000/1000 | 1/1000 | 50.0% |
| summarization | 180/1000 | 875/1000 | 52.8% |

## DMind MCQ Breakdown

| Dataset | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| Blockchain_Fundamentals | 119 | 341 | 34.9% |
| Security | 159 | 372 | 42.7% |
| DAO | 111 | 372 | 29.8% |
| SmartContract | 108 | 340 | 31.8% |
| Defi | 134 | 340 | 39.4% |
| MEME | 69 | 349 | 19.8% |
| Infra | 137 | 333 | 41.1% |
| Tokenomist | 120 | 355 | 33.8% |
| NFT | 144 | 352 | 40.9% |

## DMind Subjective Breakdown

| Dataset | Score | Max | Accuracy |
|---------|-------|-----|----------|
| Blockchain_Fundamentals | 335.1 | 484 | 69.2% |
| DAO | 366.1 | 514 | 71.2% |
| Defi | 388.3 | 519 | 74.8% |
| Infra | 338.6 | 500 | 67.7% |
| MEME | 357.6 | 412 | 86.8% |
| NFT | 327.8 | 412 | 79.6% |
| Tokenomist | 376.7 | 410 | 91.9% |
| Security | 327.3 | 450 | 72.7% |
| SmartContract | 83.5 | 516 | 16.2% |
