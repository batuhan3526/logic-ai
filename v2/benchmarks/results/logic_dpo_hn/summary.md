# Benchmark Results: output/logic_dpo/merged_model

**Timestamp:** 2026-03-26T18:31:24.525017

**H-Neurons:** output/h_neurons_logic/h_neurons.json

| Benchmark | Score |
|-----------|-------|
| TruthfulQA Truthful | 42.7% |
| TruthfulQA Informative | 89.6% |
| HaluEval Overall | 48.3% |
| DMind MCQ | 34.4% |
| DMind Subjective | 77.3% |
| FalseQA Correction Rate | 0.0% |
| FalseQA Over-Compliance | 0.0% |
| FaithEval Faithfulness | 0.0% |
| Sycophancy Flip Rate | 8.5% |
| Sycophancy Stand Rate | 91.5% |

## HaluEval Breakdown

| Subset | H.Detect | R.Accuracy | Overall |
|--------|----------|------------|---------|
| qa | 14/1000 | 985/1000 | 50.0% |
| dialogue | 595/1000 | 303/1000 | 44.9% |
| summarization | 0/1000 | 1000/1000 | 50.0% |

## DMind MCQ Breakdown

| Dataset | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| Blockchain_Fundamentals | 115 | 341 | 33.7% |
| Security | 162 | 372 | 43.5% |
| DAO | 115 | 372 | 30.9% |
| SmartContract | 102 | 340 | 30.0% |
| Defi | 140 | 340 | 41.2% |
| MEME | 70 | 349 | 20.1% |
| Infra | 136 | 333 | 40.8% |
| Tokenomist | 120 | 355 | 33.8% |
| NFT | 126 | 352 | 35.8% |

## DMind Subjective Breakdown

| Dataset | Score | Max | Accuracy |
|---------|-------|-----|----------|
| Blockchain_Fundamentals | 364.9 | 484 | 75.4% |
| DAO | 418.7 | 514 | 81.5% |
| Defi | 442.3 | 519 | 85.2% |
| Infra | 410.0 | 500 | 82.0% |
| MEME | 395.9 | 412 | 96.1% |
| NFT | 370.5 | 412 | 89.9% |
| Tokenomist | 390.2 | 410 | 95.2% |
| Security | 370.7 | 450 | 82.4% |
| SmartContract | 94.5 | 516 | 18.3% |

## Sycophancy Details

- Total questions: 175
- Initially correct: 141 (80.6%)
- Flipped (sycophantic): 12
- Stood ground: 129
- Flip rate: 8.5%
- Stand rate: 91.5%
