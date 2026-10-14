# Benchmark Results: /home/whiteowl/last_shot/v2/output/baseline_dpo/merged_model

**Timestamp:** 2026-06-01T21:01:54.182359

**H-Neurons:** caa_ab:mul=-3.0:layers=8-22

| Benchmark | Score |
|-----------|-------|
| TruthfulQA Truthful | 24.5% |
| TruthfulQA Informative | 96.3% |
| HaluEval Overall | 50.0% |
| DMind MCQ | 20.9% |
| DMind Subjective | 9.9% |
| FalseQA Correction Rate | 0.0% |
| FalseQA Over-Compliance | 0.0% |
| FaithEval Faithfulness | 0.0% |
| Sycophancy Flip Rate | 0.0% |
| Sycophancy Stand Rate | 100.0% |

## HaluEval Breakdown

| Subset | H.Detect | R.Accuracy | Overall |
|--------|----------|------------|---------|
| qa | 0/1000 | 1000/1000 | 50.0% |
| dialogue | 0/1000 | 1000/1000 | 50.0% |
| summarization | 0/1000 | 1000/1000 | 50.0% |

## DMind MCQ Breakdown

| Dataset | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| Blockchain_Fundamentals | 63 | 341 | 18.5% |
| Security | 96 | 372 | 25.8% |
| DAO | 81 | 372 | 21.8% |
| SmartContract | 57 | 340 | 16.8% |
| Defi | 55 | 340 | 16.2% |
| MEME | 65 | 349 | 18.6% |
| Infra | 74 | 333 | 22.2% |
| Tokenomist | 52 | 355 | 14.6% |
| NFT | 115 | 352 | 32.7% |

## DMind Subjective Breakdown

| Dataset | Score | Max | Accuracy |
|---------|-------|-----|----------|
| Blockchain_Fundamentals | 48.4 | 484 | 10.0% |
| DAO | 49.3 | 514 | 9.6% |
| Defi | 62.8 | 519 | 12.1% |
| Infra | 36.7 | 500 | 7.3% |
| MEME | 39.3 | 412 | 9.5% |
| NFT | 59.9 | 412 | 14.5% |
| Tokenomist | 51.9 | 410 | 12.7% |
| Security | 40.3 | 450 | 9.0% |
| SmartContract | 27.0 | 516 | 5.2% |

## Sycophancy Details

- Total questions: 175
- Initially correct: 0 (0.0%)
- Flipped (sycophantic): 0
- Stood ground: 0
- Flip rate: 0.0%
- Stand rate: 100.0%
