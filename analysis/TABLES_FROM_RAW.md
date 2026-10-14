# Tablo 4–7 ve ek tablolar — ham dosyalardan üretildi

**Tarih:** 2026-10-03. Üretici: `analysis/gen_tables.py`. Kaynak: `v2/benchmarks/results/*/results.json`, `v2/output/h_neurons_*/`, `v1/benchmarks/results/base_llama3b/results.json`, `finance_qwen3/benchmarks/`.

Wilson %95 aralıkları; iki-oranlı iki-yönlü z-testi. Base modelin TruthfulQA/HaluEval/DMind değerleri V1 koşusundan (2026-03-19), sycophancy V2 koşusundan (2026-03-21). DMind-subjective puan/maks-puan oranı olduğu için Wilson aralığı verilmedi.


## Tablo 4 — Ana sonuçlar (Wilson %95)

| Configuration | TruthfulQA truthful (n=817) | TruthfulQA informative (n=817) | HaluEval (n=6000) | DMind MCQ (n=3154) | DMind subj. (pts/max) | Sycophancy flip (n) |
|---|---|---|---|---|---|---|
| Base (Llama-3.2-3B-Instruct) | 39.5 [36.2, 42.9] | 98.0 [96.8, 98.8] | 46.1 [44.9, 47.4] | 33.9 [32.2, 35.5] | 76.3 (3219/4217) | 4.3 [2.0, 9.0] (n=141) |
| Domain arm | 40.8 [37.4, 44.2] | 82.9 [80.1, 85.3] | 49.9 [48.6, 51.2] | 32.9 [31.3, 34.6] | 75.8 (3197/4217) | 14.6 [9.7, 21.5] (n=137) |
| Domain arm + HN | 41.6 [38.3, 45.0] | 82.6 [79.9, 85.1] | 49.7 [48.5, 51.0] | 32.8 [31.1, 34.4] | 76.2 (3215/4217) | 13.4 [8.7, 20.2] (n=134) |
| Logic arm | 40.8 [37.4, 44.2] | 89.2 [86.9, 91.2] | 48.9 [47.7, 50.2] | 34.7 [33.0, 36.3] | 76.3 (3218/4217) | 7.0 [3.8, 12.4] (n=143) |
| Logic arm + HN | 42.7 [39.4, 46.1] | 89.6 [87.3, 91.5] | 48.3 [47.0, 49.5] | 34.4 [32.8, 36.1] | 77.3 (3258/4217) | 8.5 [4.9, 14.3] (n=141) |

Ham sayılar (doğru/toplam):

| Configuration | truthful | informative | HaluEval | DMind MCQ | flips/initially-correct |
|---|---|---|---|---|---|
| Base (Llama-3.2-3B-Instruct) | 323/817 | 801/817 | 2768/6000 | 1068/3154 | 6/141 |
| Domain arm | 333/817 | 677/817 | 2994/6000 | 1038/3154 | 20/137 |
| Domain arm + HN | 340/817 | 675/817 | 2984/6000 | 1033/3154 | 18/134 |
| Logic arm | 333/817 | 729/817 | 2936/6000 | 1093/3154 | 10/143 |
| Logic arm + HN | 349/817 | 732/817 | 2897/6000 | 1086/3154 | 12/141 |

### LaTeX (main_results.tex)
```latex
\begin{table}[t]\centering\small
\caption{Main results with Wilson 95\% intervals. Base TruthfulQA/HaluEval/DMind values come from the V1 evaluation run (2026-03-19); all other cells from the V2 run with identical prompts, decoding and scorers. DMind-subjective is a points ratio and carries no interval.}
\label{tab:main}
\begin{tabular}{lcccccc}\toprule
Configuration & \makecell{TruthfulQA\\truthful} & \makecell{TruthfulQA\\informative} & HaluEval & \makecell{DMind\\MCQ} & \makecell{DMind\\subj.} & \makecell{Sycophancy\\flip} \\ \midrule
Base (Llama-3.2-3B-Instruct) & 39.5 {\scriptsize[36.2, 42.9]} & 98.0 {\scriptsize[96.8, 98.8]} & 46.1 {\scriptsize[44.9, 47.4]} & 33.9 {\scriptsize[32.2, 35.5]} & 76.3 & 4.3 {\scriptsize[2.0, 9.0]} (n=141) \\
Domain arm & 40.8 {\scriptsize[37.4, 44.2]} & 82.9 {\scriptsize[80.1, 85.3]} & 49.9 {\scriptsize[48.6, 51.2]} & 32.9 {\scriptsize[31.3, 34.6]} & 75.8 & 14.6 {\scriptsize[9.7, 21.5]} (n=137) \\
Domain arm + HN & 41.6 {\scriptsize[38.3, 45.0]} & 82.6 {\scriptsize[79.9, 85.1]} & 49.7 {\scriptsize[48.5, 51.0]} & 32.8 {\scriptsize[31.1, 34.4]} & 76.2 & 13.4 {\scriptsize[8.7, 20.2]} (n=134) \\
Logic arm & 40.8 {\scriptsize[37.4, 44.2]} & 89.2 {\scriptsize[86.9, 91.2]} & 48.9 {\scriptsize[47.7, 50.2]} & 34.7 {\scriptsize[33.0, 36.3]} & 76.3 & 7.0 {\scriptsize[3.8, 12.4]} (n=143) \\
Logic arm + HN & 42.7 {\scriptsize[39.4, 46.1]} & 89.6 {\scriptsize[87.3, 91.5]} & 48.3 {\scriptsize[47.0, 49.5]} & 34.4 {\scriptsize[32.8, 36.1]} & 77.3 & 8.5 {\scriptsize[4.9, 14.3]} (n=141) \\
\bottomrule\end{tabular}\end{table}
```

## Tablo 5 — İkili iki-oranlı z-testleri (B − A, pp)

| Comparison (A → B) | Metric | A % | B % | Δ pp | z | p |
|---|---|---|---|---|---|---|
| RQ1: SFT mixture | TruthfulQA truthful | 40.8 | 40.8 | +0.0 | +0.00 | 1.00 |
| RQ1: SFT mixture | TruthfulQA informative | 82.9 | 89.2 | +6.4 | +3.71 | <0.001 |
| RQ1: SFT mixture | HaluEval | 49.9 | 48.9 | -1.0 | -1.06 | 0.29 |
| RQ1: SFT mixture | DMind MCQ | 32.9 | 34.7 | +1.7 | +1.46 | 0.14 |
| RQ1: SFT mixture | Sycophancy flip | 14.6 | 7.0 | -7.6 | -2.06 | 0.04 |
| RQ2: suppression, domain arm | TruthfulQA truthful | 40.8 | 41.6 | +0.9 | +0.35 | 0.72 |
| RQ2: suppression, domain arm | TruthfulQA informative | 82.9 | 82.6 | -0.2 | -0.13 | 0.90 |
| RQ2: suppression, domain arm | HaluEval | 49.9 | 49.7 | -0.2 | -0.18 | 0.86 |
| RQ2: suppression, domain arm | DMind MCQ | 32.9 | 32.8 | -0.2 | -0.13 | 0.89 |
| RQ2: suppression, domain arm | Sycophancy flip | 14.6 | 13.4 | -1.2 | -0.28 | 0.78 |
| RQ2: suppression, logic arm | TruthfulQA truthful | 40.8 | 42.7 | +2.0 | +0.80 | 0.42 |
| RQ2: suppression, logic arm | TruthfulQA informative | 89.2 | 89.6 | +0.4 | +0.24 | 0.81 |
| RQ2: suppression, logic arm | HaluEval | 48.9 | 48.3 | -0.7 | -0.71 | 0.48 |
| RQ2: suppression, logic arm | DMind MCQ | 34.7 | 34.4 | -0.2 | -0.19 | 0.85 |
| RQ2: suppression, logic arm | Sycophancy flip | 7.0 | 8.5 | +1.5 | +0.48 | 0.63 |
| adaptation, domain arm | TruthfulQA truthful | 39.5 | 40.8 | +1.2 | +0.50 | 0.61 |
| adaptation, domain arm | TruthfulQA informative | 98.0 | 82.9 | -15.2 | -10.44 | <0.001 |
| adaptation, domain arm | HaluEval | 46.1 | 49.9 | +3.8 | +4.13 | <0.001 |
| adaptation, domain arm | DMind MCQ | 33.9 | 32.9 | -1.0 | -0.80 | 0.42 |
| adaptation, domain arm | Sycophancy flip | 4.3 | 14.6 | +10.3 | +2.96 | 0.00 |
| adaptation, logic arm | TruthfulQA truthful | 39.5 | 40.8 | +1.2 | +0.50 | 0.61 |
| adaptation, logic arm | TruthfulQA informative | 98.0 | 89.2 | -8.8 | -7.30 | <0.001 |
| adaptation, logic arm | HaluEval | 46.1 | 48.9 | +2.8 | +3.07 | 0.00 |
| adaptation, logic arm | DMind MCQ | 33.9 | 34.7 | +0.8 | +0.66 | 0.51 |
| adaptation, logic arm | Sycophancy flip | 4.3 | 7.0 | +2.7 | +1.00 | 0.32 |

Not: base-vs-adapted TruthfulQA satırları, TruthfulQA'nın her iki kolun eğitim verisinde bulunması nedeniyle yalnızca tamlık için verilmiştir (bkz. MAKALE_EKSIKLER.md K1).

### LaTeX (significance.tex)
```latex
\begin{table}[t]\centering\small
\caption{Pairwise two-sided two-proportion $z$-tests (unpaired). $\Delta$ is B minus A in percentage points.}\label{tab:sig}
\begin{tabular}{llrr}\toprule Comparison (A$\to$B) & Metric & $\Delta$ (pp) & $p$ \\ \midrule
RQ1: SFT mixture & TruthfulQA truthful & +0.0 & 1.00 \\
RQ1: SFT mixture & TruthfulQA informative & +6.4 & <0.001 \\
RQ1: SFT mixture & HaluEval & -1.0 & 0.29 \\
RQ1: SFT mixture & DMind MCQ & +1.7 & 0.14 \\
RQ1: SFT mixture & Sycophancy flip & -7.6 & 0.04 \\
RQ2: suppression, domain arm & TruthfulQA truthful & +0.9 & 0.72 \\
RQ2: suppression, domain arm & TruthfulQA informative & -0.2 & 0.90 \\
RQ2: suppression, domain arm & HaluEval & -0.2 & 0.86 \\
RQ2: suppression, domain arm & DMind MCQ & -0.2 & 0.89 \\
RQ2: suppression, domain arm & Sycophancy flip & -1.2 & 0.78 \\
RQ2: suppression, logic arm & TruthfulQA truthful & +2.0 & 0.42 \\
RQ2: suppression, logic arm & TruthfulQA informative & +0.4 & 0.81 \\
RQ2: suppression, logic arm & HaluEval & -0.7 & 0.48 \\
RQ2: suppression, logic arm & DMind MCQ & -0.2 & 0.85 \\
RQ2: suppression, logic arm & Sycophancy flip & +1.5 & 0.63 \\
adaptation, domain arm & TruthfulQA truthful & +1.2 & 0.61 \\
adaptation, domain arm & TruthfulQA informative & -15.2 & <0.001 \\
adaptation, domain arm & HaluEval & +3.8 & <0.001 \\
adaptation, domain arm & DMind MCQ & -1.0 & 0.42 \\
adaptation, domain arm & Sycophancy flip & +10.3 & 0.00 \\
adaptation, logic arm & TruthfulQA truthful & +1.2 & 0.61 \\
adaptation, logic arm & TruthfulQA informative & -8.8 & <0.001 \\
adaptation, logic arm & HaluEval & +2.8 & 0.00 \\
adaptation, logic arm & DMind MCQ & +0.8 & 0.51 \\
adaptation, logic arm & Sycophancy flip & +2.7 & 0.32 \\
\bottomrule\end{tabular}\end{table}
```

## Tablo 6 — H-Neuron keşif istatistikleri

Ortak nöron: 17 (68% of domain set, 59% of logic set). En yüksek ağırlıklı iki nöron her iki kolda L1:5002 ve L4:1751.

| Statistic | Domain arm | Logic arm |
|---|---|---|
| Discovery pool | 4,000 TriviaQA + 1,000 synthetic finance questions (`generate_finance_qa`), 10 samples each at T=1 | same |
| Consistency filter outcome | 808 consistently-correct / 835 consistently-wrong / 3,357 discarded (of 5,000) | 964 / 1,147 / 2,889 |
| Within-question agreement (recomputed from stage_5_consistency.json) | mean share of most common answer 35.1% (correct) vs 12.5% (wrong), ratio 2.8; all-10-agree 1.1% vs 0.1% | 40.7% vs 13.0%, ratio 3.1; all-10-agree 2.2% vs 0.0% |
| Domain-gate accuracy (200 held-out domain Qs) | 8/200 = 4.0% (window 15–35%: fail) | 9/200 = 4.5% (window 15–35%: fail) |
| Probe | L1 logistic regression, C=0.5, CETT features on down_proj inputs | same |
| Probe 5-fold CV accuracy | 94.65% | 95.07% |
| Selected neurons (of 229,376 FFN units) | 25 | 29 |
| Positive / negative weights | 25 / 26 | 29 / 22 |
| Layer distribution (early 0–9 / middle 10–19 / late 20–27) | early 10 / middle 7 / late 8 | early 15 / middle 9 / late 5 |
| Selected α (early / middle / late) | 1.15 / 0.85 / 0.45 | 1.0 / 1.0 / 0.0 |
| Held-out accuracy, no suppression → selected | 71.5% → 72.4% (+0.9 pp; n=1000, binomial SE ≈ 1.4 pp) | 74.9% → 75.3% (+0.4 pp; n=1000, binomial SE ≈ 1.4 pp) |
| α-sweep | coarse 64 + fine 92 = 156; range 70.5–72.4%; SD 0.32 pp (fine only 0.31 pp); 127/156 within ±0.5 pp of median | coarse 64 + fine 66 = 130; range 67.3–75.3%; SD 2.18 pp (fine only 0.38 pp); 62/130 within ±0.5 pp of median |

### LaTeX (hneurons.tex)
```latex
\begin{table}[t]\centering\small\caption{H-Neuron discovery statistics for the two arms.}\label{tab:hn}
\begin{tabular}{p{4.2cm}p{4.6cm}p{4.6cm}}\toprule Statistic & Domain arm & Logic arm \\ \midrule
Discovery pool & \multicolumn{2}{p{9.2cm}}{4,000 TriviaQA + 1,000 synthetic finance questions (generate_finance_qa), 10 samples each at T=1} \\
Consistency filter outcome & 808 consistently-correct / 835 consistently-wrong / 3,357 discarded (of 5,000) & 964 / 1,147 / 2,889 \\
Within-question agreement (recomputed from stage_5_consistency.json) & mean share of most common answer 35.1\% (correct) vs 12.5\% (wrong), ratio 2.8; all-10-agree 1.1\% vs 0.1\% & 40.7\% vs 13.0\%, ratio 3.1; all-10-agree 2.2\% vs 0.0\% \\
Domain-gate accuracy (200 held-out domain Qs) & 8/200 = 4.0\% (window 15--35\%: fail) & 9/200 = 4.5\% (window 15--35\%: fail) \\
Probe & \multicolumn{2}{p{9.2cm}}{L1 logistic regression, C=0.5, CETT features on down_proj inputs} \\
Probe 5-fold CV accuracy & 94.65\% & 95.07\% \\
Selected neurons (of 229,376 FFN units) & 25 & 29 \\
Positive / negative weights & 25 / 26 & 29 / 22 \\
Layer distribution (early 0--9 / middle 10--19 / late 20--27) & early 10 / middle 7 / late 8 & early 15 / middle 9 / late 5 \\
Selected $\alpha$ (early / middle / late) & 1.15 / 0.85 / 0.45 & 1.0 / 1.0 / 0.0 \\
Held-out accuracy, no suppression $\rightarrow$ selected & 71.5\% $\rightarrow$ 72.4\% (+0.9 pp; n=1000, binomial SE $\approx$ 1.4 pp) & 74.9\% $\rightarrow$ 75.3\% (+0.4 pp; n=1000, binomial SE $\approx$ 1.4 pp) \\
$\alpha$-sweep & coarse 64 + fine 92 = 156; range 70.5--72.4\%; SD 0.32 pp (fine only 0.31 pp); 127/156 within $\pm$0.5 pp of median & coarse 64 + fine 66 = 130; range 67.3--75.3\%; SD 2.18 pp (fine only 0.38 pp); 62/130 within $\pm$0.5 pp of median \\
Shared neurons & \multicolumn{2}{l}{17 (top-2 in both: L1:5002, L4:1751)} \\
\bottomrule\end{tabular}\end{table}
```

## Tablo 7 — Finans replikasyonu (Qwen3-1.7B)

| Configuration | TruthfulQA truthful (n=817) | TruthfulQA informative (n=817) | DMind MCQ % | DMind subj. % | HaluEval % |
|---|---|---|---|---|---|
| Base | 39.9 [36.6, 43.3] | 99.4 [98.6, 99.7] | 35.4 | 76.3 | 49.9 |
| Plain DAPT+SFT (classical) | 34.0 [30.9, 37.3] | 96.0 [94.4, 97.1] | 35.3 | 75.5 | 50.3 |
| Logic SFT v3 | 31.6 [28.5, 34.8] | 83.5 [80.8, 85.9] | 22.4 | 73.9 | 50.0 |
| Logic SFT v4 (heavy annotation) | 28.4 [25.4, 31.6] | 70.0 [66.8, 73.1] | 34.1 | 75.7 | 49.1 |
| Logic SFT v5 (light dose) | 30.8 [27.8, 34.1] | 89.8 [87.6, 91.7] | 31.2 | 73.7 | 50.0 |
| Logic DPO v5 | 32.6 [29.4, 35.8] | 88.4 [86.0, 90.4] | 31.6 | 74.9 | 50.0 |

Testler (informative): 
- Plain DAPT+SFT (classical) → Logic SFT v4 (heavy annotation): Δ -25.9 pp, p <0.001
- Logic SFT v4 (heavy annotation) → Logic SFT v5 (light dose): Δ +19.8 pp, p <0.001
- Plain DAPT+SFT (classical) → Logic SFT v5 (light dose): Δ -6.1 pp, p <0.001
- Logic SFT v5 (light dose) → Logic DPO v5: Δ -1.5 pp, p =0.341
Testler (truthful): 
- Plain DAPT+SFT (classical) → Logic SFT v4 (heavy annotation): Δ -5.6 pp, p =0.014
- Logic SFT v4 (heavy annotation) → Logic SFT v5 (light dose): Δ +2.4 pp, p =0.279

**Protokol uyarısı (tabloya dipnot olarak girmeli):** v3 MCQ reasoning prompt + 256 tok; v4/v5 direct-letter + 20 tok; v5 TruthfulQA/DMind'da ek logic-enriched user prompt; v5 SFT rank 32 / lr 3e-5 (v4 'ultra-gentle'); DA v5 = DA v4 adapter symlink. Yalnızca HaluEval v4–v5 arasında tam adil.

### LaTeX (finance_results.tex)
```latex
\begin{table}[t]\centering\small\caption{Finance replication on Qwen3-1.7B. TruthfulQA cells carry Wilson 95\% intervals; the other columns are the benchmark runner's reported percentages. Evaluation prompts were not held constant across iterations (see footnote).}\label{tab:finance}
\begin{tabular}{lccccc}\toprule Configuration & \makecell{TruthfulQA\\truthful} & \makecell{TruthfulQA\\informative} & \makecell{DMind\\MCQ} & \makecell{DMind\\subj.} & HaluEval \\ \midrule
Base & 39.9 {\scriptsize[36.6, 43.3]} & 99.4 {\scriptsize[98.6, 99.7]} & 35.4 & 76.3 & 49.9 \\
Plain DAPT+SFT (classical) & 34.0 {\scriptsize[30.9, 37.3]} & 96.0 {\scriptsize[94.4, 97.1]} & 35.3 & 75.5 & 50.3 \\
Logic SFT v3 & 31.6 {\scriptsize[28.5, 34.8]} & 83.5 {\scriptsize[80.8, 85.9]} & 22.4 & 73.9 & 50.0 \\
Logic SFT v4 (heavy annotation) & 28.4 {\scriptsize[25.4, 31.6]} & 70.0 {\scriptsize[66.8, 73.1]} & 34.1 & 75.7 & 49.1 \\
Logic SFT v5 (light dose) & 30.8 {\scriptsize[27.8, 34.1]} & 89.8 {\scriptsize[87.6, 91.7]} & 31.2 & 73.7 & 50.0 \\
Logic DPO v5 & 32.6 {\scriptsize[29.4, 35.8]} & 88.4 {\scriptsize[86.0, 90.4]} & 31.6 & 74.9 & 50.0 \\
\bottomrule\end{tabular}\end{table}
```

## Ek Tablo A — HaluEval cevap yanlılığı (makaleye girmeli)

Her alt sette 1,000 halüsinasyonlu (doğru cevap: Yes) ve 1,000 sadık (doğru cevap: No) örnek. `h_correct` = Yes dediği halüsinasyonlu örnek, `r_correct` = No dediği sadık örnek. Her sayı 1,000 üzerinden. Biri ~1,000, diğeri ~0 ise model o alt sette hep aynı cevabı veriyor; ikisi de ~500 civarındaysa rastgele.

| Configuration | QA (h / r) | Dialogue (h / r) | Summarisation (h / r) | Overall |
|---|---|---|---|---|
| Base (Llama-3.2-3B-Instruct) | 203 / 594 | 981 / 1 | 54 / 935 | 46.1% |
| Domain arm | 408 / 615 | 936 / 16 | 38 / 981 | 49.9% |
| Domain arm + HN | 397 / 627 | 925 / 20 | 33 / 982 | 49.7% |
| Logic arm | 19 / 986 | 619 / 312 | 1 / 999 | 48.9% |
| Logic arm + HN | 14 / 985 | 595 / 303 | 0 / 1000 | 48.3% |

Yorum: Dialogue ve summarisation'da tüm modeller tek cevaba kilitli (base dialogue 981/1 → hep Yes; logic summarisation 1/999 → hep No). QA'da domain kolu 408/615 (yüzde 51, şans), logic kolu 19/986 (hep No). Hiçbir konfigürasyon ayrım yapmıyor; genel doğruluk Yes/No yanlılığının aritmetiği. 'Both arms gain on HaluEval' cümlesi kaldırılmalı.


## Ek Tablo B — DMind MCQ kategori kırılımı

| Category | n | Base (Llama-3.2-3B-Instruct) | Domain arm | Domain arm + HN | Logic arm | Logic arm + HN |
|---|---|---|---|---|---|---|
| Blockchain_Fundamentals | 341 | 34.9 | 32.8 | 32.0 | 35.2 | 33.7 |
| Security | 372 | 40.3 | 40.3 | 39.8 | 44.1 | 43.5 |
| DAO | 372 | 33.1 | 29.6 | 29.6 | 30.9 | 30.9 |
| SmartContract | 340 | 29.4 | 28.5 | 29.1 | 29.7 | 30.0 |
| Defi | 340 | 38.2 | 36.8 | 35.3 | 38.8 | 41.2 |
| MEME | 349 | 21.2 | 18.1 | 18.3 | 19.2 | 20.1 |
| Infra | 333 | 40.8 | 39.6 | 39.6 | 43.8 | 40.8 |
| Tokenomist | 355 | 29.0 | 28.7 | 27.9 | 33.2 | 33.8 |
| NFT | 352 | 37.8 | 41.8 | 43.2 | 36.9 | 35.8 |

Soru-düzeyi iki-oranlı test (domain vs logic, 1,038/3,154 vs 1,093/3,154): z=+1.46, p=0.143

## Ek Tablo C — DMind subjective kategori kırılımı (puan oranı %)

| Category | Base (Llama-3.2-3B-Instruct) | Domain arm | Domain arm + HN | Logic arm | Logic arm + HN |
|---|---|---|---|---|---|
| Blockchain_Fundamentals | 73.2 | 71.8 | 73.0 | 75.4 | 75.4 |
| Security | 83.0 | 78.9 | 83.7 | 82.8 | 82.4 |
| DAO | 78.3 | 81.6 | 78.9 | 82.4 | 81.5 |
| SmartContract | 18.0 | 18.0 | 17.1 | 18.1 | 18.3 |
| Defi | 86.5 | 84.1 | 86.1 | 83.3 | 85.2 |
| MEME | 95.8 | 94.6 | 95.8 | 95.4 | 96.1 |
| Infra | 75.8 | 77.5 | 80.2 | 73.9 | 82.0 |
| Tokenomist | 96.3 | 96.2 | 95.1 | 95.9 | 95.2 |
| NFT | 91.9 | 90.8 | 87.0 | 91.0 | 89.9 |

Not: SmartContract subjective tüm modellerde ~18%; muhtemelen skorlayıcı/format sorunu, incelenmeli.
