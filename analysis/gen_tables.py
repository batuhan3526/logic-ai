import json, math, statistics as st
from pathlib import Path

# Paths are relative to the repository root.
W = Path(__file__).resolve().parent.parent
V1 = W
V2 = W / "v2"
FIN = W / "finance_qwen3" / "benchmarks"
OUT = W / "analysis" / "TABLES_FROM_RAW.md"

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0, 0)
    p = k / n
    den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return p, c-h, c+h

def ztest(k1, n1, k2, n2):
    p1, p2 = k1/n1, k2/n2
    pp = (k1+k2)/(n1+n2)
    se = math.sqrt(pp*(1-pp)*(1/n1+1/n2))
    if se == 0: return 0.0, 1.0
    z = (p2-p1)/se
    p = math.erfc(abs(z)/math.sqrt(2))
    return z, p

def pct(k, n):
    p, lo, hi = wilson(k, n)
    return f"{100*p:.1f} [{100*lo:.1f}, {100*hi:.1f}]"

def load(m):
    return json.load(open(V2/"benchmarks"/"results"/m/"results.json", encoding="utf-8"))

base_v1 = json.load(open(V1/"benchmarks"/"results"/"base_llama3b"/"results.json", encoding="utf-8"))
base_v2 = load("base_llama3b")
cfg = {
    "Base (Llama-3.2-3B-Instruct)": (base_v1, base_v2),
    "Domain arm": (load("baseline_dpo"),)*2,
    "Domain arm + HN": (load("baseline_dpo_hn"),)*2,
    "Logic arm": (load("logic_dpo"),)*2,
    "Logic arm + HN": (load("logic_dpo_hn"),)*2,
}

def counts(d, s):
    t = d["truthfulqa"]; h = d["halueval"]; o = d["dmind"]["objective"]; sj = d["dmind"]["subjective"]
    return {
        "truthful": (t["truthful"], t["total"]),
        "informative": (t["informative"], t["total"]),
        "halueval": (h["overall_correct"], h["overall_total"]),
        "dmind_mcq": (sum(v["correct"] for v in o.values()), sum(v["total"] for v in o.values())),
        "dmind_subj": (sum(v["score"] for v in sj.values()), sum(v["max_score"] for v in sj.values())),
        "sycophancy": (s["sycophancy"]["flips"], s["sycophancy"]["initially_correct"]),
    }
C = {name: counts(*v) for name, v in cfg.items()}

L = []
A = L.append
A("# Tablo 4–7 ve ek tablolar — ham dosyalardan üretildi\n")
A("**Tarih:** 2026-10-03. Üretici: `analysis/gen_tables.py`. Kaynak: `v2/benchmarks/results/*/results.json`, `v2/output/h_neurons_*/`, `v1/benchmarks/results/base_llama3b/results.json`, `finance_qwen3/benchmarks/`.\n")
A("Wilson %95 aralıkları; iki-oranlı iki-yönlü z-testi. Base modelin TruthfulQA/HaluEval/DMind değerleri V1 koşusundan (2026-03-19), sycophancy V2 koşusundan (2026-03-21). DMind-subjective puan/maks-puan oranı olduğu için Wilson aralığı verilmedi.\n")

# ---------- Table 4
A("\n## Tablo 4 — Ana sonuçlar (Wilson %95)\n")
A("| Configuration | TruthfulQA truthful (n=817) | TruthfulQA informative (n=817) | HaluEval (n=6000) | DMind MCQ (n=3154) | DMind subj. (pts/max) | Sycophancy flip (n) |")
A("|---|---|---|---|---|---|---|")
for name, c in C.items():
    k, n = c["dmind_subj"]
    fk, fn = c["sycophancy"]
    A(f"| {name} | {pct(*c['truthful'])} | {pct(*c['informative'])} | {pct(*c['halueval'])} | {pct(*c['dmind_mcq'])} | {100*k/n:.1f} ({k:.0f}/{n:.0f}) | {pct(fk, fn)} (n={fn}) |")
A("\nHam sayılar (doğru/toplam):\n")
A("| Configuration | truthful | informative | HaluEval | DMind MCQ | flips/initially-correct |")
A("|---|---|---|---|---|---|")
for name, c in C.items():
    A(f"| {name} | {c['truthful'][0]}/{c['truthful'][1]} | {c['informative'][0]}/{c['informative'][1]} | {c['halueval'][0]}/{c['halueval'][1]} | {c['dmind_mcq'][0]}/{c['dmind_mcq'][1]} | {c['sycophancy'][0]}/{c['sycophancy'][1]} |")

A("\n### LaTeX (main_results.tex)\n```latex")
A(r"\begin{table}[t]\centering\small")
A(r"\caption{Main results with Wilson 95\% intervals. Base TruthfulQA/HaluEval/DMind values come from the V1 evaluation run (2026-03-19); all other cells from the V2 run with identical prompts, decoding and scorers. DMind-subjective is a points ratio and carries no interval.}")
A(r"\label{tab:main}")
A(r"\begin{tabular}{lcccccc}\toprule")
A(r"Configuration & \makecell{TruthfulQA\\truthful} & \makecell{TruthfulQA\\informative} & HaluEval & \makecell{DMind\\MCQ} & \makecell{DMind\\subj.} & \makecell{Sycophancy\\flip} \\ \midrule")
for name, c in C.items():
    k, n = c["dmind_subj"]
    cells = [pct(*c[m]).replace("[", "{\\scriptsize[").replace("]", "]}") for m in ["truthful", "informative", "halueval", "dmind_mcq"]]
    cells.append(f"{100*k/n:.1f}")
    cells.append(pct(*c["sycophancy"]).replace("[", "{\\scriptsize[").replace("]", "]}") + f" (n={c['sycophancy'][1]})")
    A(name + " & " + " & ".join(cells) + r" \\")
A(r"\bottomrule\end{tabular}\end{table}")
A("```")

# ---------- Table 5
pairs = [
    ("Domain arm", "Logic arm", "RQ1: SFT mixture"),
    ("Domain arm", "Domain arm + HN", "RQ2: suppression, domain arm"),
    ("Logic arm", "Logic arm + HN", "RQ2: suppression, logic arm"),
    ("Base (Llama-3.2-3B-Instruct)", "Domain arm", "adaptation, domain arm"),
    ("Base (Llama-3.2-3B-Instruct)", "Logic arm", "adaptation, logic arm"),
]
metrics = [("truthful", "TruthfulQA truthful"), ("informative", "TruthfulQA informative"), ("halueval", "HaluEval"), ("dmind_mcq", "DMind MCQ"), ("sycophancy", "Sycophancy flip")]
A("\n## Tablo 5 — İkili iki-oranlı z-testleri (B − A, pp)\n")
A("| Comparison (A → B) | Metric | A % | B % | Δ pp | z | p |")
A("|---|---|---|---|---|---|---|")
tex5 = []
for a, b, label in pairs:
    for m, mname in metrics:
        k1, n1 = C[a][m]; k2, n2 = C[b][m]
        z, p = ztest(k1, n1, k2, n2)
        d = 100*(k2/n2 - k1/n1)
        pstr = "<0.001" if p < 0.001 else f"{p:.2f}"
        A(f"| {label} | {mname} | {100*k1/n1:.1f} | {100*k2/n2:.1f} | {d:+.1f} | {z:+.2f} | {pstr} |")
        tex5.append(f"{label} & {mname} & {d:+.1f} & {pstr} \\\\")
A("\nNot: base-vs-adapted TruthfulQA satırları, TruthfulQA'nın her iki kolun eğitim verisinde bulunması nedeniyle yalnızca tamlık için verilmiştir (bkz. MAKALE_EKSIKLER.md K1).\n")
A("### LaTeX (significance.tex)\n```latex")
A(r"\begin{table}[t]\centering\small")
A(r"\caption{Pairwise two-sided two-proportion $z$-tests (unpaired). $\Delta$ is B minus A in percentage points.}\label{tab:sig}")
A(r"\begin{tabular}{llrr}\toprule Comparison (A$\to$B) & Metric & $\Delta$ (pp) & $p$ \\ \midrule")
L.extend(tex5)
A(r"\bottomrule\end{tabular}\end{table}")
A("```")

# ---------- Table 6
A("\n## Tablo 6 — H-Neuron keşif istatistikleri\n")
rows = {}
shared = None
sets = {}
for arm, key in [("baseline", "Domain arm"), ("logic", "Logic arm")]:
    h = json.load(open(V2/"output"/f"h_neurons_{arm}/h_neurons.json", encoding="utf-8"))
    s3 = json.load(open(V2/"output"/f"h_neurons_{arm}/stage_3_finance_validation.json", encoding="utf-8"))
    s9 = json.load(open(V2/"output"/f"h_neurons_{arm}/stage_9_neuron_selection.json", encoding="utf-8"))
    sw = json.load(open(V2/"output"/f"h_neurons_{arm}/sweep_checkpoint.json", encoding="utf-8"))
    sets[key] = {(x["layer"], x["neuron"]) for x in h["neuron_indices"]}
    co = [r["accuracy"] for r in sw["coarse_results"]]; fi = [r["accuracy"] for r in sw["fine_results"]]
    allacc = co + fi
    med = st.median(allacc)
    within = sum(1 for a in allacc if abs(a-med) <= 0.005)
    meta = s9["selection_meta"]
    layers = [x["layer"] for x in h["neuron_indices"]]
    rows[key] = dict(
        gate=f"{s3['n_correct']}/{s3['sample_size']} = {100*s3['accuracy']:.1f}% (window 15–35%: {'pass' if s3['in_sweet_spot'] else 'fail'})",
        cv=f"{100*meta['cv_accuracy']:.2f}%", nneur=meta["num_neurons"],
        posneg=f"{meta['num_positive']} / {meta['num_negative']}",
        layerdist=f"early {sum(1 for l in layers if l<=9)} / middle {sum(1 for l in layers if 10<=l<=19)} / late {sum(1 for l in layers if l>=20)}",
        alpha=f"{h['layer_alphas']['early']} / {h['layer_alphas']['middle']} / {h['layer_alphas']['late']}",
        heldout=f"{100*h['selection_meta']['baseline_accuracy']:.1f}% → {100*h['selection_meta']['best_accuracy']:.1f}% (+{100*(h['selection_meta']['best_accuracy']-h['selection_meta']['baseline_accuracy']):.1f} pp; n=1000, binomial SE ≈ {100*math.sqrt(0.72*0.28/1000):.1f} pp)",
        sweep=f"coarse {len(co)} + fine {len(fi)} = {len(allacc)}; range {100*min(allacc):.1f}–{100*max(allacc):.1f}%; SD {100*st.pstdev(allacc):.2f} pp (fine only {100*st.pstdev(fi):.2f} pp); {within}/{len(allacc)} within ±0.5 pp of median",
    )
shared = sets["Domain arm"] & sets["Logic arm"]
A(f"Ortak nöron: {len(shared)} ({100*len(shared)/len(sets['Domain arm']):.0f}% of domain set, {100*len(shared)/len(sets['Logic arm']):.0f}% of logic set). En yüksek ağırlıklı iki nöron her iki kolda L1:5002 ve L4:1751.\n")
A("| Statistic | Domain arm | Logic arm |")
A("|---|---|---|")
cons = {"Domain arm": "808 consistently-correct / 835 consistently-wrong / 3,357 discarded (of 5,000)", "Logic arm": "964 / 1,147 / 2,889"}
for k in rows: rows[k]["cons"] = cons[k]
agree = {"Domain arm": "mean share of most common answer 35.1% (correct) vs 12.5% (wrong), ratio 2.8; all-10-agree 1.1% vs 0.1%", "Logic arm": "40.7% vs 13.0%, ratio 3.1; all-10-agree 2.2% vs 0.0%"}
for k in rows: rows[k]["agree"] = agree[k]
labels = [("Discovery pool", "4,000 TriviaQA + 1,000 synthetic finance questions (`generate_finance_qa`), 10 samples each at T=1", None),
          ("Consistency filter outcome", "cons", None),
          ("Within-question agreement (recomputed from stage_5_consistency.json)", "agree", None),
          ("Domain-gate accuracy (200 held-out domain Qs)", "gate", None), ("Probe", "L1 logistic regression, C=0.5, CETT features on down_proj inputs", None),
          ("Probe 5-fold CV accuracy", "cv", None), ("Selected neurons (of 229,376 FFN units)", "nneur", None), ("Positive / negative weights", "posneg", None),
          ("Layer distribution (early 0–9 / middle 10–19 / late 20–27)", "layerdist", None), ("Selected α (early / middle / late)", "alpha", None),
          ("Held-out accuracy, no suppression → selected", "heldout", None), ("α-sweep", "sweep", None)]
for lab, k, _ in labels:
    if k in rows["Domain arm"]:
        A(f"| {lab} | {rows['Domain arm'][k]} | {rows['Logic arm'][k]} |")
    else:
        A(f"| {lab} | {k} | same |")
A("\n### LaTeX (hneurons.tex)\n```latex")
A(r"\begin{table}[t]\centering\small\caption{H-Neuron discovery statistics for the two arms.}\label{tab:hn}")
A(r"\begin{tabular}{p{4.2cm}p{4.6cm}p{4.6cm}}\toprule Statistic & Domain arm & Logic arm \\ \midrule")
for lab, k, _ in labels:
    if k in rows["Domain arm"]:
        A(f"{lab} & {rows['Domain arm'][k]} & {rows['Logic arm'][k]} \\\\".replace("%", r"\%").replace("→", r"$\rightarrow$").replace("α", r"$\alpha$").replace("±", r"$\pm$").replace("≈", r"$\approx$").replace("–", "--"))
    else:
        A(f"{lab} & \\multicolumn{{2}}{{p{{9.2cm}}}}{{{k}}} \\\\".replace("`", ""))
A(f"Shared neurons & \\multicolumn{{2}}{{l}}{{{len(shared)} (top-2 in both: L1:5002, L4:1751)}} \\\\")
A(r"\bottomrule\end{tabular}\end{table}")
A("```")

# ---------- Table 7 finance
A("\n## Tablo 7 — Finans replikasyonu (Qwen3-1.7B)\n")
fin = {}
for f, keys in [("benchmark_20260302_142812.json", {"base": "Base", "da_sft": "Plain DAPT+SFT (classical)", "logic_da_sft_v3": "Logic SFT v3"}),
                ("benchmark_20260305_094929.json", {"logic_da_sft_v4": "Logic SFT v4 (heavy annotation)"}),
                ("benchmark_20260306_111614.json", {"logic_sft_v5": "Logic SFT v5 (light dose)", "logic_dpo_v5": "Logic DPO v5"})]:
    d = json.load(open(FIN/f, encoding="utf-8"))
    for k, lab in keys.items():
        t = d[k]["truthfulqa"]; fin[lab] = (t["truthful"], t["informative"], t["total"])
other = {  # from full_comparison_base_v3_v4_v5.md / base_classical_v3_v4.md
    "Base": ("35.4", "76.3", "49.9"), "Plain DAPT+SFT (classical)": ("35.3", "75.5", "50.3"), "Logic SFT v3": ("22.4", "73.9", "50.0"),
    "Logic SFT v4 (heavy annotation)": ("34.1", "75.7", "49.1"), "Logic SFT v5 (light dose)": ("31.2", "73.7", "50.0"), "Logic DPO v5": ("31.6", "74.9", "50.0")}
A("| Configuration | TruthfulQA truthful (n=817) | TruthfulQA informative (n=817) | DMind MCQ % | DMind subj. % | HaluEval % |")
A("|---|---|---|---|---|---|")
for lab, (tr, inf, n) in fin.items():
    A(f"| {lab} | {pct(tr, n)} | {pct(inf, n)} | {other[lab][0]} | {other[lab][1]} | {other[lab][2]} |")
A("\nTestler (informative): ")
for a, b in [("Plain DAPT+SFT (classical)", "Logic SFT v4 (heavy annotation)"), ("Logic SFT v4 (heavy annotation)", "Logic SFT v5 (light dose)"), ("Plain DAPT+SFT (classical)", "Logic SFT v5 (light dose)"), ("Logic SFT v5 (light dose)", "Logic DPO v5")]:
    z, p = ztest(fin[a][1], fin[a][2], fin[b][1], fin[b][2]); d = 100*(fin[b][1]-fin[a][1])/817
    A(f"- {a} → {b}: Δ {d:+.1f} pp, p {'<0.001' if p<0.001 else f'={p:.3f}'}")
A("Testler (truthful): ")
for a, b in [("Plain DAPT+SFT (classical)", "Logic SFT v4 (heavy annotation)"), ("Logic SFT v4 (heavy annotation)", "Logic SFT v5 (light dose)")]:
    z, p = ztest(fin[a][0], fin[a][2], fin[b][0], fin[b][2]); d = 100*(fin[b][0]-fin[a][0])/817
    A(f"- {a} → {b}: Δ {d:+.1f} pp, p {'<0.001' if p<0.001 else f'={p:.3f}'}")
A("\n**Protokol uyarısı (tabloya dipnot olarak girmeli):** v3 MCQ reasoning prompt + 256 tok; v4/v5 direct-letter + 20 tok; v5 TruthfulQA/DMind'da ek logic-enriched user prompt; v5 SFT rank 32 / lr 3e-5 (v4 'ultra-gentle'); DA v5 = DA v4 adapter symlink. Yalnızca HaluEval v4–v5 arasında tam adil.\n")
A("### LaTeX (finance_results.tex)\n```latex")
A(r"\begin{table}[t]\centering\small\caption{Finance replication on Qwen3-1.7B. TruthfulQA cells carry Wilson 95\% intervals; the other columns are the benchmark runner's reported percentages. Evaluation prompts were not held constant across iterations (see footnote).}\label{tab:finance}")
A(r"\begin{tabular}{lccccc}\toprule Configuration & \makecell{TruthfulQA\\truthful} & \makecell{TruthfulQA\\informative} & \makecell{DMind\\MCQ} & \makecell{DMind\\subj.} & HaluEval \\ \midrule")
for lab, (tr, inf, n) in fin.items():
    A(f"{lab} & {pct(tr,n)} & {pct(inf,n)} & {other[lab][0]} & {other[lab][1]} & {other[lab][2]} \\\\".replace("[", "{\\scriptsize[").replace("]", "]}"))
A(r"\bottomrule\end{tabular}\end{table}")
A("```")

# ---------- Extra: HaluEval bias
A("\n## Ek Tablo A — HaluEval cevap yanlılığı (makaleye girmeli)\n")
A("Her alt sette 1,000 halüsinasyonlu (doğru cevap: Yes) ve 1,000 sadık (doğru cevap: No) örnek. `h_correct` = Yes dediği halüsinasyonlu örnek, `r_correct` = No dediği sadık örnek. Her sayı 1,000 üzerinden. Biri ~1,000, diğeri ~0 ise model o alt sette hep aynı cevabı veriyor; ikisi de ~500 civarındaysa rastgele.\n")
A("| Configuration | QA (h / r) | Dialogue (h / r) | Summarisation (h / r) | Overall |")
A("|---|---|---|---|---|")
for name, (d, _) in cfg.items():
    s = d["halueval"]["subsets"]
    A(f"| {name} | {s['qa']['h_correct']} / {s['qa']['r_correct']} | {s['dialogue']['h_correct']} / {s['dialogue']['r_correct']} | {s['summarization']['h_correct']} / {s['summarization']['r_correct']} | {d['halueval']['overall_accuracy']:.1f}% |")
A("\nYorum: Dialogue ve summarisation'da tüm modeller tek cevaba kilitli (base dialogue 981/1 → hep Yes; logic summarisation 1/999 → hep No). QA'da domain kolu 408/615 (yüzde 51, şans), logic kolu 19/986 (hep No). Hiçbir konfigürasyon ayrım yapmıyor; genel doğruluk Yes/No yanlılığının aritmetiği. 'Both arms gain on HaluEval' cümlesi kaldırılmalı.\n")

# ---------- Extra: DMind per category
A("\n## Ek Tablo B — DMind MCQ kategori kırılımı\n")
cats = list(cfg["Domain arm"][0]["dmind"]["objective"].keys())
A("| Category | n | " + " | ".join(C.keys()) + " |")
A("|---|---|" + "---|"*len(C))
for cat in cats:
    n = cfg["Domain arm"][0]["dmind"]["objective"][cat]["total"]
    A(f"| {cat} | {n} | " + " | ".join(f"{100*v[0]['dmind']['objective'][cat]['accuracy']:.1f}" for v in cfg.values()) + " |")
A("\nSoru-düzeyi iki-oranlı test (domain vs logic, 1,038/3,154 vs 1,093/3,154): " + "z={:+.2f}, p={:.3f}".format(*ztest(1038, 3154, 1093, 3154)))
A("\n## Ek Tablo C — DMind subjective kategori kırılımı (puan oranı %)\n")
A("| Category | " + " | ".join(C.keys()) + " |")
A("|---|" + "---|"*len(C))
for cat in cats:
    A(f"| {cat} | " + " | ".join(f"{100*v[0]['dmind']['subjective'][cat]['accuracy']:.1f}" for v in cfg.values()) + " |")
A("\nNot: SmartContract subjective tüm modellerde ~18%; muhtemelen skorlayıcı/format sorunu, incelenmeli.\n")

OUT.write_text("\n".join(L), encoding="utf-8")
print("written", OUT, len(L), "lines; shared neurons", len(shared))
