#!/usr/bin/env python3
"""Over-Compliance Analysis — Stage 10 of V2 Pipeline.

Combines FalseQA, FaithEval, and Sycophancy benchmark results into
a unified over-compliance report. Compares H-Neurons before/after
and cross-references with V1 H-Neurons findings.

Usage:
    python scripts/run_overcompliance.py

Outputs:
    benchmarks/results/overcompliance_analysis.json
"""
import json
import sys
from pathlib import Path

BASE_DIR = Path("/home/whiteowl/last_shot/v2")
V1_DIR = Path("/home/whiteowl/last_shot")
RESULTS_DIR = BASE_DIR / "benchmarks" / "results"

# Models to analyze
MODELS = {
    "base": "base_llama3b",
    "baseline_dpo": "baseline_dpo",
    "logic_dpo": "logic_dpo",
    "baseline_dpo_hn": "baseline_dpo_hn",
    "logic_dpo_hn": "logic_dpo_hn",
}

# Over-compliance benchmarks
OC_BENCHMARKS = ["falseqa", "faitheval", "sycophancy"]


def load_results(model_dir: str) -> dict | None:
    """Load results.json for a model."""
    path = RESULTS_DIR / model_dir / "results.json"
    if not path.exists():
        # Try V1 results for base model
        v1_path = V1_DIR / "benchmarks" / "results" / model_dir / "results.json"
        if v1_path.exists():
            with open(v1_path) as f:
                return json.load(f)
        print(f"  [WARN] No results found for {model_dir}")
        return None
    with open(path) as f:
        return json.load(f)


def extract_compliance_rate(results: dict, benchmark: str) -> float | None:
    """Extract compliance rate for a specific benchmark."""
    if results is None:
        return None

    bm = results.get(benchmark, {})

    if benchmark == "falseqa":
        # compliance_rate = 1 - correction_rate
        correction_rate = bm.get("correction_rate")
        if correction_rate is not None:
            return 1.0 - correction_rate
        return bm.get("compliance_rate")

    elif benchmark == "faitheval":
        # compliance_rate = 1 - faithfulness_rate
        faithfulness_rate = bm.get("faithfulness_rate")
        if faithfulness_rate is not None:
            return 1.0 - faithfulness_rate
        return bm.get("compliance_rate")

    elif benchmark == "sycophancy":
        # flip_rate IS the compliance rate
        return bm.get("flip_rate") or bm.get("compliance_rate")

    return None


def compute_overcompliance_score(rates: dict) -> float | None:
    """Compute unified over-compliance score (average of 3 rates)."""
    valid = [v for v in rates.values() if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def load_h_neurons(path: str) -> dict | None:
    """Load H-Neurons JSON."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def compare_h_neurons_v1_v2(v1_neurons: dict, v2_neurons: dict) -> dict:
    """Compare H-Neurons between V1 and V2."""
    if not v1_neurons or not v2_neurons:
        return {"comparison_available": False}

    v1_set = set()
    for n in v1_neurons.get("neurons", []):
        v1_set.add((n.get("layer"), n.get("neuron")))

    v2_set = set()
    for n in v2_neurons.get("neurons", []):
        v2_set.add((n.get("layer"), n.get("neuron")))

    shared = v1_set & v2_set
    v1_only = v1_set - v2_set
    v2_only = v2_set - v1_set

    return {
        "comparison_available": True,
        "v1_neuron_count": len(v1_set),
        "v2_neuron_count": len(v2_set),
        "shared_count": len(shared),
        "shared_ratio": len(shared) / max(len(v1_set), 1),
        "shared_neurons": sorted([{"layer": l, "neuron": n} for l, n in shared],
                                  key=lambda x: (x["layer"], x["neuron"])),
        "v1_only_count": len(v1_only),
        "v2_only_count": len(v2_only),
    }


def main():
    marker = RESULTS_DIR / ".overcompliance_analysis_complete"
    if marker.exists():
        print("[SKIP] Over-compliance analysis already complete")
        return

    print("=" * 60)
    print("  OVER-COMPLIANCE ANALYSIS (Stage 10)")
    print("=" * 60)

    # ---- 1. Collect compliance rates per model per benchmark ----
    analysis = {"models": {}}

    for model_label, model_dir in MODELS.items():
        print(f"\n  Analyzing {model_label}...")
        results = load_results(model_dir)

        rates = {}
        for bm in OC_BENCHMARKS:
            rate = extract_compliance_rate(results, bm)
            rates[bm] = rate
            status = f"{rate:.4f}" if rate is not None else "N/A"
            print(f"    {bm}: {status}")

        oc_score = compute_overcompliance_score(rates)
        print(f"    Over-Compliance Score: {oc_score:.4f}" if oc_score else "    Over-Compliance Score: N/A")

        analysis["models"][model_label] = {
            "compliance_rates": rates,
            "overcompliance_score": oc_score,
        }

    # ---- 2. H-Neurons before/after comparison ----
    print("\n  H-Neurons Impact on Over-Compliance:")

    for pipeline in ["baseline", "logic"]:
        dpo_label = f"{pipeline}_dpo"
        hn_label = f"{pipeline}_dpo_hn"

        dpo_data = analysis["models"].get(dpo_label, {})
        hn_data = analysis["models"].get(hn_label, {})

        dpo_score = dpo_data.get("overcompliance_score")
        hn_score = hn_data.get("overcompliance_score")

        if dpo_score is not None and hn_score is not None:
            delta = hn_score - dpo_score
            direction = "↓ (improved)" if delta < 0 else "↑ (worsened)" if delta > 0 else "= (no change)"
            print(f"    {pipeline}: {dpo_score:.4f} → {hn_score:.4f} ({delta:+.4f} {direction})")

            analysis["models"][hn_label]["hn_delta"] = {
                "delta": delta,
                "improved": delta < 0,
            }

    # ---- 3. V1 vs V2 H-Neurons comparison ----
    print("\n  V1 vs V2 H-Neurons Comparison:")

    hn_comparisons = {}
    for pipeline in ["baseline", "logic"]:
        v1_path = V1_DIR / "output" / f"h_neurons_{pipeline}" / "h_neurons.json"
        v2_path = BASE_DIR / "output" / f"h_neurons_{pipeline}" / "h_neurons.json"

        v1_hn = load_h_neurons(str(v1_path))
        v2_hn = load_h_neurons(str(v2_path))

        comparison = compare_h_neurons_v1_v2(v1_hn, v2_hn)
        hn_comparisons[pipeline] = comparison

        if comparison.get("comparison_available"):
            print(f"    {pipeline}: V1={comparison['v1_neuron_count']} neurons, "
                  f"V2={comparison['v2_neuron_count']} neurons, "
                  f"Shared={comparison['shared_count']} ({comparison['shared_ratio']:.1%})")
        else:
            print(f"    {pipeline}: comparison not available (missing data)")

    analysis["h_neurons_v1_v2_comparison"] = hn_comparisons

    # ---- 4. Summary table ----
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    header = f"{'Model':<20} {'FalseQA':>10} {'FaithEval':>10} {'Sycophancy':>10} {'OC Score':>10}"
    print(f"\n  {header}")
    print(f"  {'-' * len(header)}")

    for model_label in MODELS:
        data = analysis["models"].get(model_label, {})
        rates = data.get("compliance_rates", {})
        oc = data.get("overcompliance_score")

        vals = []
        for bm in OC_BENCHMARKS:
            r = rates.get(bm)
            vals.append(f"{r:.4f}" if r is not None else "N/A")
        oc_str = f"{oc:.4f}" if oc is not None else "N/A"

        print(f"  {model_label:<20} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {oc_str:>10}")

    # ---- 5. Save results ----
    output_path = RESULTS_DIR / "overcompliance_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to {output_path}")

    # Mark complete
    marker.touch()
    print("\n[DONE] Over-compliance analysis complete")


if __name__ == "__main__":
    main()
