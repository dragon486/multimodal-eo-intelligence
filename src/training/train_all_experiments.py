"""Triad Experiment Runner (Option B).

Trains and evaluates all three benchmark models sequentially:
1. E1: Optical-Only Baseline (Sentinel-2 6 bands)
2. E4: SAR Radar-Only Baseline (Sentinel-1 VV/VH)
3. E5: Proposed Multimodal Cross-Attention Fused Model (Optical + SAR)

Full Checkpoint & Power-Outage Resilience:
- Automatically resumes any experiment from its last completed epoch
- Automatically skips already completed experiments
- Produces the comprehensive ablation comparison table for reports and publications
"""

import sys
import json
from pathlib import Path
import torch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.train_flood import run_experiment


def run_all_triad_experiments(epochs: int = 10, batch_size: int = 2):
    experiments = [
        {"exp_id": "E1", "mode": "optical_only", "desc": "Optical-Only Baseline (Sentinel-2)"},
        {"exp_id": "E4", "mode": "sar_only", "desc": "Radar-Only Baseline (Sentinel-1 SAR)"},
        {"exp_id": "E5", "mode": "multimodal", "desc": "Proposed Multimodal Fused (Optical + SAR)"},
    ]

    base_dir = Path(__file__).resolve().parents[2]
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_json = reports_dir / "experiment_triad_summary.json"

    # Load existing summary if partially completed
    results = {}
    if summary_json.exists():
        try:
            with open(summary_json, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}

    print("\n" + "=" * 70)
    print("STARTING FULL EXPERIMENT TRIAD (OPTION B: E1, E4, E5)")
    print("Resilience: Auto-resumes from last epoch if interrupted or powered off")
    print("=" * 70)

    for item in experiments:
        exp_key = f"{item['exp_id']}_{item['mode']}"
        last_ckpt = base_dir / "checkpoints" / f"last_{item['exp_id'].lower()}_{item['mode']}.pt"

        # Check if already completed all epochs
        if last_ckpt.exists():
            try:
                ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
                if ckpt.get("epoch", 0) >= epochs and exp_key in results:
                    print(f"\n[ALREADY COMPLETE] {item['exp_id']} ({item['desc']}) completed all {epochs} epochs. Skipping to next.")
                    continue
            except Exception:
                pass

        print(f"\n>>> Next in Queue: {item['exp_id']} - {item['desc']}")

        res = run_experiment(
            exp_id=item["exp_id"],
            mode=item["mode"],
            epochs=epochs,
            batch_size=batch_size,
            resume=True,
            use_mock=False,
        )

        results[exp_key] = res

        # Save cumulative results
        with open(summary_json, "w") as f:
            json.dump(results, f, indent=2)

    # Generate Markdown Ablation Table
    table_md = reports_dir / "benchmark_table.md"
    with open(table_md, "w") as f:
        f.write("# Multimodal Remote Sensing Benchmark Ablation Matrix\n\n")
        f.write("| Experiment ID | Modalities Used | Test Water IoU | Test Water F1 | Test Mean IoU | Bolivia Held-Out IoU |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: |\n")

        for item in experiments:
            exp_key = f"{item['exp_id']}_{item['mode']}"
            r = results.get(exp_key, {})
            test_res = r.get("test_results", {}).get("test", {})
            bolivia_res = r.get("test_results", {}).get("bolivia", {})

            w_iou = f"{test_res.get('water_iou', 0) * 100:.2f}%" if test_res else "N/A"
            w_f1 = f"{test_res.get('water_f1', 0) * 100:.2f}%" if test_res else "N/A"
            m_iou = f"{test_res.get('mean_iou', 0) * 100:.2f}%" if test_res else "N/A"
            b_iou = f"{bolivia_res.get('water_iou', 0) * 100:.2f}%" if bolivia_res else "N/A"

            f.write(f"| **{item['exp_id']}** | {item['desc']} | {w_iou} | {w_f1} | {m_iou} | {b_iou} |\n")

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS IN TRIAD COMPLETED!")
    print(f"Summary JSON saved: {summary_json}")
    print(f"Benchmark Markdown table saved: {table_md}")
    print("=" * 70)


if __name__ == "__main__":
    epochs = 10
    batch_size = 2
    if len(sys.argv) > 1:
        try:
            epochs = int(sys.argv[1])
        except ValueError:
            pass
    run_all_triad_experiments(epochs=epochs, batch_size=batch_size)
